#!/usr/bin/env python3
"""
Skrypt treningowy dla Faster R-CNN ResNet50 FPN.
Detekcja obiektów na taśmie górniczej.

Użycie:
    python src/train.py --annotations annotations/coco_annotations.json --images data/training_images
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Dodaj ścieżkę projektu
sys.path.insert(0, str(Path(__file__).parent.parent))

from uczenie.dataset import ConveyorBeltDataset, collate_fn
import config


def get_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Tworzy model Mask R-CNN ResNet50 FPN.
    
    Args:
        num_classes: Liczba klas (włącznie z background)
        pretrained: Czy użyć pretrenowanych wag
    
    Returns:
        Model Mask R-CNN
    """
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
    else:
        model = maskrcnn_resnet50_fpn(weights=None)
    
    # 1. Podmień głowicę klasyfikacyjną (box predictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 2. Podmień głowicę masek (mask predictor)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def freeze_backbone(model: torch.nn.Module):
    """Zamraża backbone (ResNet), trenuje tylko head."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone zamrożony - trenuję tylko head")


def unfreeze_backbone(model: torch.nn.Module):
    """Odmraża backbone do fine-tuningu."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("Backbone odblokowany - fine-tuning całego modelu")


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Trenuje model przez jedną epokę.
    
    Returns:
        Średni loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        # Przenieś na device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Ewaluacja modelu na zbiorze walidacyjnym.
    
    Returns:
        Średni loss
    """
    model.train()  # Faster R-CNN wymaga train() do obliczenia loss
    total_loss = 0.0
    num_batches = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
    metadata: dict = None,
    json_data: dict = None,
):
    """Zapisuje checkpoint modelu oraz opcjonalny plik JSON z metadanymi."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metadata': metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"Zapisano checkpoint: {path}")

    # Zapisz JSON jeśli podano
    if json_data:
        json_path = path.with_suffix('.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Zapisano metadane: {json_path}")
        except Exception as e:
            print(f"Błąd zapisu metadanych JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="Trening Faster R-CNN")
    parser.add_argument("--annotations", type=Path, default=config.ANNOTATIONS_FILE,
                        help=f"Plik anotacji COCO JSON (domyślnie: {config.ANNOTATIONS_FILE})")
    parser.add_argument("--val-annotations", type=Path, default=None,
                        help="Plik anotacji COCO JSON (val) - opcjonalny")
    parser.add_argument("--images", type=Path, default=config.IMAGES_DIR,
                        help=f"Katalog z obrazami (domyślnie: {config.IMAGES_DIR})")
    parser.add_argument("--output", type=Path, default=config.TRAINING["checkpoint_dir"],
                        help=f"Katalog na modele (domyślnie: {config.TRAINING['checkpoint_dir']})")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Liczba epok")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Proporcja danych walidacyjnych")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epoki z zamrożonym backbone")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, auto")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Używam urządzenia: {device}")
    
    # Katalog na checkpointy
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    print("\n=== Ładowanie danych ===")
    
    if args.val_annotations:
        print("Używam oddzielnych plików train/val")
        train_dataset = ConveyorBeltDataset(
            annotations_file=str(args.annotations),
            images_dir=str(args.images),
        )
        val_dataset = ConveyorBeltDataset(
            annotations_file=str(args.val_annotations),
            images_dir=str(args.images),
        )
        full_dataset = train_dataset # Dla pobrania klas
    else:
        print("Używam jednego pliku i losowego podziału")
        full_dataset = ConveyorBeltDataset(
            annotations_file=str(args.annotations),
            images_dir=str(args.images),
        )
        
        # Podział na train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    num_classes = full_dataset.get_num_classes()
    print(f"Liczba klas: {num_classes}")
    print(f"Klasy: {full_dataset.get_category_names()}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoadery
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # Model
    print("\n=== Tworzenie modelu ===")
    model = get_model(num_classes, pretrained=True)
    model.to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Konfiguracja do zapisu w JSON
    config_info = {
        'model_type': 'Mask R-CNN ResNet50 FPN',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': type(optimizer).__name__,
        'val_split': args.val_split,
        'freeze_epochs': args.freeze_epochs,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
        'augmentation': config.AUGMENTATION['enabled'],
    }
    
    # Trening
    print("\n=== Rozpoczynam trening ===")
    best_val_loss = float('inf')
    training_log = []
    
    # Faza 1: Zamrożony backbone
    if args.freeze_epochs > 0:
        freeze_backbone(model)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoka {epoch}/{args.epochs} ---")
        
        # Odblokuj backbone po freeze_epochs
        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            # Zmniejsz LR dla fine-tuningu
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
            print(f"LR zmniejszony do: {args.lr * 0.1}")
        
        # Trening
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        train_time = time.time() - start_time
        
        # Walidacja
        val_loss = evaluate(model, val_loader, device) if len(val_dataset) > 0 else 0.0
        
        # Scheduler step
        scheduler.step()
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': train_time,
        }
        training_log.append(log_entry)
        
        print(f"Epoka {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"LR={log_entry['lr']:.6f}, Czas={train_time:.1f}s")
        
        # Zapisz najlepszy model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                args.output / "best_model.pth",
                metadata={
                    'num_classes': num_classes,
                    'class_names': full_dataset.get_category_names(),
                },
                json_data={
                    'config': config_info,
                    'stats': {'epoch': epoch, 'val_loss': val_loss, 'train_loss': train_loss},
                    'training_log': training_log
                }
            )
        
        # Zapisz checkpoint co 10 epok
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                args.output / f"checkpoint_epoch{epoch}.pth",
                json_data={
                    'config': config_info,
                    'stats': {'epoch': epoch, 'val_loss': val_loss, 'train_loss': train_loss},
                    'training_log': training_log
                }
            )
    
    # Zapisz ostatni model i log
    save_checkpoint(
        model, optimizer, args.epochs, val_loss,
        args.output / "final_model.pth",
        metadata={
            'num_classes': num_classes,
            'class_names': full_dataset.get_category_names(),
            'training_log': training_log,
        },
        json_data={
            'config': config_info,
            'stats': {'epoch': args.epochs, 'val_loss': val_loss, 'train_loss': train_loss},
            'training_log': training_log
        }
    )
    
    # Zapisz log treningowy
    log_path = args.output / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\nZapisano log treningowy: {log_path}")
    
    print("\n=== Trening zakończony! ===")
    print(f"Najlepszy val_loss: {best_val_loss:.4f}")
    print(f"Modele zapisane w: {args.output}")


if __name__ == "__main__":
    main()
