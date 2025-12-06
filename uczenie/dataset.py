#!/usr/bin/env python3
"""
Dataset COCO dla detekcji obiektów na taśmie górniczej.
Kompatybilny z torchvision Faster R-CNN.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ConveyorBeltDataset(Dataset):
    """
    Dataset dla detekcji obiektów na taśmie górniczej.
    Ładuje anotacje w formacie COCO.
    """
    
    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        transforms=None,
    ):
        """
        Args:
            annotations_file: Ścieżka do pliku COCO JSON
            images_dir: Katalog z obrazami
            transforms: Transformacje (albumentations lub torchvision)
        """
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        
        # Wczytaj anotacje COCO
        with open(annotations_file, 'r') as f:
            self.coco = json.load(f)
        
        # Zbuduj mapowania
        self.images = {img['id']: img for img in self.coco['images']}
        self.categories = {cat['id']: cat for cat in self.coco['categories']}
        
        # Grupuj anotacje po image_id
        self.image_annotations = {}
        for ann in self.coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Lista ID obrazów z anotacjami
        self.image_ids = list(self.images.keys())
        
        # Mapowanie kategorii (COCO może mieć ID nieciągłe)
        self.cat_id_to_continuous = {
            cat_id: idx + 1  # 0 zarezerwowane dla background
            for idx, cat_id in enumerate(sorted(self.categories.keys()))
        }
        
        print(f"Załadowano dataset: {len(self.image_ids)} obrazów")
        print(f"Kategorie: {[self.categories[k]['name'] for k in sorted(self.categories.keys())]}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Wczytaj obraz
        img_path = self.images_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pobierz anotacje
        annotations = self.image_annotations.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []
        
        img_h, img_w = image.shape[:2]
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Konwertuj na [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            
            # Mapuj kategorię na ciągły indeks
            labels.append(self.cat_id_to_continuous[ann['category_id']])
            
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

            # Generuj maskę z poligonów
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for seg in ann.get('segmentation', []):
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
            masks.append(mask)
        
        # Konwertuj na tensory
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Brak anotacji - puste tensory
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_h, img_w), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
        }
        
        # Konwertuj obraz na tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Zastosuj transformacje
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def get_num_classes(self) -> int:
        """Zwraca liczbę klas (włącznie z background)."""
        return len(self.categories) + 1
    
    def get_category_names(self) -> List[str]:
        """Zwraca nazwy kategorii."""
        return ['background'] + [
            self.categories[k]['name']
            for k in sorted(self.categories.keys())
        ]


def collate_fn(batch):
    """
    Collate function dla DataLoader.
    Obrazy i targety mają różne rozmiary, więc zwracamy listy.
    """
    return tuple(zip(*batch))


def get_dataloaders(
    train_annotations: str,
    val_annotations: str,
    images_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
):
    """
    Tworzy DataLoadery dla treningu i walidacji.
    """
    from torch.utils.data import DataLoader
    
    train_dataset = ConveyorBeltDataset(train_annotations, images_dir)
    val_dataset = ConveyorBeltDataset(val_annotations, images_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, train_dataset.get_num_classes()


if __name__ == "__main__":
    # Test datasetu
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    dataset = ConveyorBeltDataset(
        annotations_file="annotations/coco_annotations.json",
        images_dir="data/training_images",
    )
    
    print(f"\nLiczba próbek: {len(dataset)}")
    print(f"Liczba klas: {dataset.get_num_classes()}")
    print(f"Nazwy klas: {dataset.get_category_names()}")
    
    # Test jednej próbki
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"\nPróbka 0:")
        print(f"  Obraz: {img.shape}")
        print(f"  Boxes: {target['boxes'].shape}")
        print(f"  Labels: {target['labels']}")
