#!/usr/bin/env python3
"""
Skrypt inferencji dla wytrenowanego modelu Faster R-CNN.
Wykrywa obiekty na obrazach lub wideo.

Użycie:
    python src/inference.py --model models/best_model.pth --image test.jpg
    python src/inference.py --model models/best_model.pth --video video.mp4
    python src/inference.py --model models/best_model.pth --frames-dir data/frames/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Dodaj katalog główny projektu do sys.path, aby importy 'src...' działały
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.visualization.drawing import draw_boxes_bgr


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, List[str]]:
    """
    Ładuje wytrenowany model.
    
    Returns:
        Model i lista nazw klas
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Pobierz metadane
    metadata = checkpoint.get('metadata', {})
    num_classes = metadata.get('num_classes', 3)  # domyślnie 3 (bg + 2 klasy)
    class_names = metadata.get('class_names', ['background', 'szew', 'tasma'])
    
    # Utwórz model
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Podmień mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    # Załaduj wagi
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Załadowano model: {checkpoint_path}")
    print(f"Klasy: {class_names}")
    
    return model, class_names


@torch.no_grad()
def detect_objects(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    confidence_threshold: float = 0.5,
) -> dict:
    """
    Wykrywa obiekty na obrazie.
    
    Args:
        model: Wytrenowany model
        image: Obraz BGR (OpenCV)
        device: Device (cuda/cpu)
        confidence_threshold: Próg pewności
    
    Returns:
        Słownik z detekcjami
    """
    # Konwertuj obraz
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Inferencja
    outputs = model(image_tensor)
    
    # Filtruj po confidence
    output = outputs[0]
    mask = output['scores'] >= confidence_threshold
    
    return {
        'boxes': output['boxes'][mask].cpu().numpy(),
        'labels': output['labels'][mask].cpu().numpy(),
        'scores': output['scores'][mask].cpu().numpy(),
        'masks': output['masks'][mask].cpu().numpy() if 'masks' in output else None,
    }




def process_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: List[str],
    device: torch.device,
    output_path: Path = None,
    confidence_threshold: float = 0.5,
    show: bool = False,
):
    """Przetwarza pojedynczy obraz."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Nie można wczytać: {image_path}")
        return
    
    # Detekcja
    detections = detect_objects(model, image, device, confidence_threshold)
    
    print(f"\n{image_path.name}: {len(detections['boxes'])} detekcji")
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        class_name = class_names[label] if label < len(class_names) else f'class_{label}'
        print(f"  - {class_name}: {score:.2f}, box={box.astype(int)}")
    
    # Rysuj
    # draw_boxes_bgr wymaga tensorów lub numpy, ale moja implementacja draw_boxes_bgr obsługuje numpy
    # Uwaga: draw_boxes_bgr modyfikuje obraz w miejscu w oryginale usera? 
    # Nie, user function `return img_bgr` i rysuje na nim. Ale lepiej przekazać kopię, jeśli chcemy zachować oryginał.
    # W process_image wczytuję image z pliku.
    result = draw_boxes_bgr(
        image.copy(),
        detections['boxes'],
        detections['labels'],
        detections['scores'],
        class_names,
        confidence_threshold
    )
    
    # Zapisz
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        print(f"Zapisano: {output_path}")
    
    # Pokaż
    if show:
        cv2.imshow("Detekcje", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result, detections


def process_video(
    model: torch.nn.Module,
    video_path: Path,
    class_names: List[str],
    device: torch.device,
    output_path: Path = None,
    confidence_threshold: float = 0.5,
    frame_skip: int = 1,
    max_frames: int = None,
):
    """Przetwarza wideo klatka po klatce."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Nie można otworzyć: {video_path}")
        return
    
    # Właściwości wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nWideo: {video_path.name}")
    print(f"Rozdzielczość: {width}x{height}, FPS: {fps}, Klatek: {total_frames}")
    
    # Writer dla wyjścia
    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps / frame_skip, (width, height))
    
    frame_idx = 0
    processed = 0
    all_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and processed >= max_frames:
            break
        
        if frame_idx % frame_skip == 0:
            # Detekcja
            detections = detect_objects(model, frame, device, confidence_threshold)
            
            # Rysuj
            result = draw_boxes_bgr(
                frame.copy(),
                detections['boxes'],
                detections['labels'],
                detections['scores'],
                class_names,
                confidence_threshold
            )
            
            # Zapisz
            if writer:
                writer.write(result)
            
            # Log
            all_detections.append({
                'frame': frame_idx,
                'num_detections': len(detections['boxes']),
                'detections': [
                    {
                        'class': class_names[l] if l < len(class_names) else f'class_{l}',
                        'score': float(s),
                        'box': b.tolist(),
                    }
                    for b, l, s in zip(detections['boxes'], detections['labels'], detections['scores'])
                ]
            })
            
            processed += 1
            if processed % 10 == 0:
                print(f"Przetworzono: {processed} klatek, bieżąca: {len(detections['boxes'])} detekcji")
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
        print(f"\nZapisano wideo: {output_path}")
    
    return all_detections


def process_frames_dir(
    model: torch.nn.Module,
    frames_dir: Path,
    class_names: List[str],
    device: torch.device,
    output_dir: Path = None,
    confidence_threshold: float = 0.5,
    max_frames: int = None,
):
    """Przetwarza katalog z klatkami."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in frames_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if max_frames:
        image_files = image_files[:max_frames]
    
    print(f"\nPrzetwarzam {len(image_files)} obrazów z {frames_dir}")
    
    all_results = []
    for img_path in image_files:
        output_path = output_dir / img_path.name if output_dir else None
        result, detections = process_image(
            model, img_path, class_names, device, output_path, confidence_threshold
        )
        all_results.append({
            'image': img_path.name,
            'detections': detections,
        })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Inferencja Faster R-CNN")
    parser.add_argument("--model", type=Path, required=True, help="Ścieżka do checkpointu")
    parser.add_argument("--image", type=Path, help="Pojedynczy obraz")
    parser.add_argument("--video", type=Path, help="Plik wideo")
    parser.add_argument("--frames-dir", type=Path, help="Katalog z klatkami")
    parser.add_argument("--output", type=Path, help="Ścieżka wyjściowa")
    parser.add_argument("--confidence", type=float, default=0.5, help="Próg pewności")
    parser.add_argument("--frame-skip", type=int, default=1, help="Co którą klatkę (dla wideo)")
    parser.add_argument("--max-frames", type=int, help="Max klatek do przetworzenia")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, auto")
    parser.add_argument("--show", action="store_true", help="Pokaż wynik")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Załaduj model
    model, class_names = load_model(str(args.model), device)
    
    # Przetwarzaj
    if args.image:
        process_image(model, args.image, class_names, device, args.output, args.confidence, args.show)
    elif args.video:
        process_video(model, args.video, class_names, device, args.output, args.confidence, args.frame_skip, args.max_frames)
    elif args.frames_dir:
        process_frames_dir(model, args.frames_dir, class_names, device, args.output, args.confidence, args.max_frames)
    else:
        print("Podaj --image, --video lub --frames-dir")


if __name__ == "__main__":
    main()
