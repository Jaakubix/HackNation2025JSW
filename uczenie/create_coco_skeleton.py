#!/usr/bin/env python3
"""
Skrypt do tworzenia szkieletu pliku anotacji COCO.
Tworzy początkową strukturę JSON z listą obrazów (bez anotacji).

Użycie:
    python src/create_coco_skeleton.py --frames-dir data/frames/video_name --output annotations/images.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASSES, ANNOTATIONS_DIR, FRAMES_DIR


def create_coco_skeleton(
    frames_dir: Path,
    output_path: Path,
    description: str = "Taśma górnicza - detekcja uszkodzeń"
) -> dict:
    """
    Tworzy szkielet pliku COCO JSON z listą obrazów.
    
    Args:
        frames_dir: Katalog z klatkami
        output_path: Ścieżka do wyjściowego pliku JSON
        description: Opis zbioru danych
    
    Returns:
        Słownik z strukturą COCO
    """
    # Znajdź wszystkie obrazy
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in frames_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(f"Nie znaleziono obrazów w: {frames_dir}")
        return None
    
    print(f"Znaleziono {len(image_files)} obrazów")
    
    # Struktura COCO
    coco = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Kopalnia Project",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Internal Use Only",
                "url": ""
            }
        ],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Dodaj kategorie (bez background)
    for class_id, class_name in CLASSES.items():
        if class_id == 0:  # Pomiń background
            continue
        coco["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "tasma" if "tasmy" in class_name or "laczenie" in class_name else "defekt"
        })
    
    # Dodaj obrazy
    for idx, image_path in enumerate(tqdm(image_files, desc="Przetwarzanie obrazów")):
        # Pobierz wymiary obrazu
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Ostrzeżenie: nie można wczytać {image_path}")
            continue
        
        height, width = img.shape[:2]
        
        # Ścieżka względna
        try:
            relative_path = image_path.relative_to(frames_dir)
        except ValueError:
            relative_path = image_path.name
        
        coco["images"].append({
            "id": idx + 1,
            "file_name": str(relative_path),
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat(),
        })
    
    # Utwórz katalog wyjściowy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Zapisz plik
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Utworzono szkielet COCO: {output_path}")
    print(f"  Obrazów: {len(coco['images'])}")
    print(f"  Kategorii: {len(coco['categories'])}")
    print(f"\nNastępny krok: Zaimportuj ten plik do Label Studio lub CVAT do anotacji.")
    
    return coco


def main():
    parser = argparse.ArgumentParser(description="Tworzenie szkieletu COCO JSON")
    parser.add_argument("-f", "--frames-dir", type=Path, default=FRAMES_DIR,
                        help=f"Katalog z klatkami (domyślnie: {FRAMES_DIR})")
    parser.add_argument("-o", "--output", type=Path, 
                        default=ANNOTATIONS_DIR / "coco_skeleton.json",
                        help="Ścieżka wyjściowa pliku JSON")
    parser.add_argument("-d", "--description", type=str,
                        default="Taśma górnicza - detekcja uszkodzeń",
                        help="Opis zbioru danych")
    
    args = parser.parse_args()
    
    if not args.frames_dir.exists():
        print(f"Błąd: Katalog nie istnieje: {args.frames_dir}")
        sys.exit(1)
    
    create_coco_skeleton(args.frames_dir, args.output, args.description)


if __name__ == "__main__":
    main()
