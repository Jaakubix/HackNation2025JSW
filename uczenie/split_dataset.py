#!/usr/bin/env python3
"""
Skrypt do podziału zbioru danych COCO na train/val/test.
Zachowuje równomierny rozkład kategorii między zbiorami.

Użycie:
    python src/split_dataset.py --input annotations/annotated.json --output-dir annotations/
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPLIT_RATIOS, ANNOTATIONS_DIR


def split_coco_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Dzieli zbiór COCO na train/val/test z zachowaniem stratyfikacji.
    
    Args:
        input_path: Ścieżka do pliku COCO JSON z anotacjami
        output_dir: Katalog wyjściowy
        train_ratio: Proporcja zbioru treningowego
        val_ratio: Proporcja zbioru walidacyjnego  
        test_ratio: Proporcja zbioru testowego
        seed: Ziarno losowości dla powtarzalności
    
    Returns:
        Słownik z statystykami podziału
    """
    # Walidacja proporcji
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Proporcje muszą sumować się do 1.0, otrzymano: {total}")
    
    # Wczytaj dane
    with open(input_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    
    print(f"Wczytano: {len(coco['images'])} obrazów, {len(coco['annotations'])} anotacji")
    
    # Zbuduj mapę image_id -> annotations
    image_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        image_annotations[ann["image_id"]].append(ann)
    
    # Zbuduj mapę image_id -> kategorie (do stratyfikacji)
    image_categories = {}
    for img in coco["images"]:
        img_id = img["id"]
        categories = set(ann["category_id"] for ann in image_annotations.get(img_id, []))
        image_categories[img_id] = frozenset(categories) if categories else frozenset({0})
    
    # Grupuj obrazy według kombinacji kategorii
    category_groups = defaultdict(list)
    for img in coco["images"]:
        key = image_categories[img["id"]]
        category_groups[key].append(img)
    
    # Podziel każdą grupę z zachowaniem proporcji
    random.seed(seed)
    
    train_images, val_images, test_images = [], [], []
    
    for group_key, group_images in category_groups.items():
        random.shuffle(group_images)
        n = len(group_images)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_images.extend(group_images[:train_end])
        val_images.extend(group_images[train_end:val_end])
        test_images.extend(group_images[val_end:])
    
    # Przygotuj zbiory ID
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    test_ids = {img["id"] for img in test_images}
    
    # Funkcja do tworzenia podzbioru COCO
    def create_subset(images: list, image_ids: set, name: str) -> dict:
        subset = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": images,
            "annotations": [
                ann for ann in coco["annotations"]
                if ann["image_id"] in image_ids
            ]
        }
        subset["info"]["description"] = f"{subset['info'].get('description', '')} - {name}"
        return subset
    
    # Utwórz podzbiory
    train_coco = create_subset(train_images, train_ids, "train")
    val_coco = create_subset(val_images, val_ids, "val")
    test_coco = create_subset(test_images, test_ids, "test")
    
    # Zapisz pliki
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": train_coco,
        "val": val_coco,
        "test": test_coco
    }
    
    stats = {"seed": seed}
    
    for name, subset in splits.items():
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        
        n_images = len(subset["images"])
        n_annotations = len(subset["annotations"])
        
        print(f"✓ {name}.json: {n_images} obrazów, {n_annotations} anotacji")
        stats[name] = {"images": n_images, "annotations": n_annotations}
    
    # Sumary
    total_images = stats["train"]["images"] + stats["val"]["images"] + stats["test"]["images"]
    print(f"\n=== PODSUMOWANIE ===")
    print(f"Train: {stats['train']['images']} ({stats['train']['images']/total_images*100:.1f}%)")
    print(f"Val:   {stats['val']['images']} ({stats['val']['images']/total_images*100:.1f}%)")
    print(f"Test:  {stats['test']['images']} ({stats['test']['images']/total_images*100:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Podział zbioru COCO na train/val/test")
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Plik COCO JSON z anotacjami")
    parser.add_argument("-o", "--output-dir", type=Path, default=ANNOTATIONS_DIR,
                        help=f"Katalog wyjściowy (domyślnie: {ANNOTATIONS_DIR})")
    parser.add_argument("--train", type=float, default=SPLIT_RATIOS["train"],
                        help="Proporcja train (domyślnie: 0.70)")
    parser.add_argument("--val", type=float, default=SPLIT_RATIOS["val"],
                        help="Proporcja val (domyślnie: 0.15)")
    parser.add_argument("--test", type=float, default=SPLIT_RATIOS["test"],
                        help="Proporcja test (domyślnie: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Ziarno losowości")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Błąd: Plik nie istnieje: {args.input}")
        sys.exit(1)
    
    split_coco_dataset(
        args.input,
        args.output_dir,
        args.train,
        args.val,
        args.test,
        args.seed
    )


if __name__ == "__main__":
    main()
