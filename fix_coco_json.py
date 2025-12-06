import json
import os
from pathlib import Path

def fix_coco_json():
    data_dir = Path("data")
    src_json = data_dir / "result.json"
    dst_json = data_dir / "_annotations.coco.json"

    if not src_json.exists():
        print(f"Brak pliku {src_json}")
        return

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Naprawa ścieżek obrazów (tylko nazwa pliku)
    # 2. Mapowanie kategorii: 0 -> 1, 1 -> 2 (jeśli startują od 0)
    
    # Sprawdzenie czy kategorie startują od 0
    min_cat_id = min(c["id"] for c in data["categories"])
    shift_ids = (min_cat_id == 0)

    cat_map = {} # old_id -> new_id
    if shift_ids:
        print("Wykryto kategorie od 0. Przesuwam ID o +1.")
        
    for cat in data["categories"]:
        old_id = cat["id"]
        new_id = old_id + 1 if shift_ids else old_id
        cat_map[old_id] = new_id
        cat["id"] = new_id
    
    # Aktualizacja obrazów
    for img in data["images"]:
        p = Path(img["file_name"])
        img["file_name"] = p.name  # tylko nazwa pliku
    
    # Aktualizacja adnotacji
    for ann in data["annotations"]:
        old_cid = ann["category_id"]
        ann["category_id"] = cat_map.get(old_cid, old_cid)

    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Zapisano naprawiony JSON do {dst_json}")
    # Opcjonalnie usuń stary splits.json żeby wymusić przeliczenie podziału
    splits = data_dir / "splits.json"
    if splits.exists():
        splits.unlink()
        print("Usunięto stary splits.json")

if __name__ == "__main__":
    fix_coco_json()
