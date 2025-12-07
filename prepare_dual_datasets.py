import json
import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path("/home/jakub-pytka/HACKNATION/Kopalnia/HackNation2025JSW")
DATA_DIR = BASE_DIR / "data"
SEAM_IMG_DIR = DATA_DIR / "seam_images"
SEAM_IMG_DIR.mkdir(exist_ok=True)

ANN_FILE = DATA_DIR / "_annotations.coco.json"

# Output JSONs
TAPE_ANN_FILE = DATA_DIR / "tape_annotations.json"
SEAM_ANN_FILE = DATA_DIR / "seam_annotations.json"

# Filter Params
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def pre_process(img):
    # Gamma
    gamma = 1.4
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE
    gray = CLAHE.apply(gray)
    return gray

def compute_mean_background(image_paths):
    print("Computing mean background...")
    accum = None
    count = 0
    for p in tqdm(image_paths, desc="Background"):
        img = cv2.imread(str(p))
        if img is None: continue
        gray = pre_process(img)
        if accum is None:
            accum = np.float32(gray)
        else:
            cv2.accumulate(gray, accum)
        count += 1
    
    if count == 0:
        return None
    return cv2.convertScaleAbs(accum / count)

def apply_filter(img, mean_bg):
    gray = pre_process(img)
    diff = cv2.absdiff(gray, mean_bg)
    diff_boost = cv2.multiply(diff, 5.0)
    soft_filtered = cv2.GaussianBlur(diff_boost, (5, 5), 0)
    return soft_filtered

def main():
    print(f"Loading {ANN_FILE}...")
    with open(ANN_FILE, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    
    # Identify Category IDs
    tape_id = None
    seam_id = None
    for cid, name in categories.items():
        if name == 'tasma': tape_id = cid
        if name == 'szew': seam_id = cid
    
    print(f"Tape ID: {tape_id}, Seam ID: {seam_id}")
    
    if tape_id is None or seam_id is None:
        print("Error: Could not find both categories.")
        return

    # --- Prepare Tape Dataset ---
    print("\nPreparing Tape Dataset...")
    tape_anns = [ann for ann in annotations if ann['category_id'] == tape_id]
    tape_img_ids = set(ann['image_id'] for ann in tape_anns)
    tape_images = [images[iid] for iid in tape_img_ids]
    
    # Remap annotations to ID 1
    tape_anns_remapped = []
    for ann in tape_anns:
        new_ann = ann.copy()
        new_ann['category_id'] = 1
        tape_anns_remapped.append(new_ann)
    
    tape_json = {
        "images": tape_images,
        "annotations": tape_anns_remapped,
        "categories": [{"id": 1, "name": "tasma"}]
    }
    
    with open(TAPE_ANN_FILE, 'w') as f:
        json.dump(tape_json, f, indent=2)
    print(f"Saved {TAPE_ANN_FILE} with {len(tape_images)} images and {len(tape_anns)} annotations.")

    # --- Prepare Seam Dataset ---
    print("\nPreparing Seam Dataset...")
    seam_anns = [ann for ann in annotations if ann['category_id'] == seam_id]
    seam_img_ids = set(ann['image_id'] for ann in seam_anns)
    seam_images_meta = [images[iid] for iid in seam_img_ids]
    
    # 1. Compute Mean Background from ALL images (or just seam images? All is better for stability)
    all_img_paths = [DATA_DIR / img['file_name'] for img in coco['images']]
    mean_bg = compute_mean_background(all_img_paths)
    
    if mean_bg is None:
        print("Error: Failed to compute background.")
        return
        
    cv2.imwrite(str(DATA_DIR / "mean_background.jpg"), mean_bg)
    
    # 2. Filter Seam Images and Save
    new_seam_images = []
    print("Filtering images...")
    for img_meta in tqdm(seam_images_meta, desc="Filtering"):
        src_path = DATA_DIR / img_meta['file_name']
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"Warning: Could not read {src_path}")
            continue
            
        filtered = apply_filter(img, mean_bg)
        
        # Save as JPG (lighter) or PNG? Original was PNG. Let's keep PNG to avoid compression artifacts on edges.
        dst_filename = f"filtered_{img_meta['file_name']}"
        dst_path = SEAM_IMG_DIR / dst_filename
        cv2.imwrite(str(dst_path), filtered)
        
        # Update meta
        new_meta = img_meta.copy()
        new_meta['file_name'] = dst_filename # Relative to SEAM_IMG_DIR? No, usually relative to root or we adjust root in loader.
        # Let's assume we will point the loader to SEAM_IMG_DIR. So filename is just basename.
        new_seam_images.append(new_meta)

    # Update annotations to match new image filenames? 
    # Actually, if we change the root dir in loader, we just need filenames to match.
    # But we prefixed them with "filtered_". So we must update file_name in JSON.
    
    # Remap annotations to ID 1
    seam_anns_remapped = []
    for ann in seam_anns:
        new_ann = ann.copy()
        new_ann['category_id'] = 1
        seam_anns_remapped.append(new_ann)
    
    seam_json = {
        "images": new_seam_images,
        "annotations": seam_anns_remapped,
        "categories": [{"id": 1, "name": "szew"}]
    }
    
    with open(SEAM_ANN_FILE, 'w') as f:
        json.dump(seam_json, f, indent=2)
    print(f"Saved {SEAM_ANN_FILE} with {len(new_seam_images)} images and {len(seam_anns)} annotations.")
    print(f"Filtered images saved to {SEAM_IMG_DIR}")

if __name__ == "__main__":
    main()
