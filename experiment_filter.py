import cv2
import numpy as np
import os
import glob

# Path to data
data_dir = "/home/jakub-pytka/HACKNATION/Kopalnia/HackNation2025JSW/data"
output_dir = "/home/jakub-pytka/HACKNATION/Kopalnia/HackNation2025JSW/experiment_output"
os.makedirs(output_dir, exist_ok=True)

# Load some images
image_paths = glob.glob(os.path.join(data_dir, "*.png"))[:20] # Take 20 images
if not image_paths:
    print("No images found.")
    exit()

print(f"Loaded {len(image_paths)} images.")

# Filter params
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def pre_process(img):
    # Gamma
    gamma = 1.4
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    # Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE
    gray = clahe.apply(gray)
    return gray

# 1. Compute Mean Background
print("Computing mean background...")
accum = None
count = 0

for p in image_paths:
    img = cv2.imread(p)
    if img is None: continue
    
    gray = pre_process(img)
    
    if accum is None:
        accum = np.float32(gray)
    else:
        cv2.accumulate(gray, accum)
    count += 1

if count > 0:
    mean_bg = accum / count
    mean_bg_uint8 = cv2.convertScaleAbs(mean_bg)
    cv2.imwrite(os.path.join(output_dir, "mean_background.jpg"), mean_bg_uint8)
    print("Saved mean_background.jpg")
else:
    print("Failed to compute background.")
    exit()

# 2. Apply Filter (Subtract Mean Background)
print("Applying filter...")
for p in image_paths:
    img = cv2.imread(p)
    gray = pre_process(img)
    
    # Diff
    diff = cv2.absdiff(gray, mean_bg_uint8)
    
    # Boost
    diff_boost = cv2.multiply(diff, 5.0)
    
    # Blur
    soft_filtered = cv2.GaussianBlur(diff_boost, (5, 5), 0)
    
    # Save
    name = os.path.basename(p)
    cv2.imwrite(os.path.join(output_dir, f"filtered_{name}"), soft_filtered)

print("Done.")
