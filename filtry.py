import cv2
import numpy as np
import sys
import json
import os
import time
import json
import os
import time
import argparse
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # LUT works with UMat in OpenCV
    return cv2.LUT(image, table)

def load_mask_from_json(json_path, width, height):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Look for "usun" category
    category_id = None
    for cat in data.get('categories', []):
        if cat['name'] == 'usun':
            category_id = cat['id']
            break
            
    if category_id is None:
        print("Warning: Category 'usun' not found in JSON.")
        return mask

    for ann in data.get('annotations', []):
        if ann['category_id'] == category_id:
            # Segmentation is a list of lists of coordinates [x1, y1, x2, y2, ...]
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
    return mask

def process_video_headless(video_path, output_filename, mask=None, duration_limit=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Błąd: Nie można otworzyć pliku wejściowego: {video_path}")
        return

    # Parametry wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = total_frames
    if duration_limit is not None:
        max_frames = int(fps * duration_limit)
        print(f"Limit czasu: {duration_limit}s ({max_frames} klatek)")

    # Konfiguracja zapisu (zapisujemy wynik w pełnej rozdzielczości)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    # --- USTAWIENIA "SOFT" ---
    
    # Inicjalizacja algorytmów
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Inicjalizacja średniej ruchomej
    ret, first = cap.read()
    if not ret: return
    
    # Convert to UMat for OpenCL
    first_umat = cv2.UMat(first)
    prev_gray = cv2.cvtColor(first_umat, cv2.COLOR_BGR2GRAY)
    prev_gray = clahe.apply(prev_gray)
    
    # avg_float musi być float32 do akumulacji, UMat
    # Fix: convertTo might not be available on UMat in Python, use numpy intermediate
    avg_float = cv2.UMat(prev_gray.get().astype("float32"))

    frame_count = 0
    print(f"Rozpoczynam przetwarzanie pliku: {video_path}")
    print(f"Wynik zostanie zapisany jako: {output_filename}")
    if duration_limit is None:
        print(f"Limit czasu: BRAK (Cały plik, {total_frames} klatek)")
    print("Tryb: Headless (Soft Filter - Grayscale + CLAHE + Optimized + OpenCL)")

    # Upload mask to UMat if it exists
    mask_umat = None
    if mask is not None:
        # Invert mask because we want to keep the area OUTSIDE the polygon? 
        # User said "removing this selected area". Usually means blacking it out.
        # If mask has 255 on the area to remove, we want to set that area to 0 in the image.
        # Or we can just use bitwise_and with inverted mask.
        # Let's assume mask has 255 on the region to REMOVE.
        # So we want to keep where mask is 0.
        # Inverted mask: 255 where we keep, 0 where we remove.
        mask_inv = cv2.bitwise_not(mask)
        mask_umat = cv2.UMat(mask_inv)

    start_time = time.time()
    
    while True:
        if duration_limit is not None and frame_count >= max_frames:
            print("\nOsiągnięto limit czasu.")
            break
            
        ret, frame = cap.read()
        if not ret: 
            break # Koniec pliku
        
        frame_count += 1
        
        # Pasek postępu w konsoli
        if frame_count % 100 == 0:
            percent = (frame_count / total_frames) * 100
            sys.stdout.write(f"\rPrzetworzono: {percent:.1f}% ({frame_count}/{total_frames})")
            sys.stdout.flush()

        # Upload to GPU (OpenCL)
        umat_frame = cv2.UMat(frame)

        # Apply mask if exists (Black out the selected area)
        if mask_umat is not None:
            umat_frame = cv2.bitwise_and(umat_frame, umat_frame, mask=mask_umat)

        # 1. Gamma & Gray
        gamma_frame = adjust_gamma(umat_frame, gamma=1.4)
        gray = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        gray = clahe.apply(gray)
        
        # 2. Background Estimation (Accumulate Weighted)
        cv2.accumulateWeighted(gray, avg_float, 0.1)
        background = cv2.convertScaleAbs(avg_float)
        
        diff = cv2.absdiff(gray, background)
        
        # Wzmocnienie różnicy
        diff_boost = cv2.multiply(diff, 5.0) # Scalar multiply works with UMat
        
        # Lekkie rozmycie
        soft_filtered = cv2.GaussianBlur(diff_boost, (5, 5), 0)

        # 5. Zapis do pliku
        # Konwersja na BGR
        output_frame_umat = cv2.cvtColor(soft_filtered, cv2.COLOR_GRAY2BGR)
        
        # Download from GPU to CPU for writing
        output_frame = output_frame_umat.get()
        out.write(output_frame)

    cap.release()
    out.release()
    
    end_time = time.time()
    elapsed = end_time - start_time
    if elapsed > 0:
        fps_proc = frame_count / elapsed
        print(f"\nŚrednia prędkość przetwarzania: {fps_proc:.2f} FPS")
        
    print("\nZakończono sukcesem.")

if __name__ == "__main__":
    import os
    
    # Enable OpenCL explicitly just in case
    cv2.ocl.setUseOpenCL(True)
    print(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "kopalnia")
    
    # Supported video extensions
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process video with soft filter and optional mask.")
    parser.add_argument("video_path", nargs="?", help="Path to the input video file.")
    parser.add_argument("--mask", help="Path to the JSON mask file.", default="result.json")
    
    args = parser.parse_args()

    # If video path is provided via CLI
    if args.video_path:
        input_path = args.video_path
        if not os.path.exists(input_path):
            # Check if it's in kopalnia dir
            possible_path = os.path.join(base_dir, "kopalnia", input_path)
            if os.path.exists(possible_path):
                input_path = possible_path
            else:
                print(f"Error: Video file {input_path} not found.")
                sys.exit(1)
        
        # Determine output path
        input_dir = os.path.dirname(os.path.abspath(input_path))
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_soft_filtered{ext}"
        output_path = os.path.join(input_dir, output_filename)
        
        # Load mask
        mask = None
        json_path = args.mask
        
        # If mask path is relative, check relative to CWD or script dir
        if not os.path.exists(json_path):
             possible_paths = [
                os.path.join(base_dir, json_path),
                os.path.join(base_dir, "HackNation2025JSW", "data", json_path)
            ]
             for p in possible_paths:
                if os.path.exists(p):
                    json_path = p
                    break
        
        if os.path.exists(json_path):
            print(f"Loading mask from: {json_path}")
            cap_temp = cv2.VideoCapture(input_path)
            if cap_temp.isOpened():
                w_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                mask = load_mask_from_json(json_path, w_temp, h_temp)
                cap_temp.release()
            else:
                print(f"Could not open {input_path} to read dims.")
        else:
            print(f"Warning: Mask file {json_path} not found. Processing without mask.")

        process_video_headless(input_path, output_path, mask)

    else:
        # Fallback to scanning directory (legacy behavior or default)
        if not os.path.exists(input_dir):
            print(f"Katalog {input_dir} nie istnieje!")
            sys.exit(1)

        print(f"Szukam nagrań w: {input_dir}")
        print(f"Filtruję WSZYSTKIE pliki (PEŁNA DŁUGOŚĆ)...")

        for filename in os.listdir(input_dir):
            _, ext = os.path.splitext(filename)
            if ext.lower() in video_extensions:
                input_path = os.path.join(input_dir, filename)
                
                # Create output filename (e.g. video.mp4 -> video_soft_filtered.mp4)
                name, ext = os.path.splitext(filename)
                # Avoid re-processing output files
                if "_soft_filtered" in name:
                    continue
                    
                output_filename = f"{name}_soft_filtered{ext}"
                output_path = os.path.join(input_dir, output_filename) 
                
                # Load mask (default logic)
                # Try multiple locations for result.json
                possible_paths = [
                    os.path.join(base_dir, "result.json"),
                    os.path.join(base_dir, "HackNation2025JSW", "data", "result.json")
                ]
                
                json_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        json_path = p
                        break
                
                mask = None
                if json_path:
                    print(f"Loading mask from: {json_path}")
                    cap_temp = cv2.VideoCapture(input_path)
                    if cap_temp.isOpened():
                        w_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h_temp = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        mask = load_mask_from_json(json_path, w_temp, h_temp)
                        cap_temp.release()
                    else:
                        print(f"Could not open {input_path} to read dims.")
                
                # Filter for specific file if requested (Legacy hardcoded check removed in favor of CLI)
                # target_file = "nagranie1.mkv"
                # if filename != target_file:
                #     continue

                process_video_headless(input_path, output_path, mask)
                print("-" * 50)