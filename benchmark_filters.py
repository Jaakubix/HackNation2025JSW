import cv2
import numpy as np
import time
import os
import sys

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def run_benchmark(config_name, video_path, duration=60):
    print(f"\n--- Running Config: {config_name} ---")
    
    # Config parsing
    use_opencl = "CPU" not in config_name
    cv2.ocl.setUseOpenCL(use_opencl)
    
    use_clahe = "No CLAHE" not in config_name
    use_blur = "No Blur" not in config_name
    blur_kernel = (3, 3) if "Small Blur" in config_name else (5, 5)
    use_gamma_pow = "Gamma Pow" in config_name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * duration)
    
    output_filename = f"benchmark_{config_name.replace(' ', '_').replace('(', '').replace(')', '')}.mp4"
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if use_clahe else None
    
    ret, first = cap.read()
    if not ret: return 0
    
    if use_opencl:
        first_umat = cv2.UMat(first)
        prev_gray = cv2.cvtColor(first_umat, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        prev_gray = clahe.apply(prev_gray)
    
    if use_opencl:
        avg_float = cv2.UMat(prev_gray.get().astype("float32"))
    else:
        avg_float = prev_gray.astype("float32")

    start_time = time.time()
    frame_count = 0
    processed_frames = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        processed_frames += 1

        if use_opencl:
            umat_frame = cv2.UMat(frame)
            
            if use_gamma_pow:
                # Gamma correction using pow on GPU
                # Normalize to 0-1, pow, scale back
                norm = cv2.multiply(umat_frame, 1.0/255.0, dtype=cv2.CV_32F)
                pow_frame = cv2.pow(norm, 1.0/1.4)
                gamma_frame = cv2.multiply(pow_frame, 255.0, dtype=cv2.CV_8U)
            else:
                gamma_frame = adjust_gamma(umat_frame, gamma=1.4)
                
            gray = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)
        else:
            gamma_frame = adjust_gamma(frame, gamma=1.4)
            gray = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)
        
        if use_clahe:
            gray = clahe.apply(gray)
        
        cv2.accumulateWeighted(gray, avg_float, 0.1)
        background = cv2.convertScaleAbs(avg_float)
        
        diff = cv2.absdiff(gray, background)
        
        if use_opencl:
            diff_boost = cv2.multiply(diff, 5.0)
        else:
            diff_boost = cv2.multiply(diff, np.array([5.0]))
            
        if use_blur:
            soft_filtered = cv2.GaussianBlur(diff_boost, blur_kernel, 0)
        else:
            soft_filtered = diff_boost

        output_frame_umat = cv2.cvtColor(soft_filtered, cv2.COLOR_GRAY2BGR)
        
        if use_opencl:
            output_frame = output_frame_umat.get()
        else:
            output_frame = output_frame_umat
            
        out.write(output_frame)

    end_time = time.time()
    elapsed = end_time - start_time
    actual_fps = processed_frames / elapsed
    
    cap.release()
    out.release()
    
    # Cleanup output file to save space
    if os.path.exists(output_filename):
        os.remove(output_filename)

    print(f"Result: {actual_fps:.2f} FPS")
    return actual_fps

if __name__ == "__main__":
    # Target file
    video_path = "kopalnia/06.10.2025 02_59_59 (UTC+02_00).mkv"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)

    configs = [
        "1. Baseline (OpenCL)",
        "2. No CLAHE",
        "3. Small Blur",
        "4. No Blur",
        "5. Gamma Pow",
        "6. Gamma Pow + No CLAHE",
        "7. Gamma Pow + Small Blur",
        "8. CPU Only"
    ]

    results = {}

    print("Starting Optimization Tournament (8 iterations, 60s each)...")
    
    for config in configs:
        fps = run_benchmark(config, video_path, duration=60)
        results[config] = fps

    print("\n--- TOURNAMENT RESULTS ---")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for config, fps in sorted_results:
        print(f"{config}: {fps:.2f} FPS")
    
    winner = sorted_results[0][0]
    print(f"\nWINNER: {winner}")
