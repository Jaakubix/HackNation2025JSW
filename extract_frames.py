import cv2
import os
import sys
import argparse

def extract_frames(video_path, output_base_dir):
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist.")
        return

    filename = os.path.basename(video_path)
    video_id, _ = os.path.splitext(filename)
    
    # Create output directory for this video ID
    output_dir = os.path.join(output_base_dir, video_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: FPS is 0, cannot calculate interval.")
        return

    interval_frames = int(fps * 2) # Every 2 seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {filename}...")
    print(f"FPS: {fps}, Interval: {interval_frames} frames (2s)")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval_frames == 0:
            # Save frame
            saved_count += 1
            output_filename = f"{saved_count:05d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            
            time_sec = frame_count / fps
            sys.stdout.write(f"\rSaved frame at {time_sec:.1f}s ({saved_count} total)")
            sys.stdout.flush()
            
        frame_count += 1
        
    cap.release()
    print(f"\nFinished. Extracted {saved_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video every 2 seconds.")
    parser.add_argument("video_file", nargs="?", help="Path to specific video file to process. If not provided, scans 'kopalnia' directory.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(base_dir, "frame")
    
    if args.video_file:
        # Process single file
        video_path = args.video_file
        if not os.path.exists(video_path):
            # Try checking relative to current dir or kopalnia dir
            possible_path = os.path.join(base_dir, "kopalnia", video_path)
            if os.path.exists(possible_path):
                video_path = possible_path
            else:
                print(f"Error: File {video_path} not found.")
                sys.exit(1)
        
        extract_frames(video_path, output_base_dir)
        
    else:
        # Process all in kopalnia directory
        input_dir = os.path.join(base_dir, "kopalnia")
        
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'}
        
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} does not exist.")
            sys.exit(1)
            
        print(f"Scanning {input_dir} for videos...")
        
        for filename in os.listdir(input_dir):
            _, ext = os.path.splitext(filename)
            if ext.lower() in video_extensions:
                # Skip filtered files to avoid duplicates/loops if they are in same dir
                if "_soft_filtered" in filename:
                    continue
                    
                video_path = os.path.join(input_dir, filename)
                extract_frames(video_path, output_base_dir)
                print("-" * 50)
