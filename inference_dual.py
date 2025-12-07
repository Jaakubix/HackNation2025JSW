import cv2
import torch
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import upload_envs, selectDevice
from torchvision.transforms import functional as F

# Load envs BEFORE importing modules that use them at top-level
upload_envs()

from train.upload.model import getPretrainedModel

def load_model(model_path, num_classes, device):
    print(f"Loading model from {model_path}...")
    model = getPretrainedModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_prediction(model, img_tensor, threshold=0.5):
    with torch.no_grad():
        outputs = model([img_tensor])
    
    output = outputs[0]
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep]

def draw_detections(img, boxes, scores, labels, color, label_map):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_name = label_map.get(label.item(), str(label.item()))
        text = f"{label_name}: {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

class FilterState:
    def __init__(self):
        self.avg_float = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def process(self, frame):
        # Gamma
        gamma = 1.4
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_frame = cv2.LUT(frame, table)
        
        gray = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        if self.avg_float is None:
            self.avg_float = np.float32(gray)
        
        cv2.accumulateWeighted(gray, self.avg_float, 0.1)
        background = cv2.convertScaleAbs(self.avg_float)
        
        diff = cv2.absdiff(gray, background)
        diff_boost = cv2.multiply(diff, 5.0)
        soft_filtered = cv2.GaussianBlur(diff_boost, (5, 5), 0)
        
        # Convert back to BGR for model input (model expects 3 channels)
        return cv2.cvtColor(soft_filtered, cv2.COLOR_GRAY2BGR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", help="Path to output video", default="output_dual.mp4")
    args = parser.parse_args()
    
    device = selectDevice()
    
    # Load Models
    # Num classes = 2 (background + class)
    # Tape Model: Class 1 = tasma
    tape_model = load_model("finetuned_models/tape_model.pth", num_classes=2, device=device)
    
    # Seam Model: Class 1 = szew
    seam_model = load_model("finetuned_models/seam_model.pth", num_classes=2, device=device)
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    filter_state = FilterState()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting inference on {total_frames} frames...")
    
    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Tape Inference (Original Frame)
        img_tensor = F.to_tensor(frame).to(device)
        tape_boxes, tape_scores, tape_labels = get_prediction(tape_model, img_tensor)
        
        # 2. Seam Inference (Filtered Frame)
        filtered_frame = filter_state.process(frame)
        filtered_tensor = F.to_tensor(filtered_frame).to(device)
        seam_boxes, seam_scores, seam_labels = get_prediction(seam_model, filtered_tensor)
        
        # 3. Draw Results
        # Draw Tape (Green)
        frame = draw_detections(frame, tape_boxes, tape_scores, tape_labels, (0, 255, 0), {1: "Tasma"})
        
        # Draw Seam (Red)
        frame = draw_detections(frame, seam_boxes, seam_scores, seam_labels, (0, 0, 255), {1: "Szew"})
        
        # Optional: Show filtered frame in corner?
        # small_filtered = cv2.resize(filtered_frame, (w//4, h//4))
        # frame[0:h//4, 0:w//4] = small_filtered
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Done. Output saved to {args.output}")

if __name__ == "__main__":
    main()
