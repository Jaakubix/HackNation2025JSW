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
    masks = output.get('masks')
    keep = scores >= threshold
    if masks is not None:
        masks = masks[keep]
    return boxes[keep], scores[keep], labels[keep], masks

def draw_detections(img, boxes, scores, labels, color, label_map, masks=None):
    """Draw detections:
    - For tape (class 1): use model mask if available (with low threshold for larger area).
    - Fallback to trapezoid if mask fails.
    - Rectangle for other classes.
    """
    h_img, w_img = img.shape[:2]
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.int().tolist()
        # Clamp coordinates
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))
        
        is_tape = (label.item() == 1)
        polygon_to_draw = None
        
        if is_tape and masks is not None and idx < len(masks):
            # Use model mask
            mask_prob = masks[idx][0].cpu().numpy()
            # Low threshold (0.3) to include more pixels -> "larger"
            mask_binary = (mask_prob > 0.3).astype('uint8') * 255
            
            # Dilate to make it slightly larger/smoother
            kernel = np.ones((5,5), np.uint8)
            mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                # Approx polygon
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) > 8:
                    approx = cv2.convexHull(approx)
                    epsilon = 0.01 * cv2.arcLength(approx, True)
                    approx = cv2.approxPolyDP(approx, epsilon, True)
                
                polygon_to_draw = approx

        if is_tape and polygon_to_draw is None:
            # Fallback trapezoid (no margin = full width)
            width = max(1, x2 - x1)
            margin = 0 # Full width as requested "larger"
            pts = np.array([
                [x1 + margin, y1],
                [x2 - margin, y1],
                [x2, y2],
                [x1, y2]
            ], np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)
            polygon_to_draw = pts

        if is_tape and polygon_to_draw is not None:
            # Draw polygon
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon_to_draw], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            cv2.polylines(img, [polygon_to_draw], True, color, 2)
            
            # Calculate width line intersection
            mid_y = (y1 + y2) // 2
            poly_pts = polygon_to_draw.reshape(-1, 2)
            intersections = []
            for i in range(len(poly_pts)):
                p1 = poly_pts[i]
                p2 = poly_pts[(i + 1) % len(poly_pts)]
                if (p1[1] <= mid_y < p2[1]) or (p2[1] <= mid_y < p1[1]):
                    if p2[1] != p1[1]:
                        x = p1[0] + (mid_y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(int(x))
            
            if len(intersections) >= 2:
                start_x = min(intersections)
                end_x = max(intersections)
                width_px = end_x - start_x
                
                # Draw line inside
                cv2.line(img, (start_x, mid_y), (end_x, mid_y), (255, 255, 0), 1)
                
                # Text above-left
                cv2.putText(img, f"Szer: {width_px}px", (start_x, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                # Fallback if intersection fails (e.g. horizontal edge at mid_y)
                cx, cy, cw, ch = cv2.boundingRect(polygon_to_draw)
                mid_y = cy + ch // 2
                cv2.line(img, (cx, mid_y), (cx + cw, mid_y), (255, 255, 0), 1)
                cv2.putText(img, f"Szer: {cw}px", (cx, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        elif not is_tape:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # label background and text
        label_name = label_map.get(label.item(), str(label.item()))
        text = f"{label_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
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
        tape_boxes, tape_scores, tape_labels, tape_masks = get_prediction(tape_model, img_tensor)
        
        # 2. Seam Inference (Filtered Frame)
        filtered_frame = filter_state.process(frame)
        filtered_tensor = F.to_tensor(filtered_frame).to(device)
        seam_boxes, seam_scores, seam_labels, seam_masks = get_prediction(seam_model, filtered_tensor)
        
        # 3. Draw Results
# Draw Tape (Green)
        frame = draw_detections(frame, tape_boxes, tape_scores, tape_labels, (0, 255, 0), {1: "Tasma"}, tape_masks)
        
        # Draw Seam (Red)
        frame = draw_detections(frame, seam_boxes, seam_scores, seam_labels, (0, 0, 255), {1: "Szew"}, seam_masks)
        
        # Optional: Show filtered frame in corner?
        # small_filtered = cv2.resize(filtered_frame, (w//4, h//4))
        # frame[0:h//4, 0:w//4] = small_filtered
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Done. Output saved to {args.output}")

if __name__ == "__main__":
    main()
