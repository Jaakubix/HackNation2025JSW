# test_trapezoid.py
"""Simple test to ensure trapezoid drawing stays within frame bounds.
Runs inference on a short video segment and checks that all polygon
vertices are inside the image dimensions.
"""
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from utils import selectDevice
import os
os.environ.setdefault('BATCH_SIZE', '1')
from train.upload.model import getPretrainedModel
from inference_dual import load_model, get_prediction, draw_detections

VIDEO_PATH = "nagranie3_1min.mp4"
DEVICE = selectDevice()

# Load models (same as inference_dual)
tape_model = load_model("finetuned_models/tape_model.pth", num_classes=2, device=DEVICE)
seam_model = load_model("finetuned_models/seam_model.pth", num_classes=2, device=DEVICE)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Process first 10 frames only
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    # Tape inference
    img_tensor = F.to_tensor(frame).to(DEVICE)
    tape_boxes, tape_scores, tape_labels = get_prediction(tape_model, img_tensor)
    # Draw (this will also perform the clamping checks)
    out = draw_detections(frame.copy(), tape_boxes, tape_scores, tape_labels, (0,255,0), {1:"Tasma"})
    # Verify that no pixel index is out of bounds (draw_detections already clamps)
    # Additional sanity: ensure polygon points are within image size
    for box in tape_boxes:
        x1, y1, x2, y2 = box.int().tolist()
        assert 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h, "Box exceeds frame bounds"
print("All checks passed for first 10 frames.")
cap.release()
