from typing import List

import numpy as np
import cv2
import torch

# ====== Stałe kolory (BGR) przypisane do nazw klas ======
NAME2COLOR = {
    "szew": (0, 0, 255),    # Czerwony (BGR: B=0, G=0, R=255)
    "tasma": (0, 255, 0),   # Zielony
    "medal-nato": (255, 0, 0),
    "gwiazda-afganistanu": (0, 100, 0),
    "zloty-medal-za-zaslugi-dla-obronnosci-kraju": (0, 255, 255),
    "srebrny-medal-sily-zbrojne-w-sluzbie-ojczyzny": (220, 220, 220),
}
DEFAULT_COLOR = (128, 128, 128)  # fallback (szary)

def _color_for_label(label_id: int, categories: List[str]) -> tuple[int, int, int]:
    if 0 <= label_id < len(categories):
        name = (categories[label_id] or "").strip().lower()
        return NAME2COLOR.get(name, DEFAULT_COLOR)
    return DEFAULT_COLOR

def _text_color_for(bg_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bg_bgr
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # luminancja sRGB
    return (0, 0, 0) if y > 145 else (255, 255, 255)

# ====== Rysowanie ======
def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def draw_boxes_bgr(img_bgr: np.ndarray,
                   boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor,
                   categories: List[str], score_thr: float,
                   masks: torch.Tensor = None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    thick = max(1, int(round(min(h, w) * 0.003)))
    fscale = max(0.4, min(1.2, min(h, w) / 800.0))
    fthick = max(1, thick // 2)

    # Kopiujemy obraz bo będziemy na nim rysować maski z alpha blend
    out_img = img_bgr.copy()

    for i, (box, lab, sc) in enumerate(zip(boxes, labels, scores)):
        sc = float(sc)
        if not np.isfinite(sc) or sc < score_thr:
            continue

        x1, y1, x2, y2 = _clamp_box(*box.tolist(), w=w, h=h)
        color = _color_for_label(int(lab), categories)

        # --- MASKI ---
        if masks is not None and i < masks.shape[0]:
            # mask: [1, H, W] -> [H, W]
            raw_mask = masks[i, 0].cpu().numpy()
            
            # W modelu Mask R-CNN maski są floatami 0..1 (sigmoida). Binaryzacja 0.5
            bin_mask = (raw_mask >= 0.5)

            if bin_mask.any():
                # Przytnij maskę do roota (opcjonalne, ale MaskRCNN zwraca maski 28x28 i nakłada je
                # za pomocą paste_masks_in_image, jednak tutaj mamy już pełne wymiary HxW
                # jeśli używamy standardowego outputu z postprocessingu).
                # Kolorujemy piksele maski
                
                # Tworzymy overlay konkretnie pod maskę
                # Używamy wycinka bounding boxa dla optymalizacji,
                # albo całej maski. Maska jest wielkości obrazu HxW
                
                # Nakładamy kolor z alpha=0.5
                roi = out_img[bin_mask]
                
                # Mieszanie
                # kolor to BGR
                # Musimy zrobić broadcast koloru
                blended = (roi.astype(np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
                out_img[bin_mask] = blended
                
                # Opcjonalnie kontur
                # contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(out_img, contours, -1, color, 1)

    # Rysujemy kontury wielokątów i etykiety na obrazie z maskami
    for i, (box, lab, sc) in enumerate(zip(boxes, labels, scores)):
        sc = float(sc)
        if not np.isfinite(sc) or sc < score_thr:
            continue
            
        x1, y1, x2, y2 = _clamp_box(*box.tolist(), w=w, h=h)
        color = _color_for_label(int(lab), categories)

        # Rysuj kontur wielokąta z maski (jeśli dostępna), w przeciwnym razie prostokąt
        contour_drawn = False
        if masks is not None and i < masks.shape[0]:
            raw_mask = masks[i, 0].cpu().numpy()
            bin_mask = (raw_mask >= 0.5).astype(np.uint8)
            if bin_mask.any():
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(out_img, contours, -1, color, thickness=thick)
                    contour_drawn = True
        
        # Fallback: jeśli brak maski lub konturu, rysuj prostokąt
        if not contour_drawn:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness=thick)

        # % pewności nad ramką (pozycja oparta na bbox dla spójności)
        pct = f"{int(round(sc * 100))}%"
        (tw, th), baseline = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)

        pad = 2
        top = max(0, y1 - th - baseline - 2*pad)
        left = max(0, min(x1, w - tw - 2*pad))

        cv2.rectangle(
            out_img,
            (left, top),
            (left + tw + 2*pad, top + th + baseline + 2*pad),
            color, thickness=-1
        )
        tcolor = _text_color_for(color)
        cv2.putText(
            out_img, pct, (left + pad, top + th + pad - 1),
            cv2.FONT_HERSHEY_SIMPLEX, fscale, tcolor, fthick, cv2.LINE_AA
        )

    return out_img

# ====== IoU i dopasowanie ======
def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32, device=a.device)
    inter_x1 = torch.max(a[:, None, 0], b[:, 0])
    inter_y1 = torch.max(a[:, None, 1], b[:, 1])
    inter_x2 = torch.min(a[:, None, 2], b[:, 2])
    inter_y2 = torch.min(a[:, None, 3], b[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))

def match_detections_to_gt(pred_boxes, pred_labels, pred_scores,
                           gt_boxes, gt_labels, iou_thr=0.5, score_thr=0.0):
    # Ew. filtr po progu (zostawione dla zgodności, ale w main już filtrujemy)
    if pred_scores.numel():
        keep = torch.isfinite(pred_scores)
        if score_thr > 0:
            keep = keep & (pred_scores >= score_thr)
        pred_boxes  = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        order = torch.argsort(pred_scores, descending=True)
        pred_boxes  = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

    iou = box_iou_xyxy(pred_boxes, gt_boxes)
    gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    tp = fp = 0
    matches = []

    for i in range(pred_boxes.shape[0]):
        same_cls = (gt_labels == pred_labels[i]) if gt_labels.numel() else torch.zeros(0, dtype=torch.bool, device=gt_labels.device)
        if iou.shape[1] > 0 and same_cls.any():
            ious = torch.where(same_cls, iou[i], torch.zeros_like(iou[i]))
            j = torch.argmax(ious).item()
            if iou[i, j] >= iou_thr and not gt_used[j]:
                tp += 1
                gt_used[j] = True
                matches.append((i, j))
            else:
                fp += 1
        else:
            fp += 1

    fn = int((~gt_used).sum().item())
    return tp, fp, fn, matches
