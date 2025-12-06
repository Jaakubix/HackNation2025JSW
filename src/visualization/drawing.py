from typing import List

import numpy as np
import cv2
import torch

# ====== Stałe kolory (BGR) przypisane do nazw klas (z config.py) ======
# ====== Stałe kolory (BGR) przypisane do nazw klas (z config.py) ======
NAME2COLOR = {
    "background": (128, 128, 128),
    "szew": (0, 255, 0),        # Zielony
    "tasma": (255, 0, 0),       # Niebieski
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
    """
    Rysuje ramki i maski na obrazie BGR.
    Args:
        img_bgr: obraz (H, W, 3)
        boxes: tensor (N, 4)
        labels: tensor (N,)
        scores: tensor (N,)
        categories: lista nazw klas (indeks = label_id)
        score_thr: próg pewności
        masks: tensor (N, 1, H, W) opcjonalnie
    """
    # Konwersja numpy do torch jeśli trzeba
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    if masks is not None and isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    h, w = img_bgr.shape[:2]
    thick = max(1, int(round(min(h, w) * 0.003)))
    fscale = max(0.4, min(1.2, min(h, w) / 800.0))
    fthick = max(1, thick // 2)


    for i, (box, lab, sc) in enumerate(zip(boxes, labels, scores)):
        sc = float(sc)
        if not np.isfinite(sc) or sc < score_thr:
            continue

        x1, y1, x2, y2 = _clamp_box(*box.tolist(), w=w, h=h)
        color = _color_for_label(int(lab), categories)

        # Maska
        if masks is not None and i < len(masks):
            mask = masks[i, 0]
            # Thresholding 0.5
            mask_bool = (mask > 0.5).cpu().numpy() if isinstance(mask, torch.Tensor) else (mask > 0.5)
            
            if mask_bool.any():
                # Alpha blending (0.5)
                alpha = 0.5
                img_bgr[mask_bool] = (
                    img_bgr[mask_bool].astype(np.float32) * (1 - alpha) + 
                    np.array(color).astype(np.float32) * alpha
                ).astype(np.uint8)
                
                # Kontur maski
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_bgr, contours, -1, color, thickness=2)

        # ramka
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=thick)

        # % pewności nad ramką
        pct = f"{int(round(sc * 100))}%"
        (tw, th), baseline = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)

        pad = 2
        top = max(0, y1 - th - baseline - 2*pad)
        left = max(0, min(x1, w - tw - 2*pad))

        cv2.rectangle(
            img_bgr,
            (left, top),
            (left + tw + 2*pad, top + th + baseline + 2*pad),
            color, thickness=-1
        )
        tcolor = _text_color_for(color)
        cv2.putText(
            img_bgr, pct, (left + pad, top + th + pad - 1),
            cv2.FONT_HERSHEY_SIMPLEX, fscale, tcolor, fthick, cv2.LINE_AA
        )

    return img_bgr
