import torch

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Oblicza macierz IoU (Intersection over Union) dla dwóch zestawów pudełek (xyxy).
    Args:
        a: tensor (N, 4) [x1, y1, x2, y2]
        b: tensor (M, 4) [x1, y1, x2, y2]
    Returns:
        iou: tensor (N, M)
    """
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
    """
    Dopasowuje predykcje do ground truth (GT) i liczy TP/FP/FN.
    Args:
        pred_boxes: (N, 4)
        pred_labels: (N,)
        pred_scores: (N,)
        gt_boxes: (M, 4)
        gt_labels: (M,)
        iou_thr: minimalne IoU, żeby uznać za trafienie
        score_thr: minimalny wynik pewności predykcji
    
    Returns:
        tp (int), fp (int), fn (int), matches (list of (pred_idx, gt_idx))
    """
    # Filtracja po progu pewności
    if pred_scores.numel():
        keep = torch.isfinite(pred_scores)
        if score_thr > 0:
            keep = keep & (pred_scores >= score_thr)
        pred_boxes  = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # Sortowanie po score malejąco (najpierw najpewniejsze)
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes  = pred_boxes[order]
        pred_labels = pred_labels[order]
        pred_scores = pred_scores[order]

    # Oblicz IoU
    iou = box_iou_xyxy(pred_boxes, gt_boxes)
    gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    
    tp = 0
    fp = 0
    matches = []

    for i in range(pred_boxes.shape[0]):
        # Sprawdź czy klasy się zgadzają
        if gt_labels.numel() > 0:
            same_cls = (gt_labels == pred_labels[i])
        else:
            same_cls = torch.zeros(0, dtype=torch.bool, device=gt_labels.device)
            
        if iou.shape[1] > 0 and same_cls.any():
            # Wybierz IoU tylko z tymi samymi klasami
            ious = torch.where(same_cls, iou[i], torch.zeros_like(iou[i]))
            
            # Najlepsze dopasowanie
            j = torch.argmax(ious).item()
            
            if iou[i, j] >= iou_thr and not gt_used[j]:
                # Trafienie!
                tp += 1
                gt_used[j] = True
                matches.append((i, j))
            else:
                # Złe dopasowanie lub GT już zajęte
                fp += 1
        else:
            # Brak pasującej klasy lub brak GT
            fp += 1

    fn = int((~gt_used).sum().item())
    return tp, fp, fn, matches
