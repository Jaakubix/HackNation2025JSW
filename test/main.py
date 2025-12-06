import os
from pathlib import Path
from typing import Dict
import json
import cv2
import numpy as np
import torch

from test.upload import DEVICE
from test.upload import load_test_dataset_and_classes, load_trained_model
from test.visualizer import match_detections_to_gt, draw_boxes_bgr
from utils import upload_envs

# --- wczytaj .env na starcie ---
upload_envs()

THIS_DIR = Path(__file__).resolve().parent

DETECTIONS_DIR_ENV = os.environ.get("DETECTIONS_DIR")
if DETECTIONS_DIR_ENV:
    OUTPUT_DIR = Path(DETECTIONS_DIR_ENV)
else:
    OUTS = Path(os.environ.get("OUTS", str(THIS_DIR.parent / "detections_out")))
    OUTPUT_DIR = OUTS / "imgs" if OUTS.name == "outs" else OUTS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _as_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default

def run():
    # --- progi z .env (z bezpiecznymi domyślnymi) ---
    VIS_SCORE_THR = _as_float("VIS_SCORE_THR", 0.8)
    EVAL_SCORE_THR = _as_float("EVAL_SCORE_THR", 0.5)
    EVAL_IOU_THR   = _as_float("EVAL_IOU_THR", 0.5)

    loader, categories, img_files = load_test_dataset_and_classes()
    model, trained_num_classes = load_trained_model()

    # dopasuj listę nazw do liczby klas w modelu
    if len(categories) < trained_num_classes:
        categories += [f"class_{i}" for i in range(len(categories), trained_num_classes)]
    elif len(categories) > trained_num_classes:
        categories = categories[:trained_num_classes]

    global_tp = global_fp = global_fn = 0
    per_cls: Dict[int, Dict[str, int]] = {}

    # śledzenie plików z błędami
    files_with_fp: list[dict] = []
    files_with_fn: list[dict] = []
    files_with_errors: list[str] = []

    from collections import Counter

    for (images, targets), img_name in zip(loader, img_files):
        image = images[0].to(DEVICE)
        gt = targets[0]
        gt_boxes = gt["boxes"].to(DEVICE)
        gt_labels = gt["labels"].to(DEVICE)

        with torch.no_grad():
            outputs = model([image])[0]

        # surowe wyniki
        boxes  = outputs.get("boxes",  torch.empty(0, device=DEVICE))
        labels = outputs.get("labels", torch.empty(0, dtype=torch.long, device=DEVICE))
        scores = outputs.get("scores", torch.empty(0, device=DEVICE))
        masks  = outputs.get("masks",  None) # [N, 1, H, W]

        # === FILTR DO EWALUACJI (EVAL_SCORE_THR) ===
        if scores.numel():
            keep = torch.isfinite(scores) & (scores >= EVAL_SCORE_THR)
        else:
            keep = torch.zeros_like(scores, dtype=torch.bool)

        boxes_eval  = boxes[keep]
        labels_eval = labels[keep]
        scores_eval = scores[keep]
        # Maski nie są kluczowe do ewaluacji detekcji (IoU liczymy po boxach w tym kodzie),
        # więc nie musimy ich filtrować do zmiennej _eval, chyba że chcielibyśmy liczyć Mask IoU.

        # METRYKI liczymy na przefiltrowanych predykcjach
        tp, fp, fn, matches = match_detections_to_gt(
            pred_boxes=boxes_eval, pred_labels=labels_eval, pred_scores=scores_eval,
            gt_boxes=gt_boxes, gt_labels=gt_labels,
            iou_thr=EVAL_IOU_THR, score_thr=0.0  # już odfiltrowane wcześniej
        )
        global_tp += tp; global_fp += fp; global_fn += fn

        # zapamiętaj pliki z błędami (na poziomie obrazu)
        pred_count = int(boxes_eval.shape[0])
        gt_count   = int(gt_boxes.shape[0])
        tp_img     = int(len(matches))
        fp_img     = max(0, pred_count - tp_img)
        fn_img     = max(0, gt_count - tp_img)
        if fp_img > 0:
            files_with_fp.append({"file": str(img_name), "fp": fp_img, "pred": pred_count})
            files_with_errors.append(str(img_name))
        if fn_img > 0:
            files_with_fn.append({"file": str(img_name), "fn": fn_img, "gt": gt_count})
            files_with_errors.append(str(img_name))

        # per-klasa – licz na tym samym zbiorze co wyżej
        tp_c: Dict[int, int] = {}
        for i_pred, _j_gt in matches:
            cid = int(labels_eval[i_pred].item())
            tp_c[cid] = tp_c.get(cid, 0) + 1

        pred_counts = Counter(int(c) for c in labels_eval.detach().cpu().tolist())
        gt_counts   = Counter(int(c) for c in gt_labels.detach().cpu().tolist())

        for cid, count in pred_counts.items():
            per_cls.setdefault(cid, {"tp": 0, "fp": 0, "fn": 0})
            per_cls[cid]["tp"] += tp_c.get(cid, 0)
            per_cls[cid]["fp"] += max(0, count - tp_c.get(cid, 0))
        for cid, count in gt_counts.items():
            per_cls.setdefault(cid, {"tp": 0, "fp": 0, "fn": 0})
            per_cls[cid]["fn"] += max(0, count - tp_c.get(cid, 0))

        # --- WIZUALIZACJA: tylko z progiem VIS_SCORE_THR ---
        img_np = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        base = Path(img_name).stem

        # Przekazujemy maski (może być None)
        masks_cpu = masks.detach().cpu() if masks is not None else None

        vis = draw_boxes_bgr(
            img_bgr,
            boxes.detach().cpu(), labels.detach().cpu(), scores.detach().cpu(),
            categories, score_thr=float(VIS_SCORE_THR),
            masks=masks_cpu
        )
        cv2.imwrite(str(OUTPUT_DIR / f"{base}_det.jpg"), vis)

    # PODSUMOWANIE
    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    g_prec, g_rec, g_f1 = prf(global_tp, global_fp, global_fn)
    lines = []
    lines.append(f"=== Global metrics (eval_score_thr={EVAL_SCORE_THR}, IoU_thr={EVAL_IOU_THR}) ===")
    lines.append(f"TP={global_tp}  FP={global_fp}  FN={global_fn}")
    lines.append(f"Precision={g_prec:.4f}  Recall={g_rec:.4f}  F1={g_f1:.4f}")
    lines.append("\n=== Per-class metrics ===")
    for cid in sorted(per_cls.keys()):
        tp = per_cls[cid]["tp"]; fp = per_cls[cid]["fp"]; fn = per_cls[cid]["fn"]
        p, r, f = prf(tp, fp, fn)
        name = categories[cid] if 0 <= cid < len(categories) else str(cid)
        lines.append(f"[{cid}] {name:25s} TP={tp:4d} FP={fp:4d} FN={fn:4d}  P={p:.3f} R={r:.3f} F1={f:.3f}")

    # === LISTA PLIKÓW Z BŁĘDAMI ===
    lines.append("\n=== Files with errors ===")
    if not files_with_fp and not files_with_fn:
        lines.append("No images with FP/FN at the chosen thresholds.")
    else:
        if files_with_fp:
            lines.append("\n-- False Positives (extra detections) --")
            for rec in files_with_fp:
                lines.append(f"[FP] {rec['file']}  extra={rec['fp']}  preds={rec['pred']}")
        if files_with_fn:
            lines.append("\n-- False Negatives (missed GT) --")
            for rec in files_with_fn:
                lines.append(f"[FN] {rec['file']}  missed={rec['fn']}  gt={rec['gt']}")

    report_txt = "\n".join(lines)
    print("\n" + report_txt)
    (OUTPUT_DIR / "metrics.txt").write_text(report_txt, encoding="utf-8")

    # zapis też do JSON
    import json as _json
    errors_payload = {
        "eval_score_thr": EVAL_SCORE_THR,
        "eval_iou_thr": EVAL_IOU_THR,
        "files_with_fp": files_with_fp,
        "files_with_fn": files_with_fn,
        "files_with_errors_unique": sorted(set(files_with_errors)),
    }
    (OUTPUT_DIR / "errors.json").write_text(_json.dumps(errors_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_json = {
        "visualization_score_threshold": float(os.environ.get("VIS_SCORE_THR", "0.7")),
        "evaluation_score_threshold": float(os.environ.get("EVAL_SCORE_THR", "0.5")),
        "iou_threshold": float(os.environ.get("EVAL_IOU_THR", "0.5")),
        "global": {"tp": global_tp, "fp": global_fp, "fn": global_fn,
                   "precision": g_prec, "recall": g_rec, "f1": g_f1},
        "per_class": {
            str(cid): {
                "name": (categories[cid] if 0 <= cid < len(categories) else str(cid)),
                "tp": per_cls[cid]["tp"], "fp": per_cls[cid]["fp"], "fn": per_cls[cid]["fn"],
                "precision": prf(per_cls[cid]['tp'], per_cls[cid]['fp'], per_cls[cid]['fn'])[0],
                "recall":    prf(per_cls[cid]['tp'], per_cls[cid]['fp'], per_cls[cid]['fn'])[1],
                "f1":        prf(per_cls[cid]['tp'], per_cls[cid]['fp'], per_cls[cid]['fn'])[2],
            } for cid in per_cls
        }
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run()
