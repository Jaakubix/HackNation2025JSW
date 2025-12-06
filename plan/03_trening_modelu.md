# 3) Trening modelu
- Dataset/loader: `src/data/dataset.py` dla COCO; `collate_fn` pod detection. Transformacje (augmentacje) parametryzowane w configu.
- Model: `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")`; ustaw liczbę klas wg anotacji. Faza 1: zamroź backbone (head-only), Faza 2: odblokuj całość.
- Hiperparametry start: lr 1e-3 (head), potem 5e-4 (full), batch_size 2–4 (wg GPU), epochs 20–50, scheduler `StepLR` (np. step 10, gamma 0.1).
- Trening: skrypt `src/train.py` (argumenty: ścieżki anotacji, ścieżki output, parametry). Loguj loss, mAP/mAR per klasa. Checkpointy do `models/` (najlepszy po val mAP + co X epok).
- Walidacja: co epokę inferencja na `val`, zapis wizualizacji do `outputs/vis/val_ep{n}/`.
- Early stopping/manual tuning jeśli brak poprawy po kilku epokach; sprawdź class imbalance (ew. weights/sampler).

