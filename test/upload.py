import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import utils

# ===== Konfiguracja =====
DEVICE = utils.selectDevice()
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./dataset"))

THIS_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.getenv("OUTPUT_DIR", str(THIS_DIR.parent / "finetuned_models")))
MODEL_NAME = os.environ.get("TEST_MODEL", os.environ["TARGET_MODEL_NAME"] + "_" + os.environ["MODEL"] + "_model") + ".pth"
WEIGHTS_PATH = WEIGHTS_DIR / f"{MODEL_NAME}"
TESTDATA_DIR = Path(os.getenv("TESTDATA_DIR", str(THIS_DIR.parent / "testdata")))
OUTPUT_DIR = Path(os.getenv("DETECTIONS_DIR", str(THIS_DIR.parent / "detections_out")))

# ===== Dataset zgodny z treningiem (z filtracją boksów) =====
class CocoDetAsTargets(Dataset):
    """
    Zwraca (image, target) z target={"boxes","labels"} w XYXY.
    Odfiltrowuje zerowe/ujemne rozmiary i klamruje boxy do rozmiaru obrazu.
    """
    def __init__(self, root: str, ann_file: str, tf=None):
        self.inner = datasets.CocoDetection(root=root, annFile=ann_file, transform=tf)
        self.coco = self.inner.coco

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, anns = self.inner[idx]

        # rozmiar obrazu
        if isinstance(img, torch.Tensor):
            h, w = int(img.shape[-2]), int(img.shape[-1])
        else:
            w, h = img.size  # PIL

        boxes_list, labels_list = [], []

        for a in anns:
            x, y, ww, hh = a["bbox"]
            if ww is None or hh is None or ww <= 0 or hh <= 0:
                continue
            x1, y1 = x, y
            x2, y2 = x + ww, y + hh

            # klamruj do granic obrazu
            x1 = max(0.0, min(float(x1), w - 1.0))
            y1 = max(0.0, min(float(y1), h - 1.0))
            x2 = max(0.0, min(float(x2), w - 1.0))
            y2 = max(0.0, min(float(y2), h - 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(int(a["category_id"]))

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target


def det_collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# ===== Kategorie =====
def build_category_names_list(coco) -> list[str]:
    cat_ids = sorted(coco.getCatIds())
    max_id = max(cat_ids) if cat_ids else 0
    names = ["background"] + [""] * max_id
    cats = coco.loadCats(cat_ids)
    id_to_name = {c["id"]: c["name"] for c in cats}
    for cid, name in id_to_name.items():
        if cid <= max_id:
            names[cid] = name
    return names


# ===== Budowa modelu (Mask R-CNN) =====
def _import_model_builders():
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as maskrcnn_fn
        has_v2 = True
    except Exception:
        from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn_fn
        has_v2 = False

    try:
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor as _FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except Exception:
        _FastRCNNPredictor = None
        MaskRCNNPredictor = None

    try:
        if has_v2:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights as _Weights
            default_weights = _Weights.COCO_V1
        else:
            default_weights = None
    except Exception:
        default_weights = None

    return maskrcnn_fn, has_v2, _FastRCNNPredictor, MaskRCNNPredictor, default_weights


class SimpleFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)
    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, 1)
        return self.cls_score(x), self.bbox_pred(x)


def _swap_predictor(model, num_classes: int, FastRCNNPredictor, MaskRCNNPredictor):
    # 1. Głowica BBox
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    if FastRCNNPredictor is not None:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model.roi_heads.box_predictor = SimpleFastRCNNPredictor(in_features, num_classes)

    # 2. Głowica Maski
    if hasattr(model.roi_heads, "mask_predictor") and MaskRCNNPredictor is not None:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


def build_inference_model(num_classes: int):
    maskrcnn_fn, _, FastRCNNPredictor, MaskRCNNPredictor, DEFAULT_WEIGHTS = _import_model_builders()
    kwargs_common = dict(
        min_size=480, max_size=640,
        rpn_pre_nms_top_n_train=200, rpn_post_nms_top_n_train=100,
        rpn_pre_nms_top_n_test=100,  rpn_post_nms_top_n_test=50,
        rpn_batch_size_per_image=64, box_batch_size_per_image=128,
        trainable_backbone_layers=2,
    )

    if DEFAULT_WEIGHTS is not None:
        model = maskrcnn_fn(weights=DEFAULT_WEIGHTS, **kwargs_common)
    else:
        model = maskrcnn_fn(pretrained=True, **kwargs_common)
    
    return _swap_predictor(model, num_classes, FastRCNNPredictor, MaskRCNNPredictor)


def _infer_num_classes_from_state(state_dict: dict) -> int:
    for k in [
        "roi_heads.box_predictor.cls_score.weight",
        "module.roi_heads.box_predictor.cls_score.weight",
    ]:
        if k in state_dict:
            return state_dict[k].shape[0]
    raise KeyError("Brak cls_score.weight w checkpointcie – nie mogę wywnioskować liczby klas.")


def load_trained_model() -> tuple[torch.nn.Module, int]:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono wag: {WEIGHTS_PATH}")
    print(f"Ładuję wagi: {WEIGHTS_PATH}")
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    trained_num_classes = _infer_num_classes_from_state(state)
    model = build_inference_model(trained_num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Uwaga: load_state_dict – missing={missing}, unexpected={unexpected}")
    model.to(DEVICE).eval()
    return model, trained_num_classes


# ===== Dane testowe – ten sam podział co w treningu =====
def _split_indices(n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)

    train_frac = float(os.environ.get("TRAIN_SIZE", "0.7"))
    valid_frac = float(os.environ.get("VALID_SIZE", "0.2"))

    train_frac = max(0.0, min(train_frac, 1.0))
    valid_frac = max(0.0, min(valid_frac, max(0.0, 1.0 - train_frac)))

    n_train = int(n * train_frac)
    n_val   = int(n * valid_frac)
    n_test  = max(0, n - n_train - n_val)

    indices = torch.randperm(n, generator=g).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _prepare_detections_out():
    """
    Czyści detections_out (i testdata – zgodnie z logiką ClearData dla all='nie')
    oraz upewnia się, że katalog istnieje.
    """
    try:
        from clear_data import main as cleardata_run
        cleardata_run("detections")  # all="nie" -> czyść detections_out i testdata
        print("[test] Wyczyszczono detections_out (tryb 'detections').")
    except Exception as e:
        print(f"[test] Ostrzeżenie: nie udało się wyczyścić detections_out: {e}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_test_dataset_and_classes() -> tuple[DataLoader, list[str], list[str]]:
    """
    Buduje DataLoader testowy. Preferuje gotowy eksport w TESTDATA_DIR,
    a jeśli go nie ma – odtwarza podział z pełnego COCO.
    Dodatkowo: PRZED zapisywaniem wyników czyszczony jest detections_out.
    """
    # Wyczyść detections_out/testdata przed generowaniem wyników
    _prepare_detections_out()

    # 1) Spróbuj gotowego eksportu
    td_ann = TESTDATA_DIR / "_annotations.coco.json"
    if TESTDATA_DIR.exists() and td_ann.exists():
        tf = transforms.Compose([transforms.ToTensor()])
        ds = CocoDetAsTargets(str(TESTDATA_DIR), str(td_ann), tf=tf)
        categories = build_category_names_list(ds.coco)

        # nazwy plików w kolejności datasetu
        img_files = [im["file_name"] for im in ds.coco.loadImgs(ds.coco.getImgIds())]
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=det_collate_fn)
        print(f"[test] Wczytano test z {TESTDATA_DIR}")
        return loader, categories, img_files

    # 2) Fallback – odtwórz podział jak w treningu
    data_dir = DATA_ROOT / "data" if (DATA_ROOT / "data").exists() else DATA_ROOT
    ann_file = data_dir / "_annotations.coco.json"
    if not data_dir.exists() or not ann_file.exists():
        raise FileNotFoundError(f"Brak katalogu/plików COCO: {data_dir} lub {ann_file}")

    tf = transforms.Compose([transforms.ToTensor()])
    full_ds = CocoDetAsTargets(str(data_dir), str(ann_file), tf=tf)
    categories = build_category_names_list(full_ds.coco)

    try:
        seed = int(os.environ.get("SEED", "1234"))
    except Exception:
        seed = 1234
    n = len(full_ds)
    _, _, test_idx = _split_indices(n, seed)
    test_ds = Subset(full_ds, test_idx)

    img_ids = full_ds.inner.ids
    img_files = []
    for i in test_idx:
        img_id = img_ids[i]
        file_name = full_ds.coco.loadImgs([img_id])[0]["file_name"]
        img_files.append(file_name)

    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=det_collate_fn)
    print(f"[test] Odtworzono test z pełnego zbioru (brak {TESTDATA_DIR}).")
    return loader, categories, img_files
