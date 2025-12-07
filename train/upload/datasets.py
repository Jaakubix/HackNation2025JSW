import os
import json
from pathlib import Path
import shutil
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets

from clear_data import main as cleardata_run

# --- ENV / ścieżki ---
TESTDATA_DIR = Path(os.getenv("TESTDATA_DIR", "./testdata"))
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))

import numpy as np

# ===== Dataset helper: COCO -> (image, target dict) =====
class CocoDetAsTargets(Dataset):
    """
    Owija torchvision.datasets.CocoDetection i zwraca:
      image: Tensor[C,H,W]
      target: dict(boxes=Tensor[N,4], labels=Tensor[N], masks=Tensor[N,H,W]) w formacie xyxy.
    Filtruje boksy o zerowej/ujemnej szer./wys. i klamruje do granic obrazu.
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
            w, h = img.size  # PIL: (W,H)

        boxes_list, labels_list = [], []
        masks_list = []

        for a in anns:
            x, y, ww, hh = a["bbox"]  # COCO: xywh
            if ww is None or hh is None or ww <= 0 or hh <= 0:
                continue
            x1, y1 = x, y
            x2, y2 = x + ww, y + hh

            # klamrowanie do [0, W-1]/[0, H-1]
            x1 = max(0.0, min(float(x1), w - 1.0))
            y1 = max(0.0, min(float(y1), h - 1.0))
            x2 = max(0.0, min(float(x2), w - 1.0))
            y2 = max(0.0, min(float(y2), h - 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(int(a["category_id"]))
            
            # MASKI: generowanie maski binarnej dla obiektu
            # annToMask zwraca np.array (h, w) z 0 i 1
            mask = self.coco.annToMask(a)
            masks_list.append(mask)

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            masks = torch.tensor(np.array(masks_list), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return img, target


# ===== Augmentacja danych dla detekcji =====
class DetectionAugmentation:
    """
    Augmentacja obrazu i targetów (bboxów + masek) dla treningu detekcji.
    Obsługuje: horizontal flip, color jitter.
    """
    def __init__(self, hflip_prob=0.5, color_jitter=True):
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        if color_jitter:
            self.jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )

    def __call__(self, image, target):
        """
        Args:
            image: Tensor [C, H, W]
            target: dict z 'boxes', 'labels', 'masks'
        Returns:
            augmented image, augmented target
        """
        # Horizontal flip
        if torch.rand(1).item() < self.hflip_prob:
            image = transforms.functional.hflip(image)
            
            # Flip boxes (x1, y1, x2, y2)
            if target["boxes"].numel() > 0:
                w = image.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]  # flip x coords
                target["boxes"] = boxes
            
            # Flip masks
            if target["masks"].numel() > 0:
                target["masks"] = torch.flip(target["masks"], dims=[-1])
        
        # Color jitter (only on image, not targets)
        if self.color_jitter and torch.rand(1).item() < 0.5:
            image = self.jitter(image)
        
        return image, target


class AugmentedDataset(Dataset):
    """Wrapper dodający augmentację do istniejącego datasetu."""
    def __init__(self, base_dataset, augmentation=None):
        self.base = base_dataset
        self.aug = augmentation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        if self.aug is not None:
            img, target = self.aug(img, target)
        return img, target

    @property
    def coco(self):
        return self.base.coco if hasattr(self.base, 'coco') else None


def _export_coco_subset(coco, img_ids, source_root: Path, dest_root: Path):
    """
    Zapisuje podzbiór COCO (images+annotations+categories) do dest_root
    oraz kopiuje tam odpowiadające obrazy.
    """
    dest_root.mkdir(parents=True, exist_ok=True)

    images = coco.loadImgs(img_ids)

    ann_ids = []
    for iid in img_ids:
        ann_ids.extend(coco.getAnnIds(imgIds=[iid]))
    annotations = coco.loadAnns(ann_ids)

    categories = coco.loadCats(coco.getCatIds())

    out_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (dest_root / "_annotations.coco.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for im in images:
        src = source_root / im["file_name"]
        dst = dest_root / im["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)


def det_collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# ===== Dane =====
def getCOCODataset():
    """
    Wczytuje niepodzielony zbiór COCO z katalogu DATA_ROOT (obrazy + _annotations.coco.json),
    następnie LOSOWO dzieli go na train/val/test wg TRAIN_SIZE i VALID_SIZE (reszta -> test)
    i zwraca: train_loader, val_loader, test_loader, num_classes
    """
    # Akceptuj zarówno DATA_ROOT jak i DATA_ROOT/data
    data_dir = DATA_ROOT / "data" if (DATA_ROOT / "data").exists() else DATA_ROOT
    
    # Custom overrides for dual training
    custom_ann = os.getenv("CUSTOM_ANN_FILE")
    custom_img_dir = os.getenv("CUSTOM_IMG_DIR")
    
    if custom_ann:
        ann_file = data_dir / custom_ann
    else:
        ann_file = data_dir / "_annotations.coco.json"
        
    if custom_img_dir:
        # If custom img dir is provided, we use it as root for CocoDetAsTargets
        # But data_dir is still used for other things?
        # CocoDetAsTargets takes 'root'.
        pass

    if not data_dir.exists() or not ann_file.exists():
        raise FileNotFoundError(f"Brak katalogu/plików adnotacji COCO: {data_dir} lub {ann_file}")

    tf = transforms.Compose([transforms.ToTensor()])

    # Pełny dataset (bez podziału)
    # Use custom_img_dir if set, else data_dir
    img_root = str(custom_img_dir) if custom_img_dir else str(data_dir)
    full_ds = CocoDetAsTargets(img_root, str(ann_file), tf=tf)

    # Liczba klas = liczba kategorii + 1 (tło)
    cat_ids = sorted(full_ds.coco.getCatIds())
    num_classes = len(cat_ids) + 1

    n = len(full_ds)
    if n == 0:
        raise RuntimeError("Zbiór danych jest pusty.")

    # Frakcje z .env (domyślnie 0.7 / 0.2)
    train_frac = float(os.getenv("TRAIN_SIZE", "0.7"))
    valid_frac = float(os.getenv("VALID_SIZE", "0.2"))

    # Bezpieczne ograniczenia
    train_frac = max(0.0, min(train_frac, 1.0))
    valid_frac = max(0.0, min(valid_frac, max(0.0, 1.0 - train_frac)))

    n_train = int(n * train_frac)
    n_val   = int(n * valid_frac)
    n_test  = max(0, n - n_train - n_val)

    # --- Losowanie indeksów ---
    # Jeśli USE_DETERMINISTIC=1 -> użyj SEED; w przeciwnym razie każdorazowo inne losowanie.
    use_det = os.getenv("USE_DETERMINISTIC", "0") == "1"
    seed = int(os.getenv("SEED", "1234")) if use_det else None
    g = torch.Generator()
    if seed is not None:
        g = g.manual_seed(seed)
        print(f"[split] Deterministyczny podział z SEED={seed}")
    else:
        print("[split] Niedeterministyczny podział (inny za każdym uruchomieniem)")

    indices = torch.randperm(n, generator=g).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # Utwórz zbiory
    train_ds_raw = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)
    
    # Augmentacja tylko na zbiorze treningowym
    aug = DetectionAugmentation(hflip_prob=0.5, color_jitter=True)
    train_ds = AugmentedDataset(train_ds_raw, augmentation=aug)
    print(f"[augmentation] Włączono augmentację dla zbioru treningowego (hflip, color jitter)")

    # Zapisz informacyjnie podział (ID obrazów + indeksy)
    try:
        img_ids_all = full_ds.inner.ids  # ID obrazów w kolejności datasetu
        payload = {
            "counts": {"total": n, "train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "indices": {"train": train_idx, "val": val_idx, "test": test_idx},
            "image_ids": {
                "train": [img_ids_all[i] for i in train_idx],
                "val":   [img_ids_all[i] for i in val_idx],
                "test":  [img_ids_all[i] for i in test_idx],
            },
            "fractions": {"train": train_frac, "val": valid_frac, "test": 1.0 - train_frac - valid_frac},
            "seed_used": seed if seed is not None else "random_each_run",
        }
        (data_dir / "splits.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[train] Ostrzeżenie: nie udało się zapisać splits.json: {e}")

    # Wyczyść ./testdata i wyeksportuj subset testowy (żeby moduł test mógł użyć gotowych danych)
    try:
        cleardata_run("testdata")
        print("[train] Wyczyszczono folder testdata przed zapisem.")
    except Exception as e:
        print(f"[train] Ostrzeżenie: nie udało się wyczyścić testdata: {e}")

    try:
        test_img_ids = [full_ds.inner.ids[i] for i in test_idx]
        _export_coco_subset(full_ds.coco, test_img_ids, Path(data_dir), TESTDATA_DIR)
        print(f"[train] Zapisano zestaw testowy do: {TESTDATA_DIR}")
    except Exception as e:
        print(f"[train] Ostrzeżenie: nie udało się zapisać testu do {TESTDATA_DIR}: {e}")

    # Loadery
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=det_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=det_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=det_collate_fn, pin_memory=True
    )

    return train_loader, val_loader, test_loader, num_classes
