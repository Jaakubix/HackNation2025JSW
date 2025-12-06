"""
Konfiguracja projektu Kopalnia - monitorowanie taśmy górniczej
"""
from pathlib import Path

# ============ ŚCIEŻKI ============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset_combined"
IMAGES_DIR = DATASET_DIR
ANNOTATIONS_FILE = DATASET_DIR / "coco_annotations.json"
FRAMES_DIR = DATA_DIR / "frames"
VIDEOS_DIR = DATA_DIR / "videos"

# ============ EKSTRAKCJA KLATEK ============
FRAME_EXTRACTION = {
    "interval_seconds": 1.0,      # Wyciągaj klatkę co N sekund (None = użyj frame_step)
    "frame_step": None,           # Wyciągaj co N-tą klatkę (None = użyj interval_seconds)
    "output_format": "jpg",       # Format wyjściowy: jpg, png
    "image_quality": 95,          # Jakość JPEG (1-100)
    "resize": None,               # Rozmiar wyjściowy (width, height) lub None
}


# ============ KLASY OBIEKTÓW (COCO) ============
CLASSES = {
    0: "background",
    1: "szew",
    2: "tasma",
}

# Odwrotne mapowanie nazwa -> id
CLASS_NAME_TO_ID = {v: k for k, v in CLASSES.items()}

# ============ PODZIAŁ DANYCH ============
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}

# ============ AUGMENTACJE ============
AUGMENTATION = {
    "enabled": True,
    "blur": {"enabled": True, "kernel_range": (3, 7)},
    "brightness": {"enabled": True, "range": (0.7, 1.3)},
    "contrast": {"enabled": True, "range": (0.8, 1.2)},
    "noise": {"enabled": True, "std_range": (5, 25)},
    "rotation": {"enabled": False, "angle_range": (-5, 5)},  # Lekkie rotacje
    "horizontal_flip": {"enabled": False},  # Wyłączone - taśma ma kierunek
}


# ============ TRENING MODELU ============
TRAINING = {
    "batch_size": 4,
    "learning_rate": 0.005,
    "num_epochs": 50,
    "num_workers": 4,
    "checkpoint_dir": PROJECT_ROOT / "models",
}

# ============ INFERENCJA ============
INFERENCE = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "device": "cuda",  # "cuda" lub "cpu"
}
