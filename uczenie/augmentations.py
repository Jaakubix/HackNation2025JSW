#!/usr/bin/env python3
"""
Moduł augmentacji danych dla obrazów taśmy górniczej.
Symuluje warunki pracy 24/7: różne oświetlenie, zabrudzenia, szumy.

Użycie jako skrypt:
    python src/augmentations.py --input data/frames/video --output data/augmented --count 3

Użycie jako moduł:
    from augmentations import AugmentationPipeline
    pipeline = AugmentationPipeline()
    augmented = pipeline.apply(image)
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUGMENTATION


class AugmentationPipeline:
    """Pipeline augmentacji obrazów."""
    
    def __init__(self, config: dict = None):
        """
        Inicjalizacja pipeline.
        
        Args:
            config: Słownik konfiguracji augmentacji (domyślnie z config.py)
        """
        self.config = config or AUGMENTATION
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Rozmycie gaussowskie."""
        cfg = self.config["blur"]
        if not cfg["enabled"]:
            return image
        
        kernel_range = cfg["kernel_range"]
        # Kernel musi być nieparzysty
        kernel_size = random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """Zmiana jasności."""
        cfg = self.config["brightness"]
        if not cfg["enabled"]:
            return image
        
        factor = random.uniform(*cfg["range"])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def apply_contrast(self, image: np.ndarray) -> np.ndarray:
        """Zmiana kontrastu."""
        cfg = self.config["contrast"]
        if not cfg["enabled"]:
            return image
        
        factor = random.uniform(*cfg["range"])
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """Dodanie szumu gaussowskiego."""
        cfg = self.config["noise"]
        if not cfg["enabled"]:
            return image
        
        std = random.uniform(*cfg["std_range"])
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def apply_rotation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Lekka rotacja obrazu.
        
        Returns:
            Tuple (obraz, kąt rotacji w stopniach)
        """
        cfg = self.config["rotation"]
        if not cfg["enabled"]:
            return image, 0.0
        
        angle = random.uniform(*cfg["angle_range"])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated, angle
    
    def apply_horizontal_flip(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Odwrócenie poziome (jeśli włączone).
        
        Returns:
            Tuple (obraz, czy odwrócono)
        """
        cfg = self.config["horizontal_flip"]
        if not cfg["enabled"]:
            return image, False
        
        if random.random() < 0.5:
            return cv2.flip(image, 1), True
        return image, False
    
    def apply(
        self,
        image: np.ndarray,
        apply_all: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Stosuje losowe augmentacje do obrazu.
        
        Args:
            image: Obraz wejściowy (BGR)
            apply_all: Jeśli True, stosuje wszystkie włączone augmentacje
        
        Returns:
            Tuple (augmentowany obraz, słownik zastosowanych transformacji)
        """
        if not self.config["enabled"]:
            return image.copy(), {}
        
        result = image.copy()
        applied = {}
        
        # Lista transformacji (bez rotacji i flip - mają wpływ na bbox)
        transforms = [
            ("blur", self.apply_blur),
            ("brightness", self.apply_brightness),
            ("contrast", self.apply_contrast),
            ("noise", self.apply_noise),
        ]
        
        for name, transform in transforms:
            if self.config[name]["enabled"]:
                if apply_all or random.random() < 0.5:
                    result = transform(result)
                    applied[name] = True
        
        # Rotacja i flip (zwracają dodatkowe info)
        if self.config["rotation"]["enabled"]:
            if apply_all or random.random() < 0.3:
                result, angle = self.apply_rotation(result)
                if angle != 0:
                    applied["rotation"] = angle
        
        if self.config["horizontal_flip"]["enabled"]:
            if apply_all or random.random() < 0.5:
                result, flipped = self.apply_horizontal_flip(result)
                if flipped:
                    applied["horizontal_flip"] = True
        
        return result, applied


def augment_directory(
    input_dir: Path,
    output_dir: Path,
    augmentations_per_image: int = 3,
    config: dict = None
) -> int:
    """
    Tworzy augmentacje dla wszystkich obrazów w katalogu.
    
    Args:
        input_dir: Katalog źródłowy z obrazami
        output_dir: Katalog wyjściowy
        augmentations_per_image: Liczba augmentacji na obraz
        config: Konfiguracja augmentacji
    
    Returns:
        Liczba utworzonych obrazów
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"Nie znaleziono obrazów w: {input_dir}")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = AugmentationPipeline(config)
    
    created = 0
    for img_path in tqdm(image_files, desc="Augmentacja"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Zachowaj oryginał
        orig_output = output_dir / f"{img_path.stem}_orig{img_path.suffix}"
        cv2.imwrite(str(orig_output), image)
        created += 1
        
        # Utwórz augmentacje
        for i in range(augmentations_per_image):
            augmented, applied = pipeline.apply(image)
            aug_output = output_dir / f"{img_path.stem}_aug{i:02d}{img_path.suffix}"
            cv2.imwrite(str(aug_output), augmented)
            created += 1
    
    print(f"\n✓ Utworzono {created} obrazów w {output_dir}")
    return created


def main():
    parser = argparse.ArgumentParser(description="Augmentacja obrazów")
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Katalog wejściowy z obrazami")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Katalog wyjściowy")
    parser.add_argument("-c", "--count", type=int, default=3,
                        help="Liczba augmentacji na obraz (domyślnie: 3)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Błąd: Katalog nie istnieje: {args.input}")
        sys.exit(1)
    
    augment_directory(args.input, args.output, args.count)


if __name__ == "__main__":
    main()
