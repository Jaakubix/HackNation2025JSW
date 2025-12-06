#!/usr/bin/env python3
"""
Narzędzie do kalibracji pix → mm.
Pozwala wyznaczyć ile pikseli odpowiada jednemu milimetrowi na obrazie.

Użycie:
    python src/calibration.py --image kalibracja.jpg --reference-mm 1200
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CALIBRATION, PROJECT_ROOT


class CalibrationTool:
    """Interaktywne narzędzie do kalibracji."""
    
    def __init__(self, image_path: Path, reference_mm: float):
        self.image_path = image_path
        self.reference_mm = reference_mm
        self.points = []
        self.image = None
        self.display_image = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Obsługa kliknięć myszy."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                self.draw_points()
    
    def draw_points(self):
        """Rysuje punkty i linię na obrazie."""
        self.display_image = self.image.copy()
        
        for i, pt in enumerate(self.points):
            cv2.circle(self.display_image, pt, 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"P{i+1}", (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(self.points) == 2:
            cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
            distance_px = np.sqrt(
                (self.points[1][0] - self.points[0][0])**2 +
                (self.points[1][1] - self.points[0][1])**2
            )
            mid_x = (self.points[0][0] + self.points[1][0]) // 2
            mid_y = (self.points[0][1] + self.points[1][1]) // 2
            cv2.putText(self.display_image, f"{distance_px:.1f} px", (mid_x, mid_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Kalibracja", self.display_image)
    
    def run(self) -> dict:
        """
        Uruchamia narzędzie kalibracji.
        
        Returns:
            Słownik z wynikami kalibracji
        """
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Nie można wczytać obrazu: {self.image_path}")
        
        self.display_image = self.image.copy()
        
        print("\n=== KALIBRACJA PIX → MM ===")
        print(f"Obraz: {self.image_path}")
        print(f"Znana szerokość obiektu: {self.reference_mm} mm")
        print("\nInstrukcja:")
        print("1. Kliknij na początek i koniec obiektu o znanej szerokości")
        print("2. Naciśnij ENTER aby zaakceptować")
        print("3. Naciśnij 'r' aby zresetować punkty")
        print("4. Naciśnij ESC aby anulować")
        
        cv2.namedWindow("Kalibracja", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Kalibracja", self.mouse_callback)
        cv2.imshow("Kalibracja", self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            
            if key == ord('r'):  # Reset
                self.points = []
                self.display_image = self.image.copy()
                cv2.imshow("Kalibracja", self.display_image)
            
            if key == 13 and len(self.points) == 2:  # ENTER
                break
        
        cv2.destroyAllWindows()
        
        # Oblicz kalibrację
        distance_px = np.sqrt(
            (self.points[1][0] - self.points[0][0])**2 +
            (self.points[1][1] - self.points[0][1])**2
        )
        
        pixels_per_mm = distance_px / self.reference_mm
        mm_per_pixel = self.reference_mm / distance_px
        
        result = {
            "calibration_image": str(self.image_path),
            "reference_width_mm": self.reference_mm,
            "reference_width_px": distance_px,
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": mm_per_pixel,
            "points": self.points,
        }
        
        print(f"\n✓ Kalibracja zakończona:")
        print(f"  Odległość: {distance_px:.1f} px = {self.reference_mm} mm")
        print(f"  {pixels_per_mm:.4f} px/mm")
        print(f"  {mm_per_pixel:.4f} mm/px")
        
        return result


def calibrate_from_contour(
    image_path: Path,
    reference_mm: float,
    auto_detect: bool = True
) -> dict:
    """
    Automatyczna kalibracja przez detekcję konturu taśmy.
    
    Args:
        image_path: Ścieżka do obrazu
        reference_mm: Znana szerokość taśmy w mm
        auto_detect: Czy próbować automatycznej detekcji
    
    Returns:
        Słownik z wynikami kalibracji
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detekcja krawędzi
    edges = cv2.Canny(gray, 50, 150)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Nie znaleziono konturów, użyj trybu interaktywnego")
        return None
    
    # Znajdź największy kontur (prawdopodobnie taśma)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Szerokość w pikselach (zakładamy poziomą taśmę)
    width_px = w
    
    result = {
        "calibration_image": str(image_path),
        "reference_width_mm": reference_mm,
        "reference_width_px": width_px,
        "pixels_per_mm": width_px / reference_mm,
        "mm_per_pixel": reference_mm / width_px,
        "detection_method": "auto_contour",
        "bounding_box": {"x": x, "y": y, "w": w, "h": h},
    }
    
    return result


def save_calibration(calibration: dict, output_path: Path = None):
    """Zapisuje kalibrację do pliku JSON."""
    if output_path is None:
        output_path = PROJECT_ROOT / "calibration.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    
    print(f"✓ Zapisano kalibrację: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Kalibracja pix → mm")
    parser.add_argument("-i", "--image", type=Path, required=True,
                        help="Obraz do kalibracji")
    parser.add_argument("-r", "--reference-mm", type=float,
                        default=CALIBRATION["reference_width_mm"],
                        help="Znana szerokość obiektu w mm")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Plik wyjściowy JSON")
    parser.add_argument("--auto", action="store_true",
                        help="Próbuj automatycznej detekcji")
    
    args = parser.parse_args()
    
    if not args.image.exists():
        print(f"Błąd: Plik nie istnieje: {args.image}")
        sys.exit(1)
    
    if args.auto:
        result = calibrate_from_contour(args.image, args.reference_mm)
        if result is None:
            print("Automatyczna kalibracja nie powiodła się, przełączam na tryb interaktywny")
            tool = CalibrationTool(args.image, args.reference_mm)
            result = tool.run()
    else:
        tool = CalibrationTool(args.image, args.reference_mm)
        result = tool.run()
    
    if result:
        save_calibration(result, args.output)


if __name__ == "__main__":
    main()
