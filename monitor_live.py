"""
Monitor Live - Skrypt do monitorowania taśmy 24/7
Obsługuje źródła: kamera, RTSP stream, pliki wideo

Użycie:
    python monitor_live.py --source rtsp://192.168.1.100:554/stream
    python monitor_live.py --source /dev/video0
    python monitor_live.py --source video.mp4 --csv output/cycles.csv
"""

import cv2
import torch
import numpy as np
import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils import upload_envs, selectDevice
from torchvision.transforms import functional as F

# Load envs
upload_envs()

from train.upload.model import getPretrainedModel
from belt_monitor import BeltMonitor
from alert_system import AlertType


def load_model(model_path: str, num_classes: int, device: torch.device):
    """Ładuje model z pliku."""
    print(f"Ładowanie modelu z {model_path}...")
    model = getPretrainedModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_prediction(model, img_tensor, threshold=0.5):
    """Wykonuje predykcję na obrazie."""
    with torch.no_grad():
        outputs = model([img_tensor])
    
    output = outputs[0]
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    keep = scores >= threshold
    return boxes[keep], scores[keep], labels[keep]


class FilterState:
    """Stan filtra do wykrywania szwów."""
    
    def __init__(self):
        self.avg_float = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process(self, frame):
        gamma = 1.4
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_frame = cv2.LUT(frame, table)
        
        gray = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        if self.avg_float is None:
            self.avg_float = np.float32(gray)
        
        cv2.accumulateWeighted(gray, self.avg_float, 0.1)
        background = cv2.convertScaleAbs(self.avg_float)
        
        diff = cv2.absdiff(gray, background)
        diff_boost = cv2.multiply(diff, 5.0)
        soft_filtered = cv2.GaussianBlur(diff_boost, (5, 5), 0)
        
        return cv2.cvtColor(soft_filtered, cv2.COLOR_GRAY2BGR)


def draw_detections_polygon(img, boxes, scores, labels, color, label_map, is_belt=False):
    """
    Rysuje detekcje na obrazie używając wielokątów.
    Dla taśmy używamy trapezu żeby lepiej oddać perspektywę.
    """
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        
        if is_belt:
            # Dla taśmy - rysuj trapez (perspektywa)
            # Górna krawędź węższa, dolna szersza (lub odwrotnie)
            margin = int((x2 - x1) * 0.05)  # 5% margines
            
            # Punkty wielokąta (trapez)
            pts = np.array([
                [x1 + margin, y1],      # Lewy górny
                [x2 - margin, y1],      # Prawy górny
                [x2, y2],               # Prawy dolny
                [x1, y2]                # Lewy dolny
            ], np.int32)
            
            # Rysuj wypełniony wielokąt z przezroczystością
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            
            # Rysuj kontur wielokąta
            cv2.polylines(img, [pts], True, color, 2)
            
            # Pomiar szerokości - linia w środku
            mid_y = (y1 + y2) // 2
            cv2.line(img, (x1, mid_y), (x2, mid_y), (255, 255, 0), 1)
            
            width = x2 - x1
            cv2.putText(img, f"Szer: {width}px", (x1 + 5, mid_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # Dla szwów - prostokąt z zaokrąglonymi rogami (symulacja)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Etykieta
        label_name = label_map.get(label.item(), str(label.item()))
        text = f"{label_name}: {score:.2f}"
        
        # Tło dla tekstu
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img


def draw_detections(img, boxes, scores, labels, color, label_map):
    """Wrapper dla kompatybilności - używa wielokątów dla taśmy."""
    is_belt = "Tasma" in label_map.values() or "tasma" in label_map.values()
    return draw_detections_polygon(img, boxes, scores, labels, color, label_map, is_belt)


def draw_status(frame, monitor: BeltMonitor, alerts_text: str = ""):
    """Rysuje status monitoringu na klatce."""
    status = monitor.get_current_status()
    h, w = frame.shape[:2]
    
    # Tło dla statusu
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Tekst statusu
    y = 35
    cv2.putText(frame, f"Cykl: #{status['current_cycle_id']}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 25
    cv2.putText(frame, f"Klatka: {status['frame_count']}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25
    cv2.putText(frame, f"Zakonczone cykle: {status['completed_cycles']}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25
    cv2.putText(frame, f"Alerty: {status['total_alerts']}", (20, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Alert na ekranie
    if alerts_text:
        cv2.rectangle(frame, (10, h - 60), (w - 10, h - 10), (0, 0, 255), -1)
        cv2.putText(frame, alerts_text[:80], (20, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def run_monitoring(
    source: str,
    csv_path: str = "output/cycles.csv",
    output_video: Optional[str] = None,
    show_preview: bool = True,
    tape_model_path: str = "finetuned_models/tape_model.pth",
    seam_model_path: str = "finetuned_models/seam_model.pth"
):
    """
    Główna funkcja monitorowania.
    
    Args:
        source: Źródło wideo (plik, kamera, RTSP)
        csv_path: Ścieżka do pliku CSV
        output_video: Opcjonalna ścieżka do zapisu wideo
        show_preview: Czy pokazywać podgląd
        tape_model_path: Ścieżka do modelu taśmy
        seam_model_path: Ścieżka do modelu szwów
    """
    device = selectDevice()
    
    # Ładowanie modeli
    print("\n=== BELT MONITOR - System Monitorowania Taśmy ===\n")
    
    if not Path(tape_model_path).exists():
        print(f"UWAGA: Model taśmy nie znaleziony: {tape_model_path}")
        print("Uruchamiam w trybie demo (bez detekcji ML)")
        tape_model = None
    else:
        tape_model = load_model(tape_model_path, num_classes=2, device=device)
    
    if not Path(seam_model_path).exists():
        print(f"UWAGA: Model szwów nie znaleziony: {seam_model_path}")
        seam_model = None
    else:
        seam_model = load_model(seam_model_path, num_classes=2, device=device)
    
    # Inicjalizacja monitora
    monitor = BeltMonitor(csv_path=csv_path)
    filter_state = FilterState()
    
    # Otwórz źródło wideo
    print(f"\nŁączenie ze źródłem: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"BŁĄD: Nie można otworzyć źródła: {source}")
        print("Próba ponownego połączenia za 5 sekund...")
        time.sleep(5)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Nie udało się połączyć. Zakończenie.")
            return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"Rozdzielczość: {w}x{h} @ {fps:.1f} FPS")
    print(f"CSV: {csv_path}")
    print("\nRozpoczynam monitorowanie... (Ctrl+C aby zakończyć)\n")
    
    # Opcjonalny zapis wideo
    out = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        print(f"Zapis wideo do: {output_video}")
    
    monitor.is_running = True
    last_alert_text = ""
    alert_display_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                # Dla plików wideo - koniec
                if not source.startswith("rtsp") and not source.startswith("/dev"):
                    print("\nKoniec pliku wideo.")
                    break
                
                # Dla streamów - spróbuj ponownie
                print("\nUtrata połączenia. Ponowne łączenie...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue
            
            # Detekcja taśmy
            if tape_model:
                img_tensor = F.to_tensor(frame).to(device)
                tape_boxes, tape_scores, tape_labels = get_prediction(tape_model, img_tensor)
            else:
                # Tryb demo - symuluj detekcję
                tape_boxes = torch.tensor([[100, 100, w-100, h-100]])
                tape_scores = torch.tensor([0.9])
                tape_labels = torch.tensor([1])
            
            # Detekcja szwów
            if seam_model:
                filtered_frame = filter_state.process(frame)
                filtered_tensor = F.to_tensor(filtered_frame).to(device)
                seam_boxes, seam_scores, seam_labels = get_prediction(seam_model, filtered_tensor)
            else:
                seam_boxes = torch.tensor([])
                seam_scores = torch.tensor([])
                seam_labels = torch.tensor([])
            
            # Przetwórz przez monitor
            cycle_data, alerts = monitor.process_frame(
                tape_boxes.cpu().numpy() if hasattr(tape_boxes, 'cpu') else tape_boxes.numpy(),
                tape_scores.cpu().numpy() if hasattr(tape_scores, 'cpu') else tape_scores.numpy(),
                seam_boxes.cpu().numpy() if hasattr(seam_boxes, 'cpu') else seam_boxes.numpy(),
                seam_scores.cpu().numpy() if hasattr(seam_scores, 'cpu') else seam_scores.numpy()
            )
            
            # Obsługa alertów
            if alerts:
                for alert in alerts:
                    severity_color = {
                        "CRITICAL": "\033[91m",  # Red
                        "HIGH": "\033[93m",      # Yellow
                        "MEDIUM": "\033[94m",    # Blue
                        "LOW": "\033[90m"        # Gray
                    }
                    reset = "\033[0m"
                    color = severity_color.get(alert.severity, "")
                    print(f"{color}[{alert.severity}] {alert.type.value}: {alert.message}{reset}")
                    last_alert_text = f"[{alert.severity}] {alert.message}"
                    alert_display_frames = int(fps * 3)  # Wyświetl przez 3 sekundy
            
            if alert_display_frames > 0:
                alert_display_frames -= 1
            else:
                last_alert_text = ""
            
            # Logowanie cyklu
            if cycle_data:
                print(f"\n✓ Cykl #{cycle_data['cycle_id']} zakończony:")
                print(f"  Max szerokość: {cycle_data['max_width']:.1f}")
                print(f"  Min szerokość: {cycle_data['min_width']:.1f}")
                print(f"  Szwy: {cycle_data['seam_count']}")
                if cycle_data['alerts']:
                    print(f"  Alerty: {len(cycle_data['alerts'])}")
                print()
            
            # Rysowanie na klatce
            if tape_model:
                frame = draw_detections(frame, tape_boxes, tape_scores, tape_labels, 
                                       (0, 255, 0), {1: "Tasma"})
            if seam_model:
                frame = draw_detections(frame, seam_boxes, seam_scores, seam_labels, 
                                       (0, 0, 255), {1: "Szew"})
            
            frame = draw_status(frame, monitor, last_alert_text)
            
            # Zapis wideo
            if out:
                out.write(frame)
            
            # Podgląd
            if show_preview:
                cv2.imshow("Belt Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nZakończono przez użytkownika.")
                    break
                elif key == ord('s'):
                    # Zapisz screenshot
                    screenshot_path = f"screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\n\nZakończono przez użytkownika (Ctrl+C)")
    
    finally:
        # Finalizacja
        monitor.is_running = False
        final_cycle = monitor.finalize()
        if final_cycle:
            print(f"\nOstatni cykl #{final_cycle['cycle_id']} zapisany.")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Podsumowanie
        status = monitor.get_current_status()
        print("\n=== PODSUMOWANIE ===")
        print(f"Przetworzone klatki: {status['frame_count']}")
        print(f"Zakończone cykle: {status['completed_cycles']}")
        print(f"Wygenerowane alerty: {status['total_alerts']}")
        print(f"CSV zapisany do: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Belt Monitor - System monitorowania taśmy górniczej 24/7"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Źródło wideo: ścieżka do pliku, /dev/video0, lub rtsp://..."
    )
    parser.add_argument(
        "--csv", "-c",
        default="output/cycles.csv",
        help="Ścieżka do pliku CSV (domyślnie: output/cycles.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Opcjonalna ścieżka do zapisu wideo wyjściowego"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Wyłącz podgląd (tryb headless)"
    )
    parser.add_argument(
        "--tape-model",
        default="finetuned_models/tape_model.pth",
        help="Ścieżka do modelu taśmy"
    )
    parser.add_argument(
        "--seam-model",
        default="finetuned_models/seam_model.pth",
        help="Ścieżka do modelu szwów"
    )
    
    args = parser.parse_args()
    
    run_monitoring(
        source=args.source,
        csv_path=args.csv,
        output_video=args.output,
        show_preview=not args.no_preview,
        tape_model_path=args.tape_model,
        seam_model_path=args.seam_model
    )


if __name__ == "__main__":
    main()
