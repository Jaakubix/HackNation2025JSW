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
    masks = output.get('masks')
    
    keep = scores >= threshold
    if masks is not None:
        masks = masks[keep]
    return boxes[keep], scores[keep], labels[keep], masks


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


def draw_detections(img, boxes, scores, labels, color, label_map, masks=None):
    """Draw detections:
    - For tape (class 1): use model mask if available (with low threshold for larger area).
    - Fallback to trapezoid if mask fails.
    - Rectangle for other classes.
    """
    h_img, w_img = img.shape[:2]
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.int().tolist()
        # Clamp coordinates
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))
        
        is_tape = (label.item() == 1)
        polygon_to_draw = None
        
        if is_tape and masks is not None and idx < len(masks):
            # Use model mask
            mask_prob = masks[idx][0].cpu().numpy()
            # Low threshold (0.3) to include more pixels -> "larger"
            mask_binary = (mask_prob > 0.3).astype('uint8') * 255
            
            # Dilate to make it slightly larger/smoother
            kernel = np.ones((5,5), np.uint8)
            mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                # Approx polygon
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) > 8:
                    approx = cv2.convexHull(approx)
                    epsilon = 0.01 * cv2.arcLength(approx, True)
                    approx = cv2.approxPolyDP(approx, epsilon, True)
                
                polygon_to_draw = approx

        if is_tape and polygon_to_draw is None:
            # Fallback trapezoid (no margin = full width)
            width = max(1, x2 - x1)
            margin = 0 # Full width as requested "larger"
            pts = np.array([
                [x1 + margin, y1],
                [x2 - margin, y1],
                [x2, y2],
                [x1, y2]
            ], np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)
            polygon_to_draw = pts

        if is_tape and polygon_to_draw is not None:
            # Draw polygon
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon_to_draw], color)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
            cv2.polylines(img, [polygon_to_draw], True, color, 2)
            
            # Calculate width line intersection
            mid_y = (y1 + y2) // 2
            poly_pts = polygon_to_draw.reshape(-1, 2)
            intersections = []
            for i in range(len(poly_pts)):
                p1 = poly_pts[i]
                p2 = poly_pts[(i + 1) % len(poly_pts)]
                if (p1[1] <= mid_y < p2[1]) or (p2[1] <= mid_y < p1[1]):
                    if p2[1] != p1[1]:
                        x = p1[0] + (mid_y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(int(x))
            
            if len(intersections) >= 2:
                start_x = min(intersections)
                end_x = max(intersections)
                width_px = end_x - start_x
                
                # Draw line inside
                cv2.line(img, (start_x, mid_y), (end_x, mid_y), (255, 255, 0), 1)
                
                # Text above-left
                cv2.putText(img, f"Szer: {width_px}px", (start_x, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                # Fallback if intersection fails (e.g. horizontal edge at mid_y)
                cx, cy, cw, ch = cv2.boundingRect(polygon_to_draw)
                mid_y = cy + ch // 2
                cv2.line(img, (cx, mid_y), (cx + cw, mid_y), (255, 255, 0), 1)
                cv2.putText(img, f"Szer: {cw}px", (cx, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        elif not is_tape:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # label background and text
        label_name = label_map.get(label.item(), str(label.item()))
        text = f"{label_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


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
    seam_model_path: str = "finetuned_models/seam_model.pth",
    seams_per_cycle: int = 1,
    progress_callback: Optional[callable] = None
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
        seams_per_cycle: Liczba szwów na cykl
        progress_callback: Funkcja wywoływana z postępem (current_frame, total_frames)
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
    monitor = BeltMonitor(csv_path=csv_path, seams_per_cycle=seams_per_cycle)
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
    
    # Determine total frames for progress callback
    is_file_source = not (source.startswith("rtsp") or source.startswith("/dev"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file_source else 0

    monitor.is_running = True
    print(f"Rozpoczynam monitorowanie. Źródło: {source}")
    print(f"Całkowita liczba klatek: {total_frames}")
    
    frame_count = 0
    last_alert_text = ""
    alert_display_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                # Dla plików wideo - koniec
                if is_file_source:
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
                tape_boxes, tape_scores, tape_labels, tape_masks = get_prediction(tape_model, img_tensor)
            else:
                # Tryb demo - symuluj detekcję
                tape_boxes = torch.tensor([[100, 100, w-100, h-100]])
                tape_scores = torch.tensor([0.9])
                tape_labels = torch.tensor([1])
                tape_masks = None
            
            # Detekcja szwów
            if seam_model:
                filtered_frame = filter_state.process(frame)
                filtered_tensor = F.to_tensor(filtered_frame).to(device)
                seam_boxes, seam_scores, seam_labels, seam_masks = get_prediction(seam_model, filtered_tensor)
            else:
                seam_boxes = torch.tensor([])
                seam_scores = torch.tensor([])
                seam_labels = torch.tensor([])
                seam_masks = None
            
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
                                       (0, 255, 0), {1: "Tasma"}, tape_masks)
            if seam_model:
                frame = draw_detections(frame, seam_boxes, seam_scores, seam_labels, 
                                       (0, 0, 255), {1: "Szew"}, seam_masks)
            
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
    parser.add_argument(
        "--seams",
        type=int,
        default=1,
        help="Liczba szwów na cykl (domyślnie: 1)"
    )
    
    args = parser.parse_args()
    
    run_monitoring(
        source=args.source,
        csv_path=args.csv,
        output_video=args.output,
        show_preview=not args.no_preview,
        tape_model_path=args.tape_model,
        seam_model_path=args.seam_model,
        seams_per_cycle=args.seams
    )


if __name__ == "__main__":
    main()
