#!/usr/bin/env python3
"""
Skrypt do ekstrakcji klatek z nagrań wideo.
Wspiera wyciąganie klatek co N sekund lub co M-tą klatkę.

Użycie:
    python src/extract_frames.py --input data/videos --output data/frames --interval 1.0
    python src/extract_frames.py --input video.mp4 --output frames/ --every-n-frames 30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
from tqdm import tqdm

# Dodaj katalog główny do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRAME_EXTRACTION, FRAMES_DIR, VIDEOS_DIR


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = None,
    frame_step: int = None,
    output_format: str = "jpg",
    image_quality: int = 95,
    resize: tuple = None,
) -> dict:
    """
    Wyciąga klatki z jednego pliku wideo.
    
    Args:
        video_path: Ścieżka do pliku wideo
        output_dir: Katalog wyjściowy na klatki
        interval_seconds: Wyciągnij klatkę co N sekund
        frame_step: Wyciągnij co N-tą klatkę (alternatywa dla interval_seconds)
        output_format: Format wyjściowy (jpg, png)
        image_quality: Jakość JPEG (1-100)
        resize: Rozmiar wyjściowy (width, height) lub None
        
    Returns:
        Słownik z metadanymi ekstrakcji
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć wideo: {video_path}")
    
    # Pobierz właściwości wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Oblicz krok klatek
    if interval_seconds is not None:
        calculated_step = max(1, int(fps * interval_seconds))
    elif frame_step is not None:
        calculated_step = frame_step
    else:
        calculated_step = int(fps)  # domyślnie co 1 sekundę
    
    # Utwórz podkatalog dla tego wideo
    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadane
    metadata = {
        "video_file": str(video_path),
        "video_name": video_name,
        "fps": fps,
        "total_frames": total_frames,
        "original_size": {"width": width, "height": height},
        "duration_seconds": duration,
        "extraction_settings": {
            "interval_seconds": interval_seconds,
            "frame_step": frame_step,
            "calculated_step": calculated_step,
            "output_format": output_format,
            "resize": resize,
        },
        "extraction_date": datetime.now().isoformat(),
        "frames": []
    }
    
    # Ekstrakcja klatek
    frame_idx = 0
    saved_count = 0
    
    # Parametry zapisu
    if output_format.lower() == "jpg":
        ext = ".jpg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
    else:
        ext = ".png"
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    
    # Progress bar
    expected_frames = total_frames // calculated_step
    pbar = tqdm(total=expected_frames, desc=f"Ekstrakcja: {video_name}", unit="klatek")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % calculated_step == 0:
            # Opcjonalne przeskalowanie
            if resize is not None:
                frame = cv2.resize(frame, resize)
            
            # Nazwa pliku z informacjami o klatce
            timestamp_sec = frame_idx / fps if fps > 0 else 0
            frame_filename = f"{video_name}_frame{frame_idx:06d}_t{timestamp_sec:.2f}s{ext}"
            frame_path = video_output_dir / frame_filename
            
            # Zapisz klatkę
            cv2.imwrite(str(frame_path), frame, encode_params)
            
            # Dodaj do metadanych
            metadata["frames"].append({
                "filename": frame_filename,
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp_sec,
                "path": str(frame_path),
            })
            
            saved_count += 1
            pbar.update(1)
        
        frame_idx += 1
    
    pbar.close()
    cap.release()
    
    metadata["extracted_count"] = saved_count
    
    # Zapisz metadane
    metadata_path = video_output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Wyekstrahowano {saved_count} klatek z {video_name}")
    print(f"  Katalog: {video_output_dir}")
    print(f"  Metadane: {metadata_path}")
    
    return metadata


def process_directory(
    input_dir: Path,
    output_dir: Path,
    **kwargs
) -> list:
    """
    Przetwarza wszystkie wideo w katalogu.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    video_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        print(f"Nie znaleziono plików wideo w: {input_dir}")
        return []
    
    print(f"Znaleziono {len(video_files)} plików wideo")
    
    all_metadata = []
    for video_path in video_files:
        try:
            metadata = extract_frames_from_video(video_path, output_dir, **kwargs)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"✗ Błąd przy przetwarzaniu {video_path}: {e}")
    
    # Zapisz zbiorczy plik metadanych
    summary_path = output_dir / "extraction_summary.json"
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "source_directory": str(input_dir),
        "output_directory": str(output_dir),
        "videos_processed": len(all_metadata),
        "total_frames_extracted": sum(m["extracted_count"] for m in all_metadata),
        "videos": all_metadata
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== PODSUMOWANIE ===")
    print(f"Przetworzone wideo: {summary['videos_processed']}")
    print(f"Łączna liczba klatek: {summary['total_frames_extracted']}")
    print(f"Podsumowanie: {summary_path}")
    
    return all_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrakcja klatek z nagrań wideo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  # Wyciągnij klatki co 1 sekundę z katalogu
  python src/extract_frames.py -i data/videos -o data/frames --interval 1.0
  
  # Wyciągnij co 30-tą klatkę z pojedynczego pliku
  python src/extract_frames.py -i video.mp4 -o frames/ --every-n-frames 30
  
  # Wyciągnij klatki co 0.5s i zmień rozmiar
  python src/extract_frames.py -i video.mp4 -o frames/ --interval 0.5 --resize 1280 720
"""
    )
    
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Plik wideo lub katalog z wideo")
    parser.add_argument("-o", "--output", type=Path, default=FRAMES_DIR,
                        help=f"Katalog wyjściowy (domyślnie: {FRAMES_DIR})")
    parser.add_argument("--interval", type=float, default=FRAME_EXTRACTION["interval_seconds"],
                        help="Wyciągaj klatkę co N sekund")
    parser.add_argument("--every-n-frames", type=int, dest="frame_step",
                        default=FRAME_EXTRACTION["frame_step"],
                        help="Wyciągaj co N-tą klatkę")
    parser.add_argument("--format", choices=["jpg", "png"],
                        default=FRAME_EXTRACTION["output_format"],
                        help="Format wyjściowy obrazów")
    parser.add_argument("--quality", type=int, default=FRAME_EXTRACTION["image_quality"],
                        help="Jakość JPEG (1-100)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        default=FRAME_EXTRACTION["resize"],
                        help="Rozmiar wyjściowy (szerokość wysokość)")
    
    args = parser.parse_args()
    
    # Walidacja
    if not args.input.exists():
        print(f"Błąd: Ścieżka nie istnieje: {args.input}")
        sys.exit(1)
    
    # Przygotuj katalog wyjściowy
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Przygotuj parametry
    kwargs = {
        "output_format": args.format,
        "image_quality": args.quality,
        "resize": tuple(args.resize) if args.resize else None,
    }
    
    # Użyj frame_step jeśli podany, w przeciwnym razie interval
    if args.frame_step is not None:
        kwargs["frame_step"] = args.frame_step
        kwargs["interval_seconds"] = None
    else:
        kwargs["interval_seconds"] = args.interval
        kwargs["frame_step"] = None
    
    # Przetwarzaj
    if args.input.is_file():
        extract_frames_from_video(args.input, args.output, **kwargs)
    else:
        process_directory(args.input, args.output, **kwargs)


if __name__ == "__main__":
    main()
