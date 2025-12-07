"""
FastAPI REST API for Belt Monitoring System
Provides endpoints for:
- Video streaming
- CSV data retrieval
- Alert management
- Real-time status
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from csv_logger import CSVLogger
from alert_system import AlertGenerator, AlertThresholds
from monitor_live import run_monitoring

app = FastAPI(
    title="Belt Monitoring API",
    description="API do monitorowania taśmy górniczej - HackNation 2025 JSW",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - pozwól na requesty z frontendu
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfiguracja ścieżek
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
VIDEO_DIR = BASE_DIR
PROCESSED_DIR = BASE_DIR / "processed_videos"

# Ensure processed directory exists
PROCESSED_DIR.mkdir(exist_ok=True)

# Global progress state
# Format: { "filename": percentage (int 0-100) }
processing_progress = {}

# Inicjalizacja komponentów
csv_logger = CSVLogger(str(OUTPUT_DIR / "cycles.csv"))
alert_generator = AlertGenerator()
    
def process_video_task(video_path: str, seams: int):
    """
    Background task to process the uploaded video.
    """
    print(f"Starting background processing for {video_path} with seams={seams}")
    input_path = Path(video_path)
    filename = input_path.name
    
    # Initialize progress
    processing_progress[filename] = 0
    
    def update_progress(current, total):
        if total > 0:
            percent = int((current / total) * 100)
            # Update only if changed to reduce overhead
            if percent != processing_progress.get(filename):
                processing_progress[filename] = percent
    
    try:
        # Generate output filename (e.g., processed_filename.mp4)
        output_filename = f"processed_{input_path.name}"
        output_path = PROCESSED_DIR / output_filename
        temp_output_path = PROCESSED_DIR / f"temp_{output_filename}"
        
        # Generate output CSV filename
        csv_filename = f"processed_{input_path.stem}.csv"
        output_csv_path = PROCESSED_DIR / csv_filename
        
        # Run monitoring (generates mpeg4 video)
        run_monitoring(
            source=video_path,
            csv_path=str(output_csv_path),
            output_video=str(temp_output_path),
            show_preview=False, # No preview for background task
            seams_per_cycle=seams,
            progress_callback=update_progress
        )
        
        # Convert to H.264 for web compatibility
        import subprocess
        print(f"Converting {temp_output_path} to H.264...")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_output_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path)
        ], check=True)
        
        # Remove temp file
        if temp_output_path.exists():
            temp_output_path.unlink()
            
        # Mark as complete
        processing_progress[filename] = 100
            
        print(f"Finished processing {video_path}. Output saved to {output_path}, CSV to {output_csv_path}")
    except Exception as e:
        print(f"Error processing video: {e}")


# ============== MODELE PYDANTIC ==============

class CycleData(BaseModel):
    cycle_id: int
    timestamp: str
    segment_count: int
    seam_count: int
    max_width: float
    min_width: float
    avg_width: float
    alerts: List[str]


class AlertData(BaseModel):
    type: str
    message: str
    severity: str
    timestamp: str
    cycle_id: int
    details: dict


class SystemStatus(BaseModel):
    is_running: bool
    total_cycles: int
    total_alerts: int
    last_update: str
    video_available: bool


class ThresholdConfig(BaseModel):
    width_decrease_pct: float = 5.0
    no_detection_frames: int = 30
    min_absolute_width: float = 50.0
    max_width_variance_pct: float = 10.0
    expected_seams_per_cycle: int = 1


# ============== ENDPOINTS ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Strona główna - przekierowanie do dokumentacji."""
    return """
    <html>
        <head>
            <title>Belt Monitoring API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }
                h1 { color: #00d4ff; }
                a { color: #00d4ff; }
                .endpoints { background: #16213e; padding: 20px; border-radius: 8px; }
                code { background: #0f3460; padding: 2px 6px; border-radius: 4px; }
            </style>
        </head>
        <body>
            <h1>🏭 Belt Monitoring API</h1>
            <p>System monitorowania taśmy górniczej - HackNation 2025 JSW</p>
            <div class="endpoints">
                <h3>Dostępne endpointy:</h3>
                <ul>
                    <li><a href="/docs">/docs</a> - Swagger UI (interaktywna dokumentacja)</li>
                    <li><a href="/redoc">/redoc</a> - ReDoc (alternatywna dokumentacja)</li>
                    <li><code>GET /api/status</code> - Status systemu</li>
                    <li><code>GET /api/cycles</code> - Dane cykli (CSV)</li>
                    <li><code>GET /api/alerts</code> - Lista alertów</li>
                    <li><code>GET /api/video/{filename}</code> - Streaming wideo</li>
                    <li><code>GET /api/csv/download</code> - Pobierz plik CSV</li>
                </ul>
            </div>
            <p style="margin-top: 20px;"><a href="/web">➡️ Przejdź do klienta webowego</a></p>
        </body>
    </html>
    """


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Pobierz status systemu monitoringu."""
    cycles = csv_logger.get_all_cycles()
    
    # Sprawdź dostępne wideo
    video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.mkv"))
    
    return SystemStatus(
        is_running=True,
        total_cycles=len(cycles),
        total_alerts=len(alert_generator.alerts_history),
        last_update=datetime.now().isoformat(),
        video_available=len(video_files) > 0
    )


@app.get("/api/cycles", response_model=List[CycleData])
async def get_cycles(
    limit: int = Query(100, description="Maksymalna liczba cykli do zwrócenia"),
    offset: int = Query(0, description="Offset (paginacja)")
):
    """
    Pobierz dane wszystkich cykli z pliku CSV.
    
    Zwraca listę cykli z informacjami o szerokości taśmy i alertach.
    """
    cycles = csv_logger.get_all_cycles()
    
    # Paginacja
    paginated = cycles[offset:offset + limit]
    
    return [CycleData(**cycle) for cycle in paginated]


@app.get("/api/cycles/{cycle_id}", response_model=CycleData)
async def get_cycle(cycle_id: int):
    """Pobierz pojedynczy cykl po ID."""
    cycles = csv_logger.get_all_cycles()
    
    for cycle in cycles:
        if cycle['cycle_id'] == cycle_id:
            return CycleData(**cycle)
    
    raise HTTPException(status_code=404, detail=f"Cykl o ID {cycle_id} nie znaleziony")


@app.get("/api/alerts", response_model=List[AlertData])
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filtruj po severity: LOW, MEDIUM, HIGH, CRITICAL")
):
    """
    Pobierz wszystkie alerty.
    
    Opcjonalnie filtruj po poziomie ważności.
    """
    alerts = alert_generator.get_all_alerts()
    
    if severity:
        alerts = [a for a in alerts if a['severity'] == severity.upper()]
    
    return [AlertData(**alert) for alert in alerts]


@app.get("/api/csv/download")
async def download_csv():
    """Pobierz plik CSV z danymi cykli."""
    csv_path = OUTPUT_DIR / "cycles.csv"
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Plik CSV nie istnieje")
    
    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename="belt_cycles.csv"
    )


@app.get("/api/progress/{filename}")
async def get_progress(filename: str):
    """Get processing progress for a file."""
    progress = processing_progress.get(filename)
    if progress is None:
        # If file exists in processed dir, assume 100%
        if (PROCESSED_DIR / f"processed_{filename}").exists():
            return {"progress": 100, "status": "completed"}
        return {"progress": 0, "status": "not_found"}
    
    status = "processing" if progress < 100 else "completed"
    return {"progress": progress, "status": status}


@app.get("/api/videos")
async def list_videos():
    """Pobierz listę dostępnych nagrań wideo (tylko przetworzone)."""
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov'}
    videos = []
    
    # Scan processed videos directory
    if PROCESSED_DIR.exists():
        for video_file in PROCESSED_DIR.glob("*"):
            if video_file.suffix in video_extensions and video_file.is_file():
                stat = video_file.stat()
                videos.append({
                    'filename': video_file.name,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    return videos


@app.get("/api/video/{filename}")
async def stream_video(filename: str):
    """
    Streamuj plik wideo.
    
    Obsługuje formaty: mp4, mkv, avi, mov
    """
    # Check processed dir first
    video_path = PROCESSED_DIR / filename
    
    if not video_path.exists():
        # Fallback to root for original files (if needed for debugging)
        video_path = VIDEO_DIR / filename
        
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Wideo {filename} nie znalezione")
    
    # Określ media type
    ext = video_path.suffix.lower()
    media_types = {
        '.mp4': 'video/mp4',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm'
    }
    media_type = media_types.get(ext, 'video/mp4')
    
    # Use FileResponse for proper Range request support (video seeking)
    return FileResponse(
        path=str(video_path),
        media_type=media_type,
        filename=filename
    )


@app.get("/api/thresholds", response_model=ThresholdConfig)
async def get_thresholds():
    """Pobierz aktualne progi alertów."""
    t = alert_generator.thresholds
    return ThresholdConfig(
        width_decrease_pct=t.width_decrease_pct,
        no_detection_frames=t.no_detection_frames,
        min_absolute_width=t.min_absolute_width,
        max_width_variance_pct=t.max_width_variance_pct,
        expected_seams_per_cycle=t.expected_seams_per_cycle
    )


@app.post("/api/thresholds")
async def update_thresholds(config: ThresholdConfig):
    """Zaktualizuj progi alertów."""
    t = alert_generator.thresholds
    t.width_decrease_pct = config.width_decrease_pct
    t.no_detection_frames = config.no_detection_frames
    t.min_absolute_width = config.min_absolute_width
    t.max_width_variance_pct = config.max_width_variance_pct
    t.expected_seams_per_cycle = config.expected_seams_per_cycle
    
    return {"message": "Progi zaktualizowane", "config": config}


@app.get("/api/stats")
async def get_statistics():
    """Pobierz statystyki zbiorcze."""
    cycles = csv_logger.get_all_cycles()
    
    if not cycles:
        return {
            "total_cycles": 0,
            "avg_max_width": 0,
            "avg_min_width": 0,
            "total_alerts": 0,
            "cycles_with_alerts": 0
        }
    
    return {
        "total_cycles": len(cycles),
        "avg_max_width": round(sum(c['max_width'] for c in cycles) / len(cycles), 2),
        "avg_min_width": round(sum(c['min_width'] for c in cycles) / len(cycles), 2),
        "total_alerts": sum(len(c['alerts']) for c in cycles),
        "cycles_with_alerts": sum(1 for c in cycles if c['alerts']),
        "first_cycle_time": cycles[0]['timestamp'] if cycles else None,
        "last_cycle_time": cycles[-1]['timestamp'] if cycles else None
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    seams: int = Form(...)
):
    """Wgraj wideo do analizy."""
    # Save original to root (or separate uploads dir)
    file_path = VIDEO_DIR / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Start background processing
    background_tasks.add_task(process_video_task, str(file_path), seams)
    
    return {
        "filename": file.filename,
        "message": "File uploaded and processing started",
        "seams_configured": seams
    }


@app.get("/api/report/{filename}")
async def get_report(filename: str):
    """
    Get CSV report for a specific processed video.
    Filename should be the processed filename (e.g. processed_video.mp4) or original?
    Let's stick to the logic: user selects processed video from list.
    """
    # Filename comes from the video list, so it is "processed_...mp4"
    input_path = PROCESSED_DIR / filename
    csv_filename = f"{input_path.stem}.csv"
    csv_path = PROCESSED_DIR / csv_filename
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
        
    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename=csv_filename
    )


# Mount static files for web client
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
