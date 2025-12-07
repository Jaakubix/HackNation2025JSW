# 🏭 Belt Monitor - System Monitorowania Taśmy Górniczej

System do automatycznego wykrywania i monitorowania taśmy górniczej z wykorzystaniem wizji komputerowej i głębokiego uczenia.

**HackNation 2025 - JSW**

---

## 📋 Funkcjonalności

- ✅ **Monitorowanie 24/7** - ciągła analiza strumienia wideo z kamer
- ✅ **Detekcja taśmy i szwów** - wykorzystanie modeli Faster R-CNN
- ✅ **Analiza szerokości** - pomiar min/max/avg szerokości w każdym cyklu
- ✅ **System alertów** - automatyczne wykrywanie anomalii:
  - Taśma uszkodzona
  - Taśma zerwana
  - Element węższy
  - Uszkodzone łączenie
- ✅ **Raportowanie CSV** - zapis wszystkich cykli do pliku CSV
- ✅ **REST API** - pełne API do integracji z innymi systemami
- ✅ **Klient webowy** - dashboard do podglądu i analizy
- ✅ **Konteneryzacja** - Docker i docker-compose

---

## 🚀 Szybki start

### Wymagania
- Python 3.12+
- CUDA (opcjonalnie, dla GPU)
- Docker (opcjonalnie)

### Instalacja

```bash
# Klonowanie repozytorium
git clone https://github.com/user/HackNation2025JSW.git
cd HackNation2025JSW

# Instalacja zależności
pip install -r requirements.txt
```

### Uruchomienie API

```bash
# Uruchomienie serwera API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Lub z Dockerem
docker-compose up -d
```

Otwórz przeglądarkę: http://localhost:8000

### Uruchomienie monitoringu

```bash
# Z pliku wideo
python monitor_live.py --source nagranie.mp4 --csv output/cycles.csv

# Z kamery
python monitor_live.py --source /dev/video0

# Z RTSP streamu
python monitor_live.py --source rtsp://192.168.1.100:554/stream

# Tryb headless (bez podglądu)
python monitor_live.py --source video.mp4 --no-preview
```

---

## 📡 API Endpoints

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/api/status` | GET | Status systemu |
| `/api/cycles` | GET | Lista cykli (paginacja) |
| `/api/cycles/{id}` | GET | Pojedynczy cykl |
| `/api/alerts` | GET | Lista alertów |
| `/api/videos` | GET | Dostępne nagrania |
| `/api/video/{filename}` | GET | Streaming wideo |
| `/api/csv/download` | GET | Pobierz CSV |
| `/api/stats` | GET | Statystyki zbiorcze |
| `/api/thresholds` | GET/POST | Konfiguracja progów |

**Dokumentacja Swagger:** http://localhost:8000/docs

---

## 🐳 Docker

```bash
# Budowanie obrazu
docker build -t belt-monitor .

# Uruchomienie
docker run -p 8000:8000 -v $(pwd)/output:/app/output belt-monitor

# Lub z docker-compose
docker-compose up -d
```

---

## 📁 Struktura projektu

```
HackNation2025JSW/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI REST API
├── web/
│   └── index.html           # Klient webowy
├── train/                   # Moduły treningowe
├── test/                    # Moduły testowe
├── finetuned_models/        # Wytrenowane modele
├── output/                  # Wyniki (CSV, raporty)
│
├── belt_monitor.py          # Główny moduł monitoringu
├── csv_logger.py            # Logger CSV
├── alert_system.py          # System alertów
├── monitor_live.py          # Skrypt 24/7
├── inference_dual.py        # Inference dwumodelowe
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 📊 Format CSV

```csv
cycle_id,timestamp,segment_count,seam_count,max_width,min_width,avg_width,alerts
1,2025-12-07T03:45:00,10,2,150.5,145.2,147.8,
2,2025-12-07T03:46:30,10,2,150.2,130.1,142.5,[MEDIUM] ELEMENT_NARROWER: Element taśmy jest węższy
```

---

## ⚙️ Konfiguracja progów alertów

Progi można zmienić przez API lub w kodzie:

```python
from alert_system import AlertThresholds

thresholds = AlertThresholds()
thresholds.width_decrease_pct = 5.0      # Alert jeśli szerokość spadła o >5%
thresholds.no_detection_frames = 30      # Alert po 30 klatkach bez detekcji
thresholds.min_absolute_width = 50.0     # Minimalna szerokość (px)
thresholds.max_width_variance_pct = 10.0 # Max wariancja w cyklu
```

---

## 🧪 Testowanie

```bash
# Test na nagraniu
python monitor_live.py --source dual_result.mp4 --csv test_output.csv

# Sprawdź wygenerowany CSV
cat output/cycles.csv

# Test API
curl http://localhost:8000/api/status
curl http://localhost:8000/api/cycles
```

---

## 👥 Autorzy

**Zespół HackNation 2025 - JSW**

---

## 📄 Licencja

Open Source - MIT License
