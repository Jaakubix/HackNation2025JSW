# Plan szczegółowy budowy aplikacji monitorującej taśmę górniczą

- ## 1) Środowisko
- Miniconda zainstalowana w `~/miniconda3`. Upewnij się, że `conda` jest w PATH (`source ~/miniconda3/bin/activate` w `.bashrc`).
- GPU env (CUDA 12.1 przykład): `conda create -n kopalnia python=3.12 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia`. CPU-only: `conda create -n kopalnia python=3.12 pytorch torchvision cpuonly -c pytorch`.
- Aktywacja: `conda activate kopalnia`; ew. `pip install --upgrade pip`.
- Pakiety: `pip install opencv-python pandas tqdm matplotlib fastapi uvicorn[standard] streamlit pytest`.
- Zweryfikuj GPU: `nvidia-smi` i `import torch; torch.cuda.is_available()`.
- Alternatywa: `plan/environment.yml` + `./install_env.sh` (py3.12 + PyTorch CUDA 12.1 + pip pakiety).
- Przygotuj strukturę repo: `src/` na kod, `data/` na surowe nagrania, `annotations/` na etykiety, `models/` na checkpointy, `outputs/` na CSV i logi.

## 2) Dane i anotacje
- Wyodrębnij klatki z nagrań testowych: skrypt `extract_frames.py` (co N klatek lub co T sekund) z zachowaniem metadanych timestamp/cycle.
- Ustal klasy: `element_tasmy`, `laczenie`, `uszkodzenie` (opcjonalnie typy: `uszkodzenie_laczenia`, `zwezenie_elementu`, `przerwanie`).
- Format anotacji: COCO (JSON) – jedyny używany format, kompatybilny z torchvision i projektem baretki (obsługuje segmentację, bboxy).
- Narzędzie do anotacji: Label Studio / CVAT. Konwencja nazw plików zgodna z klatkami.
- Podział na zbiory: train/val/test (np. 70/15/15) z uwzględnieniem różnych warunków (oświetlenie/brud/cień). Zapisać listy plików.
- Augmentacje: blur, brightness/contrast, gaussian noise, horizontal flip (tylko jeśli symetria ma sens), lekkie rotacje; parametry w configu.

## 3) Trening modelu
- Loader danych: dataset COCO z mapowaniem klas; collate_fn dla detection.
- Hiperparametry startowe: lr 1e-3, batch_size zależny od GPU (np. 2–4), scheduler (StepLR), epochs 20–50 w zależności od konwergencji.
- Weights: `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")`; zamrozić backbone na kilka epok lub trenować całość w drugiej fazie.
- Logowanie: mAP/mAR per klasa, loss curves; zapisywać checkpoint co kilka epok i najlepszy po val mAP.
- Walidacja w trakcie: inference na kilku nagraniach/klatkach referencyjnych, zapisywać wizualizacje bboxów do `outputs/vis/`.

## 4) Inferencja na nagraniach
- Skrypt `infer_video.py`: wideo → klatki → detekcje (score threshold, NMS) → przypisanie do cykli.
- Detekcje: filtrowanie po score (np. >0.5), kategoria (element/łączenie/uszkodzenie). Kalibracja pix→mm (próba: znany obiekt referencyjny lub znana szerokość taśmy).
- Identyfikacja cyklu: na podstawie czasu obrotu taśmy lub markerów wizualnych; na początek przyjąć stały czas cyklu T i dzielić strumień na okna T.
- Pomiar szerokości: z bboxów (szerokość w px → mm po kalibracji). Jeśli w przyszłości segmentacja: szerokość z konturów dla większej dokładności.
- Agregacja w cyklu: dla każdej klatki brać max/min szerokość; na koniec cyklu zapisać `width_max`, `width_min`, `cycle_id`, timestamp.

## 5) CSV i porównanie cykli
- Struktura CSV: kolumny `cycle_id`, `timestamp_start`, `timestamp_end`, `width_max`, `width_min`, ew. `alerts`.
- Porównanie: dla cyklu N i N-1 sprawdzać progi odchyleń (np. różnica szerokości > X mm, brak detekcji elementów = zerwanie).
- Reguły alertów: 
  - `zerwanie`: brak elementów/łączeń przez cały cykl lub drastyczny spadek liczby detekcji.
  - `zwezenie_elementu`: `width_min` spada poniżej progu lub różni się od poprzedniego cyklu o >X%.
  - `uszkodzone_laczenie`: detekcje klasy uszkodzenie w rejonie łączeń.
- Logowanie alertów: zapis do CSV/JSON oraz stdout; opcjonalnie webhook/email w dalszej fazie.

## 6) Testy i walidacja
- Unit: parsowanie CSV, logika progów alertów, kalibracja piksel→mm, agregacja max/min szerokości z detekcji.
- Integracja: przepuszczenie krótkiego nagrania testowego przez `infer_video.py`, sprawdzenie, że CSV powstaje i alerty są zgodne z oczekiwaniem.
- Smoke: test na pojedynczej klatce z ręcznie znaną szerokością (spodziewany wynik w granicach tolerancji).
- CI (opcjonalnie): prosty workflow z unit testami (CPU-only).

## 7) Pakowanie i uruchamianie
- Skrypty CLI:
  - `train.py`: ścieżki do danych/anotacji, hparamy, output checkpoint.
  - `infer_video.py`: ścieżka do wideo/folderu, model checkpoint, output CSV + wizualizacje.
- Dokumentacja uruchomienia: komendy conda create + pip install w env + train + infer na folderze nagrań. Plik README z przykładami.
- Struktura outputów: `outputs/csv/`, `outputs/logs/`, `outputs/vis/`, `models/`.

## 8) Rozszerzenia (opcjonalne)
- Segmentacja (Mask R-CNN lub YOLO-seg) dla precyzyjnego pomiaru szerokości.
- Śledzenie obiektów między klatkami (SORT/DeepSORT) dla stabilniejszego cyklu.
- Panel podglądu (np. prosty Streamlit) dla wizualizacji alertów i statystyk.
