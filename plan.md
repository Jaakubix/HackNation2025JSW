# Plan budowy aplikacji monitorującej taśmę górniczą

## 1) Środowisko
- Utwórz środowisko conda (Python 3.x) z PyTorch/torchvision (Faster R-CNN ResNet50), OpenCV, pandas, tqdm; opcjonalnie CUDA.
- Przygotuj skrypt startowy do trenowania i inferencji (entrypointy CLI).

## 2) Dane i anotacje
- Zebrać klatki z nagrań testowych (różne oświetlenie/brudzenie); wyciągnąć klatki co M sekund/klatek.
- Anotacje: klasy `element_tasmy`, `laczenie`, `uszkodzenie` (lub typy uszkodzeń); format COCO (JSON) pod torchvision.
- Split train/val/test; dodać augmentacje (blur, brightness, noise) pod 24/7.

## 3) Trening modelu
- Fine-tuning Faster R-CNN ResNet50 na anotacjach; dobrać batch size/lr pod GPU/CPU.
- Logi metryk (mAP, precision/recall per klasa); checkpointy modelu.
- Weryfikacja na nagraniach testowych (inicjalny POC).

## 4) Inferencja na nagraniach
- Pipeline: wczytaj wideo → wytnij klatki → detekcja obiektów → filtrowanie po score → śledzenie cyklu taśmy (identyfikacja cykli po położeniu/markerach/czasie).
- Pomiar szerokości: z bboxów (kalibracja pix→mm) lub konturów (jeśli później dodamy segmentację). Wylicz max/min szerokość w cyklu.

## 5) CSV i porównanie cykli
- Dla każdego cyklu zapisz: `cycle_id`, `width_max`, `width_min`, timestamp.
- Porównaj z poprzednim cyklem: wykryj odstępstwa i generuj alerty (zerwanie, węższy element, uszkodzone łączenie).
- Logi/alerty do pliku lub stdout; prosta konfiguracja progów.

## 6) Testy i walidacja
- Testy jednostkowe: parsowanie CSV, logika porównań cykli, kalibracja i pomiar szerokości.
- Testy integracyjne: przepuszczenie krótkiego nagrania testowego i weryfikacja, że CSV i alerty są poprawne.

## 7) Pakowanie i uruchamianie
- Skrypt CLI do inferencji na folderze nagrań lub live streamie.
- Instrukcja uruchomienia (conda env, komendy trening/inferencja).
