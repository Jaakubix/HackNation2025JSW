# 2) Dane i anotacje
- Wyciągnij klatki z nagrań: skrypt `src/extract_frames.py` (argumenty: źródłowy folder wideo, wyjście `data/frames`, co N klatek/T sekund; zapis timestamps).
- Ustal klasy: `element_tasmy`, `laczenie`, `uszkodzenie` (opcjonalnie typy: `uszkodzenie_laczenia`, `zwezenie_elementu`, `przerwanie`).
- Format anotacji: COCO (JSON) – jedyny używany format, kompatybilny z torchvision i projektem baretki.
- Narzędzia anotacji: Label Studio lub CVAT; wyeksportuj do `annotations/train.json`, `annotations/val.json`, `annotations/test.json`.
- Podział zbioru: train/val/test (np. 70/15/15), rozkład warunków (oświetlenie, zabrudzenia) równomierny.
- Augmentacje (do ustalenia w configu): blur, brightness/contrast, gaussian noise, lekkie rotacje, horizontal flip jeśli ma sens; zapisz parametry w `config.py`.
- Dokumentuj kalibrację pix→mm: nagraj klatkę z obiektem referencyjnym lub znaną szerokością taśmy; zapisz wartość do configu.

