# 4) Inferencja na nagraniach
- Skrypt `src/infer_video.py`: wejście folder wideo lub pojedyncze wideo; wyjście detekcje, CSV cykli, wizualizacje.
- Przetwarzanie: odczyt klatek (OpenCV) → detekcje Faster R-CNN → filtracja po score (np. >0.5) → NMS (torchvision wbudowany).
- Identyfikacja cykli: startowo okna czasowe T (czas pełnego obrotu). Później ulepszenie: marker wizualny/śledzenie łączenia do detekcji startu cyklu.
- Pomiar szerokości: z bboxów (szerokość px → mm przez kalibrację z configu). Opcja na przyszłość: kontury/segmentacja dla większej dokładności.
- Agregacja per cykl: dla klatek w cyklu zapisz max/min szerokość i timestampy start/end. Przechowaj surowe detekcje (opcjonalnie) w JSON do debugowania.
- Wizualizacje: rysuj bboxy i etykiety z wynikiem szerokości; zapisz do `outputs/vis/<video>/`.

