# Instrukcje – monitoring taśmy górniczej (JSW IT)

- **Cel**: program ma 24/7 monitorować z kamer cykle taśmy górniczej złożonej z N elementów połączonych N łączeniami.
- **Rejestracja cyklu**: w każdym cyklu zapisz w pliku CSV jego `id`, największą szerokość i najmniejszą szerokość.
- **Porównania**: w kolejnym cyklu porównaj wyniki z poprzednim i zgłaszaj wykryte odstępstwa.
- **Przykładowe alerty**: taśma uszkodzona, taśma całkowicie zerwana, element taśmy jest węższy, uszkodzone łączenie taśmy.

## Środowisko i narzędzia
- Anaconda (conda) do zarządzania środowiskiem.
- Python 3.x (w środowisku conda).
- PyTorch + torchvision; model bazowy Faster R-CNN ResNet50 fine-tuning na naszych danych.
- OpenCV do przetwarzania wideo/klatek, pandas do CSV, ew. tqdm do logów postępu.
- GPU z CUDA (opcjonalnie) dla szybszej inferencji i treningu; na CPU będzie wolniej.

## Jak dodać nowe dane do treningu?

Aby douczyć model na nowych zdjęciach, wykonaj następujące kroki:

### 1. Przygotuj zdjęcia
1.  Wrzuć wszystkie zdjęcia (zarówno stare jak i nowe) do jednego folderu:
    `data/dataset_combined/images/`
    *(Możesz po prostu dokopiować nowe pliki do tych, które już tam są)*.

### 2. Przygotuj anotacje (COCO)
1.  Musisz mieć plik JSON w formacie COCO, który opisuje te zdjęcia.
2.  Jeśli masz nowy plik JSON (np. z Label Studio), nazwij go np. `nowe_anotacje.json` i wrzuć do `data/dataset_combined/`.
3.  Jeśli chcesz połączyć stare anotacje z nowymi, możesz użyć skryptu:
    ```bash
    python3 uczenie/merge_annotations.py
    ```
    *(Wymaga edycji ścieżek wewnątrz skryptu `uczenie/merge_annotations.py`)*.

### 3. Uruchom trening
Użyj komendy wskazującej na Twój plik z anotacjami i folder ze zdjęciami:

```bash
python3 uczenie/train.py \
  --annotations data/dataset_combined/nowe_anotacje.json \
  --images data/dataset_combined/images \
  --epochs 30
```

Skrypt automatycznie podzieli dane na zbiór treningowy i walidacyjny.
