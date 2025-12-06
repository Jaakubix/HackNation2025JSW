# 1) Środowisko
- Zainstalowana Miniconda w `/home/jakub-pytka/miniconda3`. Upewnij się, że `conda` jest w PATH (np. dodaj `source ~/miniconda3/bin/activate` do `.bashrc`).
- Utwórz env GPU (przykład CUDA 12.1): `conda create -n kopalnia python=3.12 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia`. CPU-only: `conda create -n kopalnia python=3.12 pytorch torchvision cpuonly -c pytorch`.
- Aktywacja: `conda activate kopalnia`. W razie potrzeby aktualizacja pip w env: `pip install --upgrade pip`.
- Doinstaluj biblioteki: `pip install opencv-python pandas tqdm matplotlib fastapi uvicorn[standard] streamlit pytest`.
- Plik `plan/environment.yml` zawiera specyfikację env (py3.12 + PyTorch CUDA 12.1 + pip pakiety). Skrypt `./install_env.sh` tworzy/aktualizuje env `kopalnia` na tej podstawie.
- Jeśli chcesz uniknąć pakietów z `~/.local`, uruchamiaj z `PYTHONNOUSERSITE=1` (opcjonalnie).
- Zweryfikuj GPU: `nvidia-smi`; w Pythonie `import torch; torch.cuda.is_available()`.
- Struktura katalogów: `src/`, `data/raw_videos/`, `data/frames/`, `annotations/`, `models/`, `outputs/csv/`, `outputs/logs/`, `outputs/vis/`, `plan/`.
- Utwórz plik konfiguracyjny (np. `src/config.py`) na ścieżki, progi i parametry; dodaj `.env.example` jeśli będą klucze/webhooki.
