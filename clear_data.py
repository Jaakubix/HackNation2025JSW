from __future__ import annotations
import os
import sys
import shutil
import stat
from pathlib import Path
import config  # Import configuration from this project

def _rm_readonly(func, path, exc_info):
    """Pomocnik do usuwania plików tylko do odczytu (np. z gita)"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def clean_dir(dir_path: Path) -> int:
    """
    Czyści katalog (usuwa wszystkie pliki i podkatalogi), ale nie usuwa samego katalogu.
    Zwraca liczbę usuniętych elementów najwyższego poziomu.
    """
    if not dir_path.exists():
        # Jeśli katalog nie istnieje, tworzymy go (żeby był gotowy na nowe dane)
        dir_path.mkdir(parents=True, exist_ok=True)
        return 0

    removed = 0
    # Iterujemy po zawartości i usuwamy
    for p in dir_path.iterdir():
        try:
            if p.is_dir():
                shutil.rmtree(p, onerror=_rm_readonly)
            else:
                p.unlink(missing_ok=True)
            removed += 1
        except Exception as e:
            print(f"[WARN] Nie udało się usunąć: {p} -> {e}")
    return removed

def _read_mode(mode_arg: str | None) -> str:
    """
    Zwraca tryb działania.
    Priorytet: argument funkcji > CLI --all/--data > ENV
    """
    mode = mode_arg

    # CLI arguments
    if mode is None:
        for s in sys.argv[1:]:
            if s.startswith("--mode="):
                mode = s.split("=", 1)[1]
                break
            if s in ("--all", "-a"):
                mode = "all"
                break
            if s in ("--data", "-d"):
                mode = "data"
                break

    # ENV variable
    if mode is None:
        mode = os.getenv("CLEAN_MODE")

    v = str(mode).strip().lower()
    
    # Mapowanie wartości
    if v in {"all", "wszystko"}:
        return "all"
    if v in {"data", "dane", "dataset"}:
        return "data"
    if v in {"output", "outputs", "wyniki"}:
        return "outputs"
    
    # Domyślnie - jeśli nic nie podano lub 'default'
    return "default"

def main(mode_arg: str | None = None):
    """
    Główna funkcja czyszcząca.
    Dostępne tryby:
    - 'data': czyści TYLKO folder dataset (`data/dataset_combined`). Użyj tego, gdy chcesz wgrać nowe zdjęcia/etykiety.
    - 'outputs': czyści TYLKO foldery wyjściowe (wyniki, checkpointy).
    - 'all': czyści WSZYSTKO (dane + wyniki).
    """
    
    # === DEFINICJA CELÓW ===
    # Katalog datasetu (zdjęcia + adnotacje)
    dataset_dir = config.DATASET_DIR 
    
    # Katalogi wynikowe (jeśli istnieją w projekcie)
    outputs_dir = config.PROJECT_ROOT / "outputs"  
    checkpoints_dir = config.TRAINING["checkpoint_dir"]
    
    # Odczyt trybu
    mode = _read_mode(mode_arg)
    
    targets = []

    if mode == "all":
        # Czyścimy wszystko OPRÓCZ CHECKPOINTÓW (modele są cenne)
        targets = [dataset_dir, outputs_dir]
        
    elif mode == "data":
        # TYLKO dane treningowe (żeby wkleić nowe)
        targets = [dataset_dir]
        
    elif mode == "outputs":
        # Tylko wyniki (żeby oczyścić przed nowym treningiem na TYCH SAMYCH danych)
        targets = [outputs_dir]
        
    else:
        # Domyślnie (np. uruchomione bez argumentów) - zapytajmy lub usuńmy outputy?
        # W Twoim przypadku bezpieczniej domyślnie NIE usuwać danych, chyba że user chce.
        # Ustawmy domyślnie na 'outputs', żeby przypadkiem nie skasować datasetu.
        print("[INFO] Nie podano trybu (użyj --data aby wyczyścić dataset). Domyślnie czyszczę tylko wyniki (bez modeli).")
        targets = [outputs_dir]

    print(f"\n[INFO] Tryb czyszczenia: {mode}")
    print("Będę czyścić następujące lokalizacje:")
    for d in targets:
        print(f"  - {d}")

    # Potwierdzenie przy usuwaniu danych (opcjonalne, ale bezpieczne)
    if mode in ["all", "data"]:
        print("\n[UWAGA] Zamierzasz usunąć DANE TRENINGOWE (zdjęcia i etykiety)!")
        # Możesz to zakomentować jeśli chcesz pełną automatyzację bez pytania:
        # resp = input("Czy na pewno kontynuować? (t/n): ")
        # if resp.lower() not in ('t', 'y', 'tak'):
        #    print("Anulowano.")
        #    return

    print("\nRozpoczynam usuwanie...")
    removed_counts = {}
    for d in targets:
        # Upewniamy się, że ścieżka jest obiektem Path
        d = Path(d)
        count = clean_dir(d)
        removed_counts[str(d)] = count

    print("\n[DONE] Zakończono. Usunięte elementy:")
    for k, v in removed_counts.items():
        print(f"  {k} : {v} plików/folderów")

if __name__ == "__main__":
    # Domyślne wywołanie. Zmień na "data" jeśli chcesz, by kliknięcie F5 czyściło dane.
    # Bezpieczniej zostawić None i sterować argumentami lub zmienić tu ręcznie.
    main("outputs")
