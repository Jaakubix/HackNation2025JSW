from __future__ import annotations
import os
import sys
import shutil
import stat
from pathlib import Path

def _rm_readonly(func, path, exc_info):
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
    dir_path.mkdir(parents=True, exist_ok=True)
    removed = 0
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
    Zwraca tryb: 'tak', 'nie' (domyślnie), 'testdata' lub 'datections' (czyści tylko detections_out).
    Priorytet: argument funkcji > CLI --all=... > ENV ALL
    """
    mode = mode_arg

    # CLI
    if mode is None:
        for s in sys.argv[1:]:
            if s.startswith("--all="):
                mode = s.split("=", 1)[1]
                break
            if s in ("--all", "-a"):   # skrót -> traktuj jako 'tak'
                mode = "tak"
                break

    # ENV
    if mode is None:
        mode = os.getenv("ALL")

    if not mode:
        return "nie"

    v = str(mode).strip().lower()
    if v in {"tak", "true", "1", "yes", "y", "t"}:
        return "tak"
    if v in {"nie", "false", "0", "no", "n", "f"}:
        return "nie"
    if v == "testdata":
        return "testdata"
    # nowy tryb: tylko detections_out; wspieramy też alias 'detections'
    if v in {"datections", "detections"}:
        return "datections"
    return "nie"

def main(all: str | None = None):
    root = Path(__file__).resolve().parent

    # Ścieżki z ENV (z sensownymi domyślnymi)
    data_dir       = Path(os.getenv("DATA_ROOT", str(root / "data")))
    detections_dir = Path(os.getenv("DETECTIONS_DIR", str(root / "detections_out")))
    testdata_dir   = Path(os.getenv("TESTDATA_DIR", str(root / "testdata")))

    mode = _read_mode(all)

    # Wybór katalogów do czyszczenia
    if mode == "tak":
        targets = [data_dir, detections_dir, testdata_dir]
    elif mode == "testdata":
        targets = [testdata_dir]
    elif mode == "datections":
        targets = [detections_dir]
    else:  # 'nie' (domyślnie) -> czyść detections_out i testdata
        targets = [detections_dir, testdata_dir]

    print(f"[INFO] Tryb czyszczenia: {mode}")
    for d in targets:
        print(f"  - {d}")

    removed_counts = {}
    for d in targets:
        removed_counts[str(d)] = clean_dir(d)

    print("[DONE] Usunięte elementy:")
    for k, v in removed_counts.items():
        print(f"  {k} = {v}")

if __name__ == "__main__":
    main("nie")
