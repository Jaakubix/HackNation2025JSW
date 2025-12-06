# 6) Testy i walidacja
- Unit (pytest): parsowanie CSV, logika progów alertów (cykl N vs N-1), konwersja pix→mm, agregacja max/min szerokości z detekcji.
- Fixtures: przykładowe detekcje i cykle w `tests/fixtures/`. Testy w `tests/test_cycles.py`, `tests/test_calibration.py`, `tests/test_csv.py`.
- Smoke: pojedyncza klatka z znaną szerokością → sprawdź wynik w tolerancji; krótkie wideo testowe → czy powstał CSV i brak błędów runtime.
- Integracja: pipeline `infer_video.py` na przykładowym nagraniu; asercja na obecność kolumn i alertów; opcjonalnie snapshot wizualizacji.
- CI (opcjonalnie): workflow na CPU z pytest (pomija testy GPU dłuższe); marker `@pytest.mark.slow` dla dłuższych testów.

