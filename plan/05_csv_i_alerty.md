# 5) CSV i alerty
- CSV per cykl: kolumny `cycle_id`, `timestamp_start`, `timestamp_end`, `width_max_mm`, `width_min_mm`, `alerts`.
- Porównanie cykli: skrypt/klasa w `src/cycles.py` pobiera aktualny cykl i poprzedni, wykrywa odchylenia wg progów w configu (procent/bezpośrednie mm).
- Reguły alertów:
  - `zerwanie`: brak detekcji elementów/łączeń w cyklu lub width_min ~0.
  - `zwezenie_elementu`: width_min spada poniżej progu lub różnica względem poprzedniego cyklu > X%.
  - `uszkodzone_laczenie`: detekcja klasy uszkodzenie w okolicy łączeń.
- Zapis alertów: pole `alerts` jako lista/tekst, dodatkowo log do `outputs/logs/alerts.log`.
- Możliwość progów konfigurowalnych (pliki YAML/JSON lub `config.py`); opcja ignorowania niskich score detekcji.

