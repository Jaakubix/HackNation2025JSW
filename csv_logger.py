"""
CSV Logger Module for Belt Monitoring System
Zapisuje dane o cyklach taśmy do pliku CSV.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import threading


class CSVLogger:
    """Logger zapisujący dane cykli taśmy do CSV."""
    
    FIELDNAMES = [
        'cycle_id',
        'timestamp',
        'segment_count',
        'seam_count',
        'max_width',
        'min_width',
        'avg_width',
        'alerts'
    ]
    
    def __init__(self, output_path: str = "output/cycles.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_file()
    
    def _init_file(self):
        """Inicjalizuje plik CSV z nagłówkami jeśli nie istnieje."""
        if not self.output_path.exists():
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
    
    def log_cycle(self, cycle_data: dict):
        """
        Zapisuje dane pojedynczego cyklu.
        
        Args:
            cycle_data: Słownik z danymi cyklu:
                - cycle_id: int
                - segment_count: int
                - seam_count: int
                - max_width: float
                - min_width: float
                - avg_width: float
                - alerts: List[str]
        """
        with self._lock:
            with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                
                row = {
                    'cycle_id': cycle_data.get('cycle_id', 0),
                    'timestamp': datetime.now().isoformat(),
                    'segment_count': cycle_data.get('segment_count', 0),
                    'seam_count': cycle_data.get('seam_count', 0),
                    'max_width': round(cycle_data.get('max_width', 0), 2),
                    'min_width': round(cycle_data.get('min_width', 0), 2),
                    'avg_width': round(cycle_data.get('avg_width', 0), 2),
                    'alerts': ';'.join(cycle_data.get('alerts', []))
                }
                writer.writerow(row)
    
    def get_all_cycles(self) -> List[dict]:
        """Zwraca wszystkie zapisane cykle."""
        cycles = []
        if self.output_path.exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Konwersja typów
                    row['cycle_id'] = int(row['cycle_id'])
                    row['segment_count'] = int(row['segment_count'])
                    row['seam_count'] = int(row['seam_count'])
                    row['max_width'] = float(row['max_width'])
                    row['min_width'] = float(row['min_width'])
                    row['avg_width'] = float(row['avg_width'])
                    row['alerts'] = row['alerts'].split(';') if row['alerts'] else []
                    cycles.append(row)
        return cycles
    
    def get_last_cycle(self) -> Optional[dict]:
        """Zwraca ostatni zapisany cykl."""
        cycles = self.get_all_cycles()
        return cycles[-1] if cycles else None
    
    def get_cycle_count(self) -> int:
        """Zwraca liczbę zapisanych cykli."""
        return len(self.get_all_cycles())
