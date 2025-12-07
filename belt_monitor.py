"""
Belt Monitor Module
Główny moduł do monitorowania taśmy górniczej - śledzenie cykli, analiza szerokości, integracja alertów.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from csv_logger import CSVLogger
from alert_system import AlertGenerator, Alert


@dataclass
class BeltSegment:
    """Reprezentacja pojedynczego segmentu taśmy."""
    segment_id: int
    width: float
    height: float
    x: int
    y: int
    confidence: float


@dataclass
class CycleData:
    """Dane pojedynczego cyklu taśmy."""
    cycle_id: int
    start_time: str
    end_time: Optional[str] = None
    segments: List[BeltSegment] = field(default_factory=list)
    seam_count: int = 0
    max_width: float = 0.0
    min_width: float = float('inf')
    avg_width: float = 0.0
    frame_count: int = 0
    widths: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'cycle_id': self.cycle_id,
            'segment_count': len(self.segments),
            'seam_count': self.seam_count,
            'max_width': self.max_width if self.max_width > 0 else 0,
            'min_width': self.min_width if self.min_width != float('inf') else 0,
            'avg_width': self.avg_width,
            'start_time': self.start_time,
            'end_time': self.end_time or datetime.now().isoformat()
        }


class BeltWidthAnalyzer:
    """Analizator szerokości taśmy na podstawie bounding boxów."""
    
    def __init__(self, orientation: str = "horizontal"):
        """
        Args:
            orientation: Orientacja taśmy - "horizontal" lub "vertical"
        """
        self.orientation = orientation
    
    def calculate_width(self, boxes: np.ndarray) -> List[float]:
        """
        Oblicza szerokości taśmy z bounding boxów.
        
        Args:
            boxes: Tensor/array bounding boxów [x1, y1, x2, y2]
            
        Returns:
            Lista szerokości dla każdego wykrytego segmentu
        """
        if len(boxes) == 0:
            return []
        
        widths = []
        for box in boxes:
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            x1, y1, x2, y2 = box[:4]
            
            if self.orientation == "horizontal":
                # Dla taśmy poziomej - szerokość to różnica Y (wysokość boxa)
                width = y2 - y1
            else:
                # Dla taśmy pionowej - szerokość to różnica X
                width = x2 - x1
            
            widths.append(float(width))
        
        return widths
    
    def get_statistics(self, widths: List[float]) -> Dict[str, float]:
        """Oblicza statystyki szerokości."""
        if not widths:
            return {'max': 0, 'min': 0, 'avg': 0, 'std': 0}
        
        return {
            'max': max(widths),
            'min': min(widths),
            'avg': sum(widths) / len(widths),
            'std': float(np.std(widths)) if len(widths) > 1 else 0
        }


class BeltCycleTracker:
    """Tracker cykli taśmy - wykrywa początek/koniec cyklu na podstawie szwów."""
    
    def __init__(self, seams_per_cycle: int = 1):
        """
        Args:
            seams_per_cycle: Ile szwów oznacza jeden pełny cykl
        """
        self.seams_per_cycle = seams_per_cycle
        self.current_cycle: Optional[CycleData] = None
        self.completed_cycles: List[CycleData] = []
        self.cycle_counter: int = 0
        self.seam_counter: int = 0
        self.last_seam_frame: int = -100  # Debounce
        self.min_frames_between_seams: int = 10  # Min klatek między detekcjami szwu
    
    def start_new_cycle(self):
        """Rozpoczyna nowy cykl."""
        self.cycle_counter += 1
        self.current_cycle = CycleData(
            cycle_id=self.cycle_counter,
            start_time=datetime.now().isoformat()
        )
        self.seam_counter = 0
    
    def process_frame(
        self,
        frame_id: int,
        belt_boxes: np.ndarray,
        seam_boxes: np.ndarray,
        belt_scores: np.ndarray,
        seam_scores: np.ndarray
    ) -> Optional[CycleData]:
        """
        Przetwarza pojedynczą klatkę.
        
        Args:
            frame_id: Numer klatki
            belt_boxes: Bounding boxy taśmy
            seam_boxes: Bounding boxy szwów
            belt_scores: Confidence scores dla taśmy
            seam_scores: Confidence scores dla szwów
            
        Returns:
            CycleData jeśli cykl się zakończył, None w przeciwnym wypadku
        """
        # Inicjalizacja pierwszego cyklu
        if self.current_cycle is None:
            self.start_new_cycle()
        
        # Analiza szerokości taśmy
        analyzer = BeltWidthAnalyzer()
        widths = analyzer.calculate_width(belt_boxes)
        
        if widths:
            self.current_cycle.widths.extend(widths)
            self.current_cycle.max_width = max(self.current_cycle.max_width, max(widths))
            self.current_cycle.min_width = min(self.current_cycle.min_width, min(widths))
        
        self.current_cycle.frame_count += 1
        
        # Detekcja szwów (z debounce)
        if len(seam_boxes) > 0 and (frame_id - self.last_seam_frame) > self.min_frames_between_seams:
            self.seam_counter += len(seam_boxes)
            self.current_cycle.seam_count += len(seam_boxes)
            self.last_seam_frame = frame_id
            
            # Sprawdź czy cykl się zakończył
            if self.seam_counter >= self.seams_per_cycle:
                return self._complete_cycle()
        
        return None
    
    def _complete_cycle(self) -> CycleData:
        """Kończy bieżący cykl i zwraca jego dane."""
        cycle = self.current_cycle
        cycle.end_time = datetime.now().isoformat()
        
        # Oblicz średnią szerokość
        if cycle.widths:
            cycle.avg_width = sum(cycle.widths) / len(cycle.widths)
        
        # Dodaj segmenty
        for i, width in enumerate(cycle.widths):
            cycle.segments.append(BeltSegment(
                segment_id=i,
                width=width,
                height=0,
                x=0, y=0,
                confidence=1.0
            ))
        
        self.completed_cycles.append(cycle)
        self.start_new_cycle()
        
        return cycle
    
    def force_complete_cycle(self) -> Optional[CycleData]:
        """Wymusza zakończenie cyklu (np. przy końcu wideo)."""
        if self.current_cycle and self.current_cycle.frame_count > 0:
            return self._complete_cycle()
        return None


class BeltMonitor:
    """Główna klasa monitoringu taśmy - integruje wszystkie komponenty."""
    
    def __init__(
        self,
        csv_path: str = "output/cycles.csv",
        seams_per_cycle: int = 1
    ):
        self.csv_logger = CSVLogger(csv_path)
        self.alert_generator = AlertGenerator()
        self.cycle_tracker = BeltCycleTracker(seams_per_cycle)
        self.width_analyzer = BeltWidthAnalyzer()
        
        self.frame_counter: int = 0
        self.is_running: bool = False
        self.last_cycle: Optional[dict] = None
    
    def process_frame(
        self,
        belt_boxes: np.ndarray,
        belt_scores: np.ndarray,
        seam_boxes: np.ndarray,
        seam_scores: np.ndarray
    ) -> Tuple[Optional[dict], List[Alert]]:
        """
        Przetwarza klatkę i zwraca dane cyklu (jeśli zakończony) oraz alerty.
        
        Returns:
            (cycle_data, alerts) - dane cyklu (lub None) i lista alertów
        """
        self.frame_counter += 1
        
        completed_cycle = self.cycle_tracker.process_frame(
            self.frame_counter,
            belt_boxes,
            seam_boxes,
            belt_scores,
            seam_scores
        )
        
        if completed_cycle:
            cycle_dict = completed_cycle.to_dict()
            
            # Generuj alerty
            alerts = self.alert_generator.analyze_cycle(cycle_dict, self.last_cycle)
            
            # Dodaj alerty do danych cyklu
            cycle_dict['alerts'] = self.alert_generator.get_alert_strings(alerts)
            
            # Zapisz do CSV
            self.csv_logger.log_cycle(cycle_dict)
            
            self.last_cycle = cycle_dict
            return cycle_dict, alerts
        
        return None, []
    
    def get_current_status(self) -> dict:
        """Zwraca bieżący status monitoringu."""
        current = self.cycle_tracker.current_cycle
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_counter,
            'completed_cycles': len(self.cycle_tracker.completed_cycles),
            'current_cycle_id': current.cycle_id if current else 0,
            'current_cycle_frames': current.frame_count if current else 0,
            'total_alerts': len(self.alert_generator.alerts_history)
        }
    
    def get_all_cycles(self) -> List[dict]:
        """Zwraca wszystkie zapisane cykle."""
        return self.csv_logger.get_all_cycles()
    
    def get_all_alerts(self) -> List[dict]:
        """Zwraca wszystkie alerty."""
        return self.alert_generator.get_all_alerts()
    
    def finalize(self) -> Optional[dict]:
        """Kończy monitoring - wymusza zakończenie ostatniego cyklu."""
        last_cycle = self.cycle_tracker.force_complete_cycle()
        if last_cycle:
            cycle_dict = last_cycle.to_dict()
            alerts = self.alert_generator.analyze_cycle(cycle_dict, self.last_cycle)
            cycle_dict['alerts'] = self.alert_generator.get_alert_strings(alerts)
            self.csv_logger.log_cycle(cycle_dict)
            return cycle_dict
        return None
