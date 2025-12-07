"""
Alert System Module for Belt Monitoring
Generuje alerty na podstawie analizy taśmy i porównań z poprzednimi cyklami.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class AlertType(Enum):
    """Typy alertów systemowych."""
    BELT_DAMAGED = "BELT_DAMAGED"
    BELT_TORN = "BELT_TORN"
    ELEMENT_NARROWER = "ELEMENT_NARROWER"
    JOINT_DAMAGED = "JOINT_DAMAGED"
    WIDTH_ANOMALY = "WIDTH_ANOMALY"
    NO_DETECTION = "NO_DETECTION"
    SEAM_MISSING = "SEAM_MISSING"


@dataclass
class Alert:
    """Reprezentacja pojedynczego alertu."""
    type: AlertType
    message: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp: str
    cycle_id: int
    details: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'cycle_id': self.cycle_id,
            'details': self.details
        }


class AlertThresholds:
    """Progi alertów - konfigurowalne."""
    
    def __init__(self):
        # Procentowy spadek szerokości do wygenerowania alertu
        self.width_decrease_pct: float = 5.0
        
        # Liczba klatek bez detekcji = alert o zerwanej taśmie
        self.no_detection_frames: int = 30
        
        # Minimalna szerokość absolutna (w pikselach)
        self.min_absolute_width: float = 50.0
        
        # Maksymalna różnica szerokości między segmentami (%)
        self.max_width_variance_pct: float = 10.0
        
        # Oczekiwana liczba szwów na cykl (jeśli mniej = alert)
        self.expected_seams_per_cycle: int = 1


class AlertGenerator:
    """Generator alertów na podstawie danych z monitoringu."""
    
    ALERT_MESSAGES = {
        AlertType.BELT_DAMAGED: "Taśma uszkodzona - wykryto anomalię szerokości",
        AlertType.BELT_TORN: "KRYTYCZNE: Taśma całkowicie zerwana - brak detekcji",
        AlertType.ELEMENT_NARROWER: "Element taśmy jest węższy niż w poprzednim cyklu",
        AlertType.JOINT_DAMAGED: "Uszkodzone łączenie taśmy (szew)",
        AlertType.WIDTH_ANOMALY: "Anomalia szerokości taśmy",
        AlertType.NO_DETECTION: "Brak detekcji taśmy",
        AlertType.SEAM_MISSING: "Brak oczekiwanego szwu w cyklu",
    }
    
    SEVERITY_MAP = {
        AlertType.BELT_DAMAGED: "HIGH",
        AlertType.BELT_TORN: "CRITICAL",
        AlertType.ELEMENT_NARROWER: "MEDIUM",
        AlertType.JOINT_DAMAGED: "HIGH",
        AlertType.WIDTH_ANOMALY: "MEDIUM",
        AlertType.NO_DETECTION: "HIGH",
        AlertType.SEAM_MISSING: "LOW",
    }
    
    def __init__(self, thresholds: Optional[AlertThresholds] = None):
        self.thresholds = thresholds or AlertThresholds()
        self.alerts_history: List[Alert] = []
        self.no_detection_counter: int = 0
    
    def analyze_cycle(
        self,
        current_cycle: dict,
        previous_cycle: Optional[dict] = None
    ) -> List[Alert]:
        """
        Analizuje cykl i generuje alerty.
        
        Args:
            current_cycle: Dane bieżącego cyklu
            previous_cycle: Dane poprzedniego cyklu (do porównań)
            
        Returns:
            Lista wygenerowanych alertów
        """
        alerts = []
        cycle_id = current_cycle.get('cycle_id', 0)
        timestamp = datetime.now().isoformat()
        
        min_width = current_cycle.get('min_width', 0)
        max_width = current_cycle.get('max_width', 0)
        seam_count = current_cycle.get('seam_count', 0)
        
        # 1. Sprawdź brak detekcji (zerwana taśma)
        if min_width == 0 and max_width == 0:
            self.no_detection_counter += 1
            if self.no_detection_counter >= self.thresholds.no_detection_frames:
                alerts.append(Alert(
                    type=AlertType.BELT_TORN,
                    message=self.ALERT_MESSAGES[AlertType.BELT_TORN],
                    severity=self.SEVERITY_MAP[AlertType.BELT_TORN],
                    timestamp=timestamp,
                    cycle_id=cycle_id,
                    details={'frames_without_detection': self.no_detection_counter}
                ))
        else:
            self.no_detection_counter = 0
        
        # 2. Sprawdź minimalną szerokość absolutną
        if 0 < min_width < self.thresholds.min_absolute_width:
            alerts.append(Alert(
                type=AlertType.BELT_DAMAGED,
                message=self.ALERT_MESSAGES[AlertType.BELT_DAMAGED],
                severity=self.SEVERITY_MAP[AlertType.BELT_DAMAGED],
                timestamp=timestamp,
                cycle_id=cycle_id,
                details={'min_width': min_width, 'threshold': self.thresholds.min_absolute_width}
            ))
        
        # 3. Porównanie z poprzednim cyklem
        if previous_cycle:
            prev_min = previous_cycle.get('min_width', 0)
            prev_max = previous_cycle.get('max_width', 0)
            
            # Sprawdź spadek szerokości
            if prev_min > 0 and min_width > 0:
                decrease_pct = ((prev_min - min_width) / prev_min) * 100
                if decrease_pct > self.thresholds.width_decrease_pct:
                    alerts.append(Alert(
                        type=AlertType.ELEMENT_NARROWER,
                        message=self.ALERT_MESSAGES[AlertType.ELEMENT_NARROWER],
                        severity=self.SEVERITY_MAP[AlertType.ELEMENT_NARROWER],
                        timestamp=timestamp,
                        cycle_id=cycle_id,
                        details={
                            'previous_min_width': prev_min,
                            'current_min_width': min_width,
                            'decrease_pct': round(decrease_pct, 2)
                        }
                    ))
        
        # 4. Sprawdź wariancję szerokości w cyklu
        if max_width > 0 and min_width > 0:
            variance_pct = ((max_width - min_width) / max_width) * 100
            if variance_pct > self.thresholds.max_width_variance_pct:
                alerts.append(Alert(
                    type=AlertType.WIDTH_ANOMALY,
                    message=self.ALERT_MESSAGES[AlertType.WIDTH_ANOMALY],
                    severity=self.SEVERITY_MAP[AlertType.WIDTH_ANOMALY],
                    timestamp=timestamp,
                    cycle_id=cycle_id,
                    details={
                        'max_width': max_width,
                        'min_width': min_width,
                        'variance_pct': round(variance_pct, 2)
                    }
                ))
        
        # 5. Sprawdź brak szwów
        if seam_count < self.thresholds.expected_seams_per_cycle:
            alerts.append(Alert(
                type=AlertType.SEAM_MISSING,
                message=self.ALERT_MESSAGES[AlertType.SEAM_MISSING],
                severity=self.SEVERITY_MAP[AlertType.SEAM_MISSING],
                timestamp=timestamp,
                cycle_id=cycle_id,
                details={
                    'expected_seams': self.thresholds.expected_seams_per_cycle,
                    'detected_seams': seam_count
                }
            ))
        
        # Zapisz do historii
        self.alerts_history.extend(alerts)
        
        return alerts
    
    def get_alert_strings(self, alerts: List[Alert]) -> List[str]:
        """Konwertuje alerty na listę stringów do CSV."""
        return [f"[{a.severity}] {a.type.value}: {a.message}" for a in alerts]
    
    def get_all_alerts(self) -> List[dict]:
        """Zwraca wszystkie alerty jako listę słowników."""
        return [a.to_dict() for a in self.alerts_history]
    
    def get_critical_alerts(self) -> List[dict]:
        """Zwraca tylko krytyczne alerty."""
        return [a.to_dict() for a in self.alerts_history if a.severity == "CRITICAL"]
