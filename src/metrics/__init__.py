# oculuz/src/metrics/__init__.py

from .metrics import MetricsCalculator, MetricAggregator
from .base_metrics.hits_persent import calculate_hits_percent_for_session
from .base_metrics.avg_width_deg import calculate_avg_width_deg_for_session
from .utils import ( # Утилиты все еще полезны
    normalize_angle_degrees,
    calculate_bearing_degrees,
    is_angle_in_fov,
    sin_cos_to_angle_degrees # Новая утилита
)

__all__ = [
    "MetricsCalculator",
    "MetricAggregator",
    "calculate_hits_percent_for_session",
    "calculate_avg_width_deg_for_session",
    "normalize_angle_degrees",
    "calculate_bearing_degrees",
    "is_angle_in_fov",
    "sin_cos_to_angle_degrees",
]