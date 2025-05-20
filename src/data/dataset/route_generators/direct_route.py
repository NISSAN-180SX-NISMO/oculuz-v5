# oculuz/src/data/dataset/route_generators/direct_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import DirectRouteConfig, CommonRouteConfig
from src.utils.geometry import calculate_destination_point

logger = logging.getLogger(__name__)


class DirectRouteGenerator(BaseRouteGenerator):
    # Указываем более конкретный тип конфига для self.config, если нужно для статического анализа
    config: DirectRouteConfig

    def __init__(self, config: DirectRouteConfig):  # Принимает только готовый объект config
        super().__init__(config)
        # Если есть специфичная для генератора логика инициализации, она идет здесь
        # В данном случае, config уже должен быть правильно загружен и настроен снаружи.

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        points: List[Dict[str, float]] = []
        current_lat, current_lon = start_lat, start_lon

        # Случайное направление для прямой
        bearing_deg = random.uniform(0, 360)

        for i in range(num_points):
            if i == 0:
                points.append({"latitude": current_lat, "longitude": current_lon})
            else:
                # Движемся на step_m в выбранном направлении
                next_lat, next_lon = calculate_destination_point(
                    current_lat, current_lon, step_m, bearing_deg
                )
                points.append({"latitude": next_lat, "longitude": next_lon})
                current_lat, current_lon = next_lat, next_lon

        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        # Источник в случайной точке рабочей области
        # (Может быть как на пути, так и в стороне)
        source_lat, source_lon = self._generate_random_point_in_area()
        return {"latitude": source_lat, "longitude": source_lon}