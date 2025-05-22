# oculuz/src/data/dataset/route_generators/direct_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import ConfigLoader # Используем новый ConfigLoader
from src.utils.geometry import calculate_destination_point

logger = logging.getLogger(__name__)

class DirectRouteGenerator(BaseRouteGenerator):
    def __init__(self, common_config: ConfigLoader, specific_config: ConfigLoader):
        """
        Args:
            common_config: ConfigLoader для общей конфигурации маршрутов.
            specific_config: ConfigLoader для конфигурации DirectRoute (может быть такой же, как common,
                             если нет специфичных параметров для DirectRoute, или отдельный файл).
        """
        super().__init__(common_config, specific_config)
        # specific_config здесь может использоваться, если DirectRoute имеет свои уникальные параметры
        # Например, self.specific_config["some_direct_route_param"]

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        points: List[Dict[str, float]] = []
        current_lat, current_lon = start_lat, start_lon

        bearing_deg = random.uniform(0, 360) # Параметр из specific_config не предполагается по ТЗ

        for i in range(num_points):
            if i == 0:
                points.append({"latitude": current_lat, "longitude": current_lon})
            else:
                next_lat, next_lon = calculate_destination_point(
                    current_lat, current_lon, step_m, bearing_deg
                )
                points.append({"latitude": next_lat, "longitude": next_lon})
                current_lat, current_lon = next_lat, next_lon
        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        source_lat, source_lon = self._generate_random_point_in_area()
        return {"latitude": source_lat, "longitude": source_lon}