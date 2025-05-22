# oculuz/src/data/dataset/route_generators/random_walk_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import ConfigLoader # Используем новый ConfigLoader
from src.utils.geometry import calculate_destination_point

logger = logging.getLogger(__name__)

class RandomWalkRouteGenerator(BaseRouteGenerator):
    # В self.specific_config будут параметры для RandomWalkRoute
    def __init__(self, common_config: ConfigLoader, specific_config: ConfigLoader):
        super().__init__(common_config, specific_config)

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        points: List[Dict[str, float]] = []
        current_lat, current_lon = start_lat, start_lon
        current_bearing_deg = random.uniform(0, 360)

        try:
            walk_params = self.specific_config["parameters"] # Предполагаем, что все параметры в "parameters"
            turn_probability = walk_params["turn_probability"]
            turn_angle_range_deg_min = walk_params["turn_angle_range_deg"][0]
            turn_angle_range_deg_max = walk_params["turn_angle_range_deg"][1]
        except KeyError as e:
            logger.error(f"Missing key in random_walk_route configuration: {e}. Returning empty path.")
            return []

        points.append({"latitude": current_lat, "longitude": current_lon})

        for _ in range(1, num_points):
            if random.random() < turn_probability:
                turn_angle = random.uniform(turn_angle_range_deg_min, turn_angle_range_deg_max)
                current_bearing_deg = (current_bearing_deg + turn_angle + 360) % 360

            next_lat, next_lon = calculate_destination_point(
                current_lat, current_lon, step_m, current_bearing_deg
            )
            points.append({"latitude": next_lat, "longitude": next_lon})
            current_lat, current_lon = next_lat, next_lon
        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        source_lat, source_lon = self._generate_random_point_in_area()
        return {"latitude": source_lat, "longitude": source_lon}