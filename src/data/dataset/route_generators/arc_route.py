# oculuz/src/data/dataset/route_generators/arc_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import ConfigLoader # Используем новый ConfigLoader
from src.utils.geometry import calculate_destination_point

logger = logging.getLogger(__name__)

class ArcRouteGenerator(BaseRouteGenerator):
    # В self.specific_config будут параметры для ArcRoute
    def __init__(self, common_config: ConfigLoader, specific_config: ConfigLoader):
        super().__init__(common_config, specific_config)

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        points: List[Dict[str, float]] = []

        try:
            arc_params = self.specific_config["parameters"] # Предполагаем, что все параметры дуги в "parameters"
            parabola_coeff_min = arc_params["parabola_coeff_range"][0]
            parabola_coeff_max = arc_params["parabola_coeff_range"][1]
            arc_length_ratio_min = arc_params["arc_length_ratio_range"][0]
            arc_length_ratio_max = arc_params["arc_length_ratio_range"][1]
            # x_max_for_arc_default = arc_params.get("x_max_for_arc_range", [50, 200]) # Опционально
        except KeyError as e:
            logger.error(f"Missing key in arc route configuration: {e}. Returning empty path.")
            return []

        a = random.uniform(parabola_coeff_min, parabola_coeff_max)
        if random.choice([True, False]):
            a = -a
        if a == 0: a = 0.001 # Avoid division by zero or flat line if not intended

        rotation_angle_rad = random.uniform(0, 2 * math.pi)

        # x_max_for_arc = random.uniform(x_max_for_arc_default[0], x_max_for_arc_default[1])
        # Для большей гибкости, x_max_for_arc можно сделать зависимым от step_m и num_points
        # или оставить как в оригинале, если его значения подобраны.
        # Используем оригинальную логику с x_max_for_arc, если она есть
        # В вашем коде он был x_max_for_arc = random.uniform(50, 200)
        x_max_for_arc = random.uniform(
            arc_params.get("x_max_local_range", [50,200])[0], # Если нет, используем дефолт
            arc_params.get("x_max_local_range", [50,200])[1]
        )


        if a < 0:
            x_start_local = -x_max_for_arc * arc_length_ratio_min # Было self.config.arc_length_ratio_range[0]
            x_end_local = x_max_for_arc * random.uniform(arc_length_ratio_min, arc_length_ratio_max)
        else:
            x_start_local = 0
            x_end_local = x_max_for_arc * random.uniform(arc_length_ratio_min, arc_length_ratio_max)


        for i in range(num_points):
            if num_points > 1:
                x_local = x_start_local + (x_end_local - x_start_local) * (i / (num_points - 1))
            else:
                x_local = x_start_local

            y_local = a * x_local ** 2

            x_offset_m = x_local * math.cos(rotation_angle_rad) - y_local * math.sin(rotation_angle_rad)
            y_offset_m = x_local * math.sin(rotation_angle_rad) + y_local * math.cos(rotation_angle_rad)

            current_bearing_rad = math.atan2(x_offset_m, y_offset_m)
            current_bearing_deg = math.degrees(current_bearing_rad)
            current_distance_m = math.sqrt(x_offset_m ** 2 + y_offset_m ** 2)

            point_lat, point_lon = calculate_destination_point(
                start_lat, start_lon, current_distance_m, current_bearing_deg
            )
            points.append({"latitude": point_lat, "longitude": point_lon})
        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        source_lat, source_lon = self._generate_random_point_in_area()
        return {"latitude": source_lat, "longitude": source_lon}