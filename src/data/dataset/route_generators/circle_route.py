# oculuz/src/data/dataset/route_generators/circle_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import ConfigLoader # Используем новый ConfigLoader
from src.utils.geometry import calculate_destination_point, haversine_distance

logger = logging.getLogger(__name__)

class CircleRouteGenerator(BaseRouteGenerator):
    # В self.specific_config будут параметры для CircleRoute
    def __init__(self, common_config: ConfigLoader, specific_config: ConfigLoader):
        super().__init__(common_config, specific_config)

    def _get_circle_params_from_config(self) -> Tuple[float, float, float, float, float, float, float, Dict[str,float]]:
        """Helper to extract circle parameters to avoid code duplication in _generate_path_coords and _place_source"""
        try:
            # Общие параметры для определения центра круга
            area_def = self.common_config["area_definition"]
            base_lat = area_def["base_latitude"]
            base_lon = area_def["base_longitude"]
            area_radius_m = area_def["area_radius_meters"]

            # Специфичные параметры круга из specific_config
            circle_params = self.specific_config["parameters"] # Предполагаем, что все параметры круга в "parameters"
            center_offset_ratio_max = circle_params["center_offset_ratio_max"]
            radius_range_meters_min = circle_params["radius_range_meters"][0]
            radius_range_meters_max = circle_params["radius_range_meters"][1]
            point_angle_perturbation_deg = circle_params["point_angle_perturbation_deg"]
            point_radius_perturbation_ratio = circle_params["point_radius_perturbation_ratio"]
            source_placement_probabilities = circle_params["source_placement_probabilities"]
            return (base_lat, base_lon, area_radius_m, center_offset_ratio_max,
                    radius_range_meters_min, radius_range_meters_max,
                    point_angle_perturbation_deg, point_radius_perturbation_ratio,
                    source_placement_probabilities)
        except KeyError as e:
            logger.error(f"Missing key in circle route configuration: {e}")
            # Возвращаем какие-то дефолты, чтобы избежать падения, но это плохая ситуация
            return 0.0, 0.0, 1000.0, 0.1, 50.0, 200.0, 5.0, 0.1, {"center":1.0}


    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:

        (base_lat, base_lon, area_radius_m, center_offset_ratio_max,
         radius_min, radius_max, angle_pert_deg, radius_pert_ratio, _) = self._get_circle_params_from_config()

        offset_angle = random.uniform(0, 2 * math.pi)
        offset_radius_m = area_radius_m * center_offset_ratio_max * random.random()

        circle_center_lat, circle_center_lon = calculate_destination_point(
            base_lat, base_lon, offset_radius_m, math.degrees(offset_angle)
        )
        circle_r_m = random.uniform(radius_min, radius_max)
        self._current_circle_center = (circle_center_lat, circle_center_lon) # Сохраняем для _place_source
        self._current_circle_radius = circle_r_m


        points: List[Dict[str, float]] = []
        angle_step_rad = 2 * math.pi / num_points
        start_angle_rad = random.uniform(0, 2 * math.pi)

        for i in range(num_points):
            base_angle = start_angle_rad + i * angle_step_rad
            angle_perturbation = math.radians(random.uniform(-angle_pert_deg, angle_pert_deg))
            radius_perturbation = circle_r_m * random.uniform(-radius_pert_ratio, radius_pert_ratio)

            current_angle_rad = base_angle + angle_perturbation
            current_radius_m = max(1.0, circle_r_m + radius_perturbation)

            point_lat, point_lon = calculate_destination_point(
                circle_center_lat, circle_center_lon,
                current_radius_m, math.degrees(current_angle_rad)
            )
            points.append({"latitude": point_lat, "longitude": point_lon})
        logger.debug(f"Circle route: center ({circle_center_lat:.4f}, {circle_center_lon:.4f}), R={circle_r_m:.2f}m")
        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        try:
            # Используем сохраненные параметры текущей окружности
            circle_center_lat, circle_center_lon = self._current_circle_center
            circle_r_m = self._current_circle_radius
            # Получаем вероятности из конфига снова
            _, _, area_radius_m, _, _, _, _, _, source_placement_probabilities = self._get_circle_params_from_config()
        except AttributeError: # Если _current_circle_center не было установлено (например, _generate_path_coords не вызывался)
            logger.error("Circle parameters not found for source placement. Placing at random area point.")
            return self._generate_random_point_in_area() # Fallback
        except KeyError as e:
            logger.error(f"Missing source_placement_probabilities in circle config: {e}. Placing at circle center.")
            return {"latitude": circle_center_lat, "longitude": circle_center_lon}


        placement_type = random.choices(
            list(source_placement_probabilities.keys()),
            weights=list(source_placement_probabilities.values()),
            k=1
        )[0]

        source_lat, source_lon = circle_center_lat, circle_center_lon

        if placement_type == "center":
            pass
        elif placement_type == "inside":
            r = circle_r_m * math.sqrt(random.random())
            angle = random.uniform(0, 2 * math.pi)
            source_lat, source_lon = calculate_destination_point(
                circle_center_lat, circle_center_lon, r, math.degrees(angle)
            )
        elif placement_type == "outside":
            max_attempts = 10
            for _ in range(max_attempts):
                s_lat, s_lon = self._generate_random_point_in_area() # Генерируем в пределах всей рабочей области
                dist_to_circle_center = haversine_distance(s_lon, s_lat, circle_center_lon, circle_center_lat)
                if dist_to_circle_center > circle_r_m:
                    source_lat, source_lon = s_lat, s_lon
                    break
            else:
                logger.warning("Could not place source outside circle after max_attempts, placing at center.")
                source_lat, source_lon = circle_center_lat, circle_center_lon
        else:
            logger.warning(f"Unknown source placement type: {placement_type}. Placing at center.")

        logger.debug(f"Source placed via '{placement_type}' strategy at ({source_lat:.4f}, {source_lon:.4f})")
        return {"latitude": source_lat, "longitude": source_lon}