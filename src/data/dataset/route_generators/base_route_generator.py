# oculuz/src/data/dataset/route_generators/base_route_generator.py

import random
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from configuration.config_loader import ConfigLoader # Используем новый ConfigLoader
# Предполагается, что утилиты geometry находятся здесь
from src.utils.geometry import haversine_distance, calculate_fspl_rssi, calculate_destination_point

logger = logging.getLogger(__name__)

class BaseRouteGenerator(ABC):
    """
    Абстрактный базовый класс для генераторов маршрутов.
    """
    def __init__(self, common_config: ConfigLoader, specific_config: ConfigLoader):
        """
        Инициализирует базовый генератор маршрутов.

        Args:
            common_config: ConfigLoader для общей конфигурации маршрутов.
            specific_config: ConfigLoader для специфичной конфигурации данного типа маршрута.
        """
        self.common_config: ConfigLoader = common_config
        self.specific_config: ConfigLoader = specific_config # Может использоваться в подклассах

        # Логирование: используем .data, если есть, или сам объект
        common_data_log = self.common_config.data if hasattr(self.common_config, 'data') else self.common_config
        specific_data_log = self.specific_config.data if hasattr(self.specific_config, 'data') else self.specific_config
        logger.info(f"Initialized {self.__class__.__name__} with common_config: {common_data_log}, specific_config: {specific_data_log}")


    @abstractmethod
    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[Dict[str, float]]:
        """
        Генерирует координаты точек маршрута. Должен быть реализован в подклассах.
        Возвращает список словарей [{'latitude': ..., 'longitude': ...}, ...].
        """
        pass

    @abstractmethod
    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Размещает источник сигнала. Должен быть реализован в подклассах.
        Возвращает словарь {'latitude': ..., 'longitude': ...}.
        """
        pass

    def _calculate_step_size(self, num_points: int) -> float:
        """
        Рассчитывает размер шага. Шаг тем больше, чем меньше точек.
        Линейная интерполяция: если num_points = min_points -> max_step, если num_points = max_points -> min_step
        """
        try:
            generation_rules = self.common_config["generation_rules"]
            min_p = generation_rules["num_points_range"][0]
            max_p = generation_rules["num_points_range"][1]
            min_s = generation_rules["step_meters_range"][0]
            max_s = generation_rules["step_meters_range"][1]
        except KeyError as e:
            logger.error(f"Missing key in common_config for step size calculation: {e}. Using default step 50m.")
            return 50.0


        if num_points <= min_p: return max_s
        if num_points >= max_p: return min_s
        if min_p == max_p: return (min_s + max_s) / 2

        step = max_s - (num_points - min_p) * (max_s - min_s) / (max_p - min_p)
        return step

    def _generate_random_point_in_area(self) -> Tuple[float, float]:
        """Генерирует случайную точку в пределах рабочей области."""
        try:
            area_config = self.common_config["area_definition"]
            base_lat = area_config["base_latitude"]
            base_lon = area_config["base_longitude"]
            area_radius_m = area_config["area_radius_meters"]
        except KeyError as e:
            logger.error(f"Missing key in common_config for random point generation: {e}. Using (0,0).")
            return 0.0, 0.0

        angle = random.uniform(0, 2 * math.pi)
        radius_m = area_radius_m * math.sqrt(random.random())
        bearing_deg = math.degrees(angle)

        dest_lat, dest_lon = calculate_destination_point(
            base_lat, base_lon, radius_m, bearing_deg
        )
        return dest_lat, dest_lon

    def generate_route(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Основной метод для генерации полного маршрута, включая RSSI.
        """
        try:
            generation_rules = self.common_config["generation_rules"]
            area_definition = self.common_config["area_definition"]
            signal_properties = self.common_config["signal_properties"]

            min_points = generation_rules["num_points_range"][0]
            max_points = generation_rules["num_points_range"][1]
            num_points = random.randint(min_points, max_points)
            fspl_k = signal_properties["fspl_k"]
            base_lat_for_empty = area_definition.get("base_latitude", 0.0) # Для случая пустого маршрута
            base_lon_for_empty = area_definition.get("base_longitude", 0.0)

        except KeyError as e:
            logger.error(f"Missing critical key in common_config for route generation: {e}. Returning empty route.")
            return [], {"latitude": 0.0, "longitude": 0.0}


        step_m = self._calculate_step_size(num_points)
        start_lat, start_lon = self._generate_random_point_in_area()

        logger.debug(f"Generating route for {self.__class__.__name__}: "
                     f"{num_points} points, step {step_m:.2f}m, "
                     f"start at ({start_lat:.4f}, {start_lon:.4f})")

        path_coords = self._generate_path_coords(num_points, start_lat, start_lon, step_m)
        if not path_coords:
             logger.warning(f"Path generation failed for {self.__class__.__name__}. Returning empty route.")
             return [], {"latitude": base_lat_for_empty, "longitude": base_lon_for_empty}

        source_coords = self._place_source(path_coords)

        path_with_rssi: List[Dict[str, Any]] = []
        for point in path_coords:
            distance_to_source = haversine_distance(
                point['longitude'], point['latitude'],
                source_coords['longitude'], source_coords['latitude']
            )
            rssi = calculate_fspl_rssi(distance_to_source, fspl_k)

            path_with_rssi.append({
                "latitude": point['latitude'],
                "longitude": point['longitude'],
                "rssi": rssi
            })

        logger.info(f"Generated route with {len(path_with_rssi)} points. Source: {source_coords}")
        return path_with_rssi, source_coords