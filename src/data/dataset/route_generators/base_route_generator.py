# oculuz/src/data/dataset/route_generators/base_route_generator.py

import random
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Type, TypeVar, Optional
import numpy as np

from configuration.config_loader import CommonRouteConfig
from src.utils.geometry import haversine_distance, calculate_fspl_rssi, calculate_destination_point

logger = logging.getLogger(__name__)
ConfigType = TypeVar('ConfigType', bound=CommonRouteConfig)

class BaseRouteGenerator(ABC):
    """
    Абстрактный базовый класс для генераторов маршрутов.
    """
    def __init__(self, config: ConfigType): # Убираем config_path
        self.config = config # Всегда используем переданный объект config
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config._to_dict()}")

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
        min_p, max_p = self.config.min_points, self.config.max_points
        min_s, max_s = self.config.min_step_meters, self.config.max_step_meters

        if num_points <= min_p: return max_s
        if num_points >= max_p: return min_s
        if min_p == max_p: return (min_s + max_s) / 2 # Если диапазон точек нулевой

        # Линейная интерполяция
        step = max_s - (num_points - min_p) * (max_s - min_s) / (max_p - min_p)
        return step

    def _generate_random_point_in_area(self) -> Tuple[float, float]:
        """Генерирует случайную точку в пределах рабочей области."""
        # Полярные координаты относительно центра области
        angle = random.uniform(0, 2 * math.pi)
        # Для равномерного распределения по площади, радиус должен быть sqrt(random)
        radius_m = self.config.area_radius_meters * math.sqrt(random.random())

        # Преобразуем смещение в метрах в смещение в градусах (приблизительно)
        # Угол направления не важен, т.к. мы потом используем calculate_destination_point
        # которая корректно работает с градусами и метрами.
        # Здесь мы просто выбираем случайное направление (bearing) и расстояние.
        bearing_deg = math.degrees(angle)

        dest_lat, dest_lon = calculate_destination_point(
            self.config.base_latitude,
            self.config.base_longitude,
            radius_m,
            bearing_deg
        )
        return dest_lat, dest_lon

    def generate_route(self) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Основной метод для генерации полного маршрута, включая RSSI.
        FoV будет рассчитываться позже, в классе Dataset.

        Returns:
            Tuple:
                - path_with_rssi: List[Dict[str, Any]] - список точек маршрута.
                  Каждая точка: {'latitude': ..., 'longitude': ..., 'rssi': ...}
                - source_coords: Dict[str, float] - координаты источника.
        """
        num_points = random.randint(self.config.min_points, self.config.max_points)
        step_m = self._calculate_step_size(num_points)

        # Начальная точка маршрута может быть случайной в пределах рабочей области
        start_lat, start_lon = self._generate_random_point_in_area()

        logger.debug(f"Generating route for {self.__class__.__name__}: "
                     f"{num_points} points, step {step_m:.2f}m, "
                     f"start at ({start_lat:.4f}, {start_lon:.4f})")

        path_coords = self._generate_path_coords(num_points, start_lat, start_lon, step_m)
        if not path_coords: # Если генерация не удалась
             logger.warning(f"Path generation failed for {self.__class__.__name__}. Returning empty route.")
             return [], {"latitude": self.config.base_latitude, "longitude": self.config.base_longitude}


        source_coords = self._place_source(path_coords)

        path_with_rssi: List[Dict[str, Any]] = []
        for point in path_coords:
            distance_to_source = haversine_distance(
                point['longitude'], point['latitude'],
                source_coords['longitude'], source_coords['latitude']
            )
            rssi = calculate_fspl_rssi(distance_to_source, self.config.fspl_k)

            path_with_rssi.append({
                "latitude": point['latitude'],
                "longitude": point['longitude'],
                "rssi": rssi # Это "чистый" RSSI
            })

        logger.info(f"Generated route with {len(path_with_rssi)} points. Source: {source_coords}")
        # logger.debug(f"Generated path_with_rssi (first point): {path_with_rssi[0] if path_with_rssi else 'N/A'}")
        return path_with_rssi, source_coords