# oculuz/src/data/dataset/route_generators/random_walk_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import RandomWalkRouteConfig
from src.utils.geometry import calculate_destination_point

logger = logging.getLogger(__name__)


class RandomWalkRouteGenerator(BaseRouteGenerator):
    config: RandomWalkRouteConfig

    def __init__(self, config: RandomWalkRouteConfig):  # Принимает только готовый объект config
        super().__init__(config)
        # Если есть специфичная для генератора логика инициализации, она идет здесь
        # В данном случае, config уже должен быть правильно загружен и настроен снаружи.

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        points: List[Dict[str, float]] = []
        current_lat, current_lon = start_lat, start_lon
        current_bearing_deg = random.uniform(0, 360)  # Начальное случайное направление

        points.append({"latitude": current_lat, "longitude": current_lon})

        for _ in range(1, num_points):
            # Решаем, поворачивать ли
            if random.random() < self.config.turn_probability:
                turn_angle = random.uniform(self.config.turn_angle_range_deg[0],
                                            self.config.turn_angle_range_deg[1])
                current_bearing_deg = (current_bearing_deg + turn_angle + 360) % 360  # Нормализуем [0, 360)

            # Делаем шаг
            next_lat, next_lon = calculate_destination_point(
                current_lat, current_lon, step_m, current_bearing_deg
            )

            # Проверка: не вышли ли мы слишком далеко за пределы "разумной" области.
            # Для "городской прогулки" не хотелось бы уходить на 100км.
            # Можно ограничить максимальное расстояние от start_lat, start_lon.
            # Например, не дальше чем area_radius_meters * 1.5 от initial_start_point
            # Если вышли, можно "развернуть" или прекратить. Для простоты пока опустим.

            points.append({"latitude": next_lat, "longitude": next_lon})
            current_lat, current_lon = next_lat, next_lon

        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        # Источник в случайной точке рабочей области
        source_lat, source_lon = self._generate_random_point_in_area()
        return {"latitude": source_lat, "longitude": source_lon}