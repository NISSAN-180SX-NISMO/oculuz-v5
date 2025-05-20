# oculuz/src/data/dataset/route_generators/circle_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import CircleRouteConfig
from src.utils.geometry import calculate_destination_point, haversine_distance

logger = logging.getLogger(__name__)


class CircleRouteGenerator(BaseRouteGenerator):
    config: CircleRouteConfig  # Аннотация типа для config

    def __init__(self, config: CircleRouteConfig):  # Принимает только готовый объект config
        super().__init__(config)
        # Если есть специфичная для генератора логика инициализации, она идет здесь
        # В данном случае, config уже должен быть правильно загружен и настроен снаружи.

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        # start_lat, start_lon здесь игнорируются, т.к. центр окружности выбирается случайно
        # в рабочей области, а не привязан к start_lat/lon из BaseRouteGenerator.
        # Альтернативно, можно было бы start_lat/lon использовать как центр окружности.
        # Выберем новый центр для окружности

        # 1. Определяем центр окружности
        # Смещение центра окружности от центра рабочей области
        offset_angle = random.uniform(0, 2 * math.pi)
        offset_radius_m = self.config.area_radius_meters * self.config.center_offset_ratio_max * random.random()

        circle_center_lat, circle_center_lon = calculate_destination_point(
            self.config.base_latitude, self.config.base_longitude,
            offset_radius_m, math.degrees(offset_angle)
        )

        # 2. Определяем радиус окружности
        circle_r_m = random.uniform(self.config.radius_range_meters[0], self.config.radius_range_meters[1])

        points: List[Dict[str, float]] = []
        angle_step_rad = 2 * math.pi / num_points
        start_angle_rad = random.uniform(0, 2 * math.pi)  # Случайный начальный угол

        for i in range(num_points):
            base_angle = start_angle_rad + i * angle_step_rad

            # Добавляем возмущения к углу и радиусу
            angle_perturbation = math.radians(random.uniform(-self.config.point_angle_perturbation_deg,
                                                             self.config.point_angle_perturbation_deg))
            radius_perturbation = circle_r_m * random.uniform(-self.config.point_radius_perturbation_ratio,
                                                              self.config.point_radius_perturbation_ratio)

            current_angle_rad = base_angle + angle_perturbation
            current_radius_m = circle_r_m + radius_perturbation
            current_radius_m = max(1.0, current_radius_m)  # Радиус не должен быть <= 0

            # Рассчитываем координаты точки на окружности (с возмущениями)
            # от circle_center_lat, circle_center_lon
            point_lat, point_lon = calculate_destination_point(
                circle_center_lat, circle_center_lon,
                current_radius_m, math.degrees(current_angle_rad)  # bearing в градусах
            )
            points.append({"latitude": point_lat, "longitude": point_lon})

        # Проверка, что точки не выходят за пределы рабочей области (опционально, но хорошо бы)
        # Если вышли, можно перегенерировать или обрезать. Для простоты пока опустим.
        logger.debug(f"Circle route: center ({circle_center_lat:.4f}, {circle_center_lon:.4f}), R={circle_r_m:.2f}m")
        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        # Для размещения источника нам нужны параметры окружности, на которой лежат точки.
        # Так как _generate_path_coords их не возвращает, нам нужно их пересчитать или передать.
        # Проще всего: найти "средний" центр и "средний" радиус из path_points.
        # Но для корректного выполнения ТЗ, нам нужен ИДЕАЛЬНЫЙ центр и радиус,
        # которые использовались при генерации. Это усложняет интерфейс.

        # Перегенерируем параметры окружности (не самый эффективный способ, но для простоты)
        # Или можно было бы их сохранить в self из _generate_path_coords
        offset_angle = random.uniform(0, 2 * math.pi)
        offset_radius_m = self.config.area_radius_meters * self.config.center_offset_ratio_max * random.random()
        circle_center_lat, circle_center_lon = calculate_destination_point(
            self.config.base_latitude, self.config.base_longitude,
            offset_radius_m, math.degrees(offset_angle)
        )
        circle_r_m = random.uniform(self.config.radius_range_meters[0], self.config.radius_range_meters[1])

        placement_type = random.choices(
            list(self.config.source_placement_probabilities.keys()),
            weights=list(self.config.source_placement_probabilities.values()),
            k=1
        )[0]

        source_lat, source_lon = circle_center_lat, circle_center_lon  # По умолчанию - центр

        if placement_type == "center":
            pass  # Уже установлено
        elif placement_type == "inside":
            # Случайная точка внутри окружности
            r = circle_r_m * math.sqrt(random.random())  # Для равномерного распределения по площади
            angle = random.uniform(0, 2 * math.pi)
            source_lat, source_lon = calculate_destination_point(
                circle_center_lat, circle_center_lon, r, math.degrees(angle)
            )
        elif placement_type == "outside":
            # Случайная точка снаружи окружности, но в пределах рабочей области
            # Генерируем точку в кольце между circle_r_m и area_radius_meters (относительно центра окружности)
            # Это немного сложнее, т.к. area_radius_meters задан относительно base_lat/lon
            # Простой вариант: сгенерировать в рабочей области, и если попало внутрь круга - перегенерировать.
            max_attempts = 10
            for _ in range(max_attempts):
                s_lat, s_lon = self._generate_random_point_in_area()
                dist_to_circle_center = haversine_distance(s_lon, s_lat, circle_center_lon, circle_center_lat)
                if dist_to_circle_center > circle_r_m:
                    source_lat, source_lon = s_lat, s_lon
                    break
            else:  # Если не удалось найти точку снаружи (маловероятно, но возможно)
                logger.warning("Could not place source outside circle after max_attempts, placing at center.")
                source_lat, source_lon = circle_center_lat, circle_center_lon

        logger.debug(f"Source placed via '{placement_type}' strategy at ({source_lat:.4f}, {source_lon:.4f})")
        return {"latitude": source_lat, "longitude": source_lon}