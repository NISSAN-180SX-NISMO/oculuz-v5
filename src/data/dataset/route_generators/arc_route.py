# oculuz/src/data/dataset/route_generators/arc_route.py
import random
import math
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .base_route_generator import BaseRouteGenerator
from configuration.config_loader import ArcRouteConfig
from src.utils.geometry import calculate_destination_point, EARTH_RADIUS_METERS

logger = logging.getLogger(__name__)


class ArcRouteGenerator(BaseRouteGenerator):
    config: ArcRouteConfig

    def __init__(self, config: ArcRouteConfig):  # Принимает только готовый объект config
        super().__init__(config)
        # Если есть специфичная для генератора логика инициализации, она идет здесь
        # В данном случае, config уже должен быть правильно загружен и настроен снаружи.

    def _generate_path_coords(self, num_points: int, start_lat: float, start_lon: float, step_m: float) -> List[
        Dict[str, float]]:
        points: List[Dict[str, float]] = []

        # Параметры параболы
        # y = a * x^2
        # Знак 'a' определяет направление "ветвей" параболы
        a = random.uniform(self.config.parabola_coeff_range[0], self.config.parabola_coeff_range[1])
        if random.choice([True, False]):  # Случайно инвертируем знак a
            a = -a

        # Случайная ориентация параболы (поворот системы координат)
        rotation_angle_rad = random.uniform(0, 2 * math.pi)

        # Максимальный x, чтобы парабола не выходила далеко за пределы рабочей области.
        # y_max ~ area_radius_meters => a * x_max^2 ~ area_radius_meters => x_max ~ sqrt(area_radius_meters / |a|)
        # Это очень грубая оценка, т.к. парабола может быть смещена.
        # Лучше ограничить длину дуги.

        # Длина дуги параболы y=ax^2 от 0 до x_val: integral(sqrt(1 + (2ax)^2)) dx from 0 to x_val
        # Это сложный интеграл. Используем параметризацию по x.
        # Общая "длина" параболы в единицах x, которую мы можем пройти.
        # Пусть шаг по x будет связан с реальным шагом step_m.
        # Начнем с x = 0.

        current_x_local = 0.0
        # Определим максимальный x так, чтобы y не превышал area_radius_meters
        # Это грубо, так как парабола может быть повернута и смещена.
        # Будем генерировать точки, пока они внутри area_radius_meters от start_lat, start_lon
        # или пока не достигнем num_points.

        # Общая длина маршрута ~ num_points * step_m
        # Пройдемся по точкам, увеличивая x_local.
        # step_x_local должен быть таким, чтобы реальный шаг был ~ step_m
        # Длина дуги ds = sqrt(dx^2 + dy^2) = sqrt(1 + (dy/dx)^2) dx
        # dy/dx = 2ax.  ds = sqrt(1 + 4a^2x^2) dx.
        # Мы хотим, чтобы ds ~ step_m.  dx ~ step_m / sqrt(1 + 4a^2x^2)

        # Проще: генерируем точки в локальной системе координат параболы (x, ax^2)
        # затем поворачиваем и сдвигаем в start_lat, start_lon.
        # Максимальный x, чтобы y не вышел за пределы area_radius_meters
        # y_max_local = self.config.area_radius_meters -> x_max_local = sqrt(y_max_local / |a|)
        # Это очень грубо, т.к. мы работаем в метрах, а не в градусах.

        # Будем генерировать N точек, равномерно распределенных по некоторому диапазону x_local.
        # Определим максимальный x_local так, чтобы вся дуга имела длину примерно num_points * step_m
        # Это сложно. Проще: выбрать x_local_max и распределить точки.
        # Пусть x_local изменяется от 0 до x_max_for_arc.
        # x_max_for_arc подберем так, чтобы парабола была "разумной" длины.
        # Например, чтобы y при x_max_for_arc был порядка нескольких сотен метров.
        # y = 100m, a = 0.005 => x^2 = 100/0.005 = 20000 => x ~ 140m.
        x_max_for_arc = random.uniform(50,
                                       200)  # Максимальное значение x в локальной системе координат параболы (в метрах)
        if a < 0:  # Если парабола ветвями вниз, симметрично отразим ее для генерации
            x_start_local = -x_max_for_arc * self.config.arc_length_ratio_range[0]
            x_end_local = x_max_for_arc * random.uniform(self.config.arc_length_ratio_range[0],
                                                         self.config.arc_length_ratio_range[1])
        else:
            x_start_local = 0  # Начнем с вершины
            x_end_local = x_max_for_arc * random.uniform(self.config.arc_length_ratio_range[0],
                                                         self.config.arc_length_ratio_range[1])

        for i in range(num_points):
            # Распределяем точки по x_local
            if num_points > 1:
                x_local = x_start_local + (x_end_local - x_start_local) * (i / (num_points - 1))
            else:
                x_local = x_start_local  # или (x_start_local + x_end_local) / 2

            y_local = a * x_local ** 2

            # Поворот локальных координат (x_local, y_local) -> (x_rotated, y_rotated)
            # Это смещения в метрах от начальной точки маршрута
            x_offset_m = x_local * math.cos(rotation_angle_rad) - y_local * math.sin(rotation_angle_rad)
            y_offset_m = x_local * math.sin(rotation_angle_rad) + y_local * math.cos(rotation_angle_rad)

            # Рассчитываем Bearing (азимут) и расстояние от start_lat, start_lon до (x_offset_m, y_offset_m)
            # Bearing: atan2(x_offset_m, y_offset_m) т.к. y_offset_m - это "северное" смещение, x_offset_m - "восточное"
            # В функции calculate_destination_point bearing 0=Север, 90=Восток
            # Наш y_offset_m соответствует смещению по Y (Север), x_offset_m по X (Восток)
            current_bearing_rad = math.atan2(x_offset_m, y_offset_m)  # atan2(east, north)
            current_bearing_deg = math.degrees(current_bearing_rad)
            current_distance_m = math.sqrt(x_offset_m ** 2 + y_offset_m ** 2)

            point_lat, point_lon = calculate_destination_point(
                start_lat, start_lon, current_distance_m, current_bearing_deg
            )
            points.append({"latitude": point_lat, "longitude": point_lon})

        return points

    def _place_source(self, path_points: List[Dict[str, float]]) -> Dict[str, float]:
        # Источник в случайной точке рабочей области
        source_lat, source_lon = self._generate_random_point_in_area()
        return {"latitude": source_lat, "longitude": source_lon}    