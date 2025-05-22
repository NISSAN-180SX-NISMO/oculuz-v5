# oculuz/src/metrics/utils.py

import math
import numpy as np  # Используется в других модулях, оставляем для единообразия


def normalize_angle_degrees(angle_deg: float) -> float:
    """
    Нормализует угол к диапазону [0, 360).
    """
    return angle_deg % 360.0


def sin_cos_to_angle_degrees(sin_val: float, cos_val: float) -> float:
    """
    Преобразует пару (sin, cos) в угол в градусах [0, 360).
    atan2(y, x) ожидает sin как y, cos как x.
    """
    angle_rad = math.atan2(sin_val, cos_val)
    angle_deg = math.degrees(angle_rad)
    return normalize_angle_degrees(angle_deg)


def calculate_bearing_degrees(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """
    Вычисляет начальный азимут (пеленг) от точки 1 к точке 2.
    Результат в градусах, в диапазоне [0, 360).
    """
    lat1_rad = math.radians(lat1_deg)
    lon1_rad = math.radians(lon1_deg)
    lat2_rad = math.radians(lat2_deg)
    lon2_rad = math.radians(lon2_deg)

    delta_lon_rad = lon2_rad - lon1_rad

    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

    initial_bearing_rad = math.atan2(y, x)
    initial_bearing_deg = math.degrees(initial_bearing_rad)

    return normalize_angle_degrees(initial_bearing_deg)


def is_angle_in_fov(angle_to_check_deg: float,
                    fov_center_deg: float,
                    fov_width_deg: float) -> bool:
    """
    Проверяет, находится ли заданный угол внутри сектора FoV.
    Все углы в градусах.
    """
    if not (0 <= fov_width_deg <= 360):
        raise ValueError(f"Ширина FoV ({fov_width_deg}) должна быть в диапазоне [0, 360] градусов.")

    if fov_width_deg == 360.0:  # Если ширина 360, то любое направление покрыто
        return True
    if fov_width_deg == 0.0:  # Если ширина 0, то попадание только если угол точно совпадает с центром
        return normalize_angle_degrees(angle_to_check_deg) == normalize_angle_degrees(fov_center_deg)

    # Нормализуем все углы для консистентности
    angle_to_check = normalize_angle_degrees(angle_to_check_deg)
    fov_center = normalize_angle_degrees(fov_center_deg)

    half_width = fov_width_deg / 2.0

    lower_bound = normalize_angle_degrees(fov_center - half_width)
    upper_bound = normalize_angle_degrees(fov_center + half_width)

    if lower_bound <= upper_bound:
        return lower_bound <= angle_to_check <= upper_bound
    else:  # lower_bound > upper_bound (сектор пересекает 0/360)
        return (angle_to_check >= lower_bound) or (angle_to_check <= upper_bound)