# oculuz/src/utils/geometry.py

import math
from typing import Tuple

import numpy as np

# Константы для FSPL
DEFAULT_TX_POWER_DBM = 20  # дБм, мощность передатчика (примерное значение для Wi-Fi)
DEFAULT_FREQUENCY_MHZ = 2400  # МГц, частота (Wi-Fi 2.4 ГГц)
# Коэффициент потерь в свободном пространстве, учитывающий частоту и константы
# PathLoss = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
# RSSI = TxPower - PathLoss (без учета усиления антенн, которые можно включить в TxPower)
# Или проще: RSSI = K - 20 * log10(d_meters)
# где K = TxPower - (20*log10(f_MHz) + 32.44) если d в км
# K = TxPower - (20*log10(f_MHz) - 27.55) если d в метрах
# Для f=2400MHz, 20*log10(2400) = 20 * 3.38 = 67.6
# K_meters = TxPower - (67.6 - 27.55) = TxPower - 40.05
FSPL_CONSTANT_K = DEFAULT_TX_POWER_DBM - 40.05  # Для расстояния в метрах и частоты 2.4 ГГц

EARTH_RADIUS_METERS = 6371000  # Радуис Земли в метрах


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Рассчитывает расстояние в метрах между двумя точками на Земле
    по их широте и долготе.
    """
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_meters = EARTH_RADIUS_METERS * c
    return distance_meters


def calculate_fspl_rssi(distance_meters: float, fspl_k: float = FSPL_CONSTANT_K) -> float:
    """
    Рассчитывает RSSI на основе модели потерь в свободном пространстве (FSPL).
    Без добавления шума.

    Args:
        distance_meters: Расстояние до источника в метрах.
        fspl_k: Константа, зависящая от мощности передатчика и частоты.

    Returns:
        RSSI в дБм.
    """
    if distance_meters <= 0:  # Избегаем log(0) или отрицательных расстояний
        # Очень близко к источнику, RSSI будет максимальным (ограниченным TxPower)
        # или можно вернуть очень высокое значение, которое потом нормализуется.
        # Для простоты вернем значение, соответствующее 1 метру.
        distance_meters = 1.0

        # Path Loss = 20 * log10(d_meters) - (K - TxPower)
    # RSSI = TxPower - (20 * log10(d_meters) - (K - TxPower)) - неверно.
    # RSSI = K - 20 * log10(d_meters)
    rssi = fspl_k - 20 * math.log10(distance_meters)
    return rssi


def calculate_destination_point(lat_deg: float, lon_deg: float, distance_m: float, bearing_deg: float) -> Tuple[
    float, float]:
    """
    Рассчитывает координаты точки назначения, находящейся на заданном расстоянии
    и азимуте от начальной точки.

    Args:
        lat_deg: Широта начальной точки в градусах.
        lon_deg: Долгота начальной точки в градусах.
        distance_m: Расстояние до точки назначения в метрах.
        bearing_deg: Азимут (направление) от начальной точки к точке назначения в градусах
                     (0=Север, 90=Восток, 180=Юг, 270=Запад).

    Returns:
        Tuple (новая_широта_градусы, новая_долгота_градусы).
    """
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    bearing_rad = math.radians(bearing_deg)

    angular_distance = distance_m / EARTH_RADIUS_METERS

    dest_lat_rad = math.asin(math.sin(lat_rad) * math.cos(angular_distance) +
                             math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing_rad))

    dest_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat_rad),
                                        math.cos(angular_distance) - math.sin(lat_rad) * math.sin(dest_lat_rad))

    dest_lat_deg = math.degrees(dest_lat_rad)
    dest_lon_deg = math.degrees(dest_lon_rad)

    return dest_lat_deg, dest_lon_deg