# oculuz/sigma_fov.py
import math
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# Параметры для заглушек, чтобы они возвращали хоть что-то осмысленное
DEFAULT_FOV_WIDTH_DEG = 30.0  # Примерная ширина FoV


def calculate_angle_between_points(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Рассчитывает угол (азимут) от точки 1 к точке 2.
    Возвращает угол в градусах (0-360), где 0 - Север, 90 - Восток.
    """
    # Приблизительный расчет для небольших расстояний, не учитывающий кривизну Земли идеально,
    # но достаточный для симуляции.
    # Для точных расчетов лучше использовать более сложные формулы (например, vincenty).
    delta_lon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)

    # Нормализация угла к диапазону 0-360
    bearing = (angle_deg + 360) % 360
    return bearing


def calculate_fov_for_point(
        measurement_points: List[Dict[str, Any]],
        # Список точек [{ "latitude": ..., "longitude": ..., "rssi": ...}, ...]
        source_coords: Dict[str, float]  # {"latitude": ..., "longitude": ...}
) -> Dict[str, float]:
    """
    Рассчитывает FoV для ПОСЛЕДНЕЙ точки в measurement_points относительно источника.
    Это ЗАГЛУШКА. Реальная логика будет сложнее.

    Args:
        measurement_points: Список всех точек измерений до текущей (включая текущую).
                           Каждая точка - словарь с 'latitude', 'longitude', 'rssi'.
        source_coords: Координаты источника сигнала {'latitude': ..., 'longitude': ...}.

    Returns:
        Словарь с параметрами FoV:
        {
            "fov_dir_deg": float,  # Направление центра FoV в градусах (0-360)
            "fov_width_deg": float # Ширина FoV в градусах
        }
    """
    if not measurement_points:
        logger.error("Получен пустой список точек для расчета FoV.")
        # Возвращаем "безопасные" значения по умолчанию или возбуждаем ошибку
        return {"fov_dir_deg": 0.0, "fov_width_deg": DEFAULT_FOV_WIDTH_DEG}

    current_point = measurement_points[-1]  # FoV считается для последней точки

    # Заглушка: FoV направлен точно на источник
    fov_direction_deg = calculate_angle_between_points(
        current_point['longitude'], current_point['latitude'],
        source_coords['longitude'], source_coords['latitude']
    )

    # Заглушка: ширина FoV зависит от RSSI (чем слабее сигнал, тем шире FoV)
    # Это очень упрощенная логика. Реальная будет зависеть от алгоритма SIGMA
    rssi = current_point.get('rssi', -80)  # среднее значение если нет RSSI
    # Пример: rssi -30 (сильный) -> ширина 10, rssi -130 (слабый) -> ширина 60
    # Линейная интерполяция в диапазоне [-130, -30] -> [60, 10]
    # y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    # x0=-130, y0=60; x1=-30, y1=10
    min_rssi, max_rssi = -130, -30
    min_width, max_width = 60, 10  # Обратите внимание на порядок: слабый сигнал - большая ширина

    clamped_rssi = max(min_rssi, min(rssi, max_rssi))  # Ограничиваем RSSI

    if (max_rssi - min_rssi) == 0:  # чтобы избежать деления на ноль
        fov_width_deg = DEFAULT_FOV_WIDTH_DEG
    else:
        fov_width_deg = min_width + (clamped_rssi - min_rssi) * \
                        (max_width - min_width) / (max_rssi - min_rssi)

    # logger.debug(
    #     f"SIGMA_FOV Stub: Calculated FoV for point "
    #     f"({current_point['latitude']:.4f}, {current_point['longitude']:.4f}) "
    #     f"towards source ({source_coords['latitude']:.4f}, {source_coords['longitude']:.4f}). "
    #     f"Direction: {fov_direction_deg:.2f} deg, Width: {fov_width_deg:.2f} deg (RSSI: {rssi})."
    # )
    return {
        "fov_dir_deg": fov_direction_deg,
        "fov_width_deg": fov_width_deg
    }


def get_point_of_view(
        measurement_points: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Возвращает "точку зрения" (PoV), к которой будет применен FoV.
    Это ЗАГЛУШКА. В простом случае это может быть последняя точка измерения.

    Args:
        measurement_points: Список точек измерений.

    Returns:
        Словарь с координатами PoV: {'latitude': ..., 'longitude': ...}.
    """
    if not measurement_points:
        logger.error("Получен пустой список точек для расчета PoV.")
        return {"latitude": 0.0, "longitude": 0.0}  # Или возбудить ошибку

    # Заглушка: PoV - это просто последняя точка в маршруте
    pov = measurement_points[-1]
    logger.debug(f"SIGMA_FOV Stub: PoV is the last measurement point: ({pov['latitude']:.4f}, {pov['longitude']:.4f})")
    return {"latitude": pov['latitude'], "longitude": pov['longitude']}


def is_source_in_fov(
        pov_coords: Dict[str, float],  # {'latitude': ..., 'longitude': ...}
        fov_dir_deg: float,
        fov_width_deg: float,
        source_coords: Dict[str, float]  # {'latitude': ..., 'longitude': ...}
) -> bool:
    """
    Проверяет, находится ли источник сигнала в пределах заданного FoV из точки зрения (PoV).
    Это ЗАГЛУШКА.
    """
    angle_to_source_deg = calculate_angle_between_points(
        pov_coords['longitude'], pov_coords['latitude'],
        source_coords['longitude'], source_coords['latitude']
    )

    # Нормализация углов для корректного сравнения
    # Приводим все углы к диапазону [0, 360)
    fov_dir_deg = fov_dir_deg % 360
    angle_to_source_deg = angle_to_source_deg % 360

    # Границы FoV
    half_width = fov_width_deg / 2
    lower_bound = (fov_dir_deg - half_width + 360) % 360
    upper_bound = (fov_dir_deg + half_width + 360) % 360

    # Проверка пересечения FoV нулевого меридиана (например, FoV от 350 до 10 градусов)
    if lower_bound > upper_bound:  # FoV пересекает 0/360 градусов
        return angle_to_source_deg >= lower_bound or angle_to_source_deg <= upper_bound
    else:  # Стандартный случай
        return lower_bound <= angle_to_source_deg <= upper_bound