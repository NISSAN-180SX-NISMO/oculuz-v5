# oculuz/src/metrics/base_metrics/hits_persent.py

import numpy as np
from typing import List, Tuple

from ..utils import calculate_bearing_degrees, is_angle_in_fov, sin_cos_to_angle_degrees


def calculate_hits_percent_for_session(
        session_predicted_fovs: List[Tuple[float, float, float]],  # (sin_dir, cos_dir, width_deg)
        session_measurement_coords: List[Tuple[float, float]],  # (lat, lon) PoV для каждого FoV
        session_source_position: Tuple[float, float]  # (lat, lon) истинного источника
) -> float:
    """
    Вычисляет HITS_PERCENT для одной сессии.
    HITS_PERCENT: Процент предсказанных FoV, которые покрывают истинный источник.

    Args:
        session_predicted_fovs (List[Tuple[float, float, float]]): Список предсказанных FoV.
            Каждый элемент: (sin_dir_center, cos_dir_center, width_deg).
        session_measurement_coords (List[Tuple[float, float]]): Координаты (lat, lon)
            каждой точки измерения (интерпретируется как PoV) в сессии.
        session_source_position (Tuple[float, float]): Координаты (lat, lon) истинного источника.

    Returns:
        float: HITS_PERCENT для сессии (от 0.0 до 100.0).
    """
    num_predictions = len(session_predicted_fovs)
    if num_predictions == 0:
        return 0.0

    if len(session_measurement_coords) != num_predictions:
        raise ValueError(
            "Количество предсказанных FoV не совпадает с количеством координат точек (PoV)."
        )

    hits_count = 0
    source_lat, source_lon = session_source_position

    for i in range(num_predictions):
        # Данные для текущей i-ой точки (предсказания)
        pred_sin_dir, pred_cos_dir, pred_fov_width_deg = session_predicted_fovs[i]
        # Координаты, откуда "смотрит" этот FoV
        pov_lat, pov_lon = session_measurement_coords[i]

        # 1. Преобразовать (sin, cos) предсказанного направления FoV в угол
        pred_fov_center_deg = sin_cos_to_angle_degrees(pred_sin_dir, pred_cos_dir)

        # 2. Рассчитать угол (азимут) от PoV на истинный источник
        angle_to_source_deg = calculate_bearing_degrees(pov_lat, pov_lon, source_lat, source_lon)

        # 3. Проверить, находится ли этот угол в предсказанном секторе FoV
        try:
            if is_angle_in_fov(angle_to_source_deg, pred_fov_center_deg, pred_fov_width_deg):
                hits_count += 1
        except ValueError as e:
            # Логируем или обрабатываем ошибку (например, если ширина FoV некорректна)
            print(f"Ошибка при проверке FoV для точки {i}: {e}")  # Заменить на logger
            # Можно пропустить эту точку или считать как 0
            continue

    hits_percentage = (hits_count / num_predictions) * 100.0 if num_predictions > 0 else 0.0
    return hits_percentage