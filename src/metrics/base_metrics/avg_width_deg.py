# oculuz/src/metrics/base_metrics/avg_width_deg.py

import numpy as np
from typing import List, Tuple


def calculate_avg_width_deg_for_session(
        session_predicted_fovs: List[Tuple[float, float, float]]  # (sin_dir, cos_dir, width_deg)
) -> float:
    """
    Вычисляет AVG_WIDTH_DEG для одной сессии.
    AVG_WIDTH_DEG: Средняя ширина предсказанных FoV в сессии.

    Args:
        session_predicted_fovs (List[Tuple[float, float, float]]): Список предсказанных FoV.
            Каждый элемент: (sin_dir_center, cos_dir_center, width_deg).
            Используется только width_deg (третий элемент кортежа).

    Returns:
        float: Средняя ширина FoV в градусах для сессии.
    """
    num_predictions = len(session_predicted_fovs)
    if num_predictions == 0:
        return 0.0

    # Извлекаем только ширину FoV из каждого предсказания
    session_fov_widths_deg: List[float] = []
    for fov_data in session_predicted_fovs:
        if len(fov_data) == 3:  # Убедимся, что данные FoV корректны
            session_fov_widths_deg.append(fov_data[2])  # width_deg
        else:
            # Логирование или обработка некорректных данных FoV
            print(f"Некорректные данные FoV: {fov_data}, ширина не будет учтена.")  # Заменить на logger

    if not session_fov_widths_deg:  # Если не удалось извлечь ни одной ширины
        return 0.0

    average_width = float(np.mean(session_fov_widths_deg))
    return average_width