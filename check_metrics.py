# В вашем скрипте обучения или оценки:
import logging
from src.metrics import MetricsCalculator # Импорт из обновленного __init__.py

# Настройка базового логирования, если еще не настроено
logging.basicConfig(level=logging.INFO)


# 1. Инициализация MetricsCalculator (теперь без аргументов sigma_fov)
metrics_calc = MetricsCalculator()

# 2. Подготовка данных (теперь с предсказанными FoV)
# Список сессий, каждая сессия - список предсказанных FoV (sin_dir, cos_dir, width_deg)
dataset_pred_fovs = [
    [(0.0, 1.0, 30.0), (0.7071, 0.7071, 45.0)], # Сессия 1: FoV1(0/360 град, 30 шир), FoV2(45 град, 45 шир)
    [(-1.0, 0.0, 60.0)]                        # Сессия 2: FoV1(270 град, 60 шир)
]
# Координаты PoV для каждого FoV (откуда "смотрит" каждый FoV)
# Это КООРДИНАТЫ ТОЧЕК ИЗМЕРЕНИЙ из вашего ТЗ
dataset_coords_pov = [
    [(56, 36), (54, 35)],   # Сессия 1, 2 точки
    [(41, -76)]                           # Сессия 2, 1 точка
]
# Позиции источников для каждой сессии
dataset_sources_pos = [
    (55, 37), # Источник для сессии 1
    (40, -74) # Источник для сессии 2
]

# 3. Опции агрегации (какие метрики как агрегировать)
agg_opts = {
    "HITS_PERCENT": "AVG",  # Средний HITS_PERCENT по всем сессиям
    "AVG_WIDTH_DEG": "MED"  # Медианная AVG_WIDTH_DEG по всем сессиям
}

# 4. Расчет метрик
try:
    calculated_metrics = metrics_calc.calculate_metrics_for_dataset(
        dataset_predicted_fovs=dataset_pred_fovs,
        dataset_measurement_coords=dataset_coords_pov, # Это PoV
        dataset_source_positions=dataset_sources_pos,
        aggregation_options=agg_opts
    )
    print("Рассчитанные метрики (обновлено):", calculated_metrics)
    # Примерный ожидаемый вывод (значения будут зависеть от входных данных):
    # Рассчитанные метрики (обновлено): {'AVG_HITS_PERCENT': ..., 'MED_AVG_WIDTH_DEG': ...}

except ValueError as e:
    print(f"Ошибка при расчете метрик: {e}")
except Exception as e:
    print(f"Непредвиденная ошибка при расчете метрик: {e}")