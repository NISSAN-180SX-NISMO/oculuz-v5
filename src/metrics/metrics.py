# oculuz/src/metrics/metrics.py

import numpy as np
from typing import List, Dict, Tuple, Callable, Any, Literal, Optional
import logging

# Импорт функций расчета базовых метрик
from .base_metrics.hits_persent import calculate_hits_percent_for_session
from .base_metrics.avg_width_deg import calculate_avg_width_deg_for_session

logger = logging.getLogger(__name__)


class MetricAggregator:
    """
    Вспомогательный класс для хранения списка значений метрик и вычисления агрегаций.
    (Без изменений)
    """

    def __init__(self, values: List[float]):
        if not isinstance(values, list):
            try:
                values = list(values)
            except TypeError:
                raise TypeError(
                    f"Входные 'values' должны быть списком или преобразуемы в список, получено: {type(values)}")
        self.values = [float(v) for v in values]

    def calculate_mean(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))

    def calculate_median(self) -> float:
        if not self.values:
            return 0.0
        return float(np.median(self.values))

    def __repr__(self) -> str:
        return f"MetricAggregator(count={len(self.values)}, mean={self.calculate_mean():.2f}, median={self.calculate_median():.2f})"


class MetricsCalculator:
    """
    Вычисляет и агрегирует различные метрики для оценки модели на основе предсказанных FoV.
    """

    def __init__(self):
        """
        Инициализация калькулятора метрик.
        Теперь не требует функций из sigma_fov.
        """
        self.metric_registry: Dict[str, Callable[[Dict[str, Any]], float]] = {
            "HITS_PERCENT": self._calculate_session_hits_percent,
            "AVG_WIDTH_DEG": self._calculate_session_avg_width_deg,
        }
        logger.info("MetricsCalculator инициализирован (не зависит от sigma_fov).")

    def _prepare_session_data_dict(self,
                                   predicted_fovs: List[Tuple[float, float, float]],
                                   measurement_coords: List[Tuple[float, float]],  # PoV для каждого FoV
                                   source_position: Tuple[float, float]
                                   ) -> Dict[str, Any]:
        """Вспомогательный метод для формирования словаря данных сессии."""
        return {
            "predicted_fovs": predicted_fovs,
            "measurement_coords": measurement_coords,  # Это PoV для каждого FoV
            "source_position": source_position
        }

    def _calculate_session_hits_percent(self, session_data_dict: Dict[str, Any]) -> float:
        """Обертка для вызова calculate_hits_percent_for_session."""
        return calculate_hits_percent_for_session(
            session_predicted_fovs=session_data_dict["predicted_fovs"],
            session_measurement_coords=session_data_dict["measurement_coords"],
            session_source_position=session_data_dict["source_position"]
        )

    def _calculate_session_avg_width_deg(self, session_data_dict: Dict[str, Any]) -> float:
        """Обертка для вызова calculate_avg_width_deg_for_session."""
        return calculate_avg_width_deg_for_session(
            session_predicted_fovs=session_data_dict["predicted_fovs"]
            # measurement_coords и source_position не нужны для этой базовой метрики
        )

    def calculate_metrics_for_dataset(
            self,
            dataset_predicted_fovs: List[List[Tuple[float, float, float]]],
            dataset_measurement_coords: List[List[Tuple[float, float]]],  # Координаты PoV для каждого FoV
            dataset_source_positions: List[Tuple[float, float]],
            aggregation_options: Optional[Dict[str, Literal["AVG", "MED"]]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет все зарегистрированные метрики по набору данных (списку сессий) и агрегирует их.

        Args:
            dataset_predicted_fovs (List[List[Tuple[float, float, float]]]): Список, где каждый элемент -
                список предсказанных FoV (sin_dir, cos_dir, width_deg) для одной сессии.
            dataset_measurement_coords (List[List[Tuple[float, float]]]): Список, где каждый элемент -
                список кортежей (lat, lon) координат точек (PoV), откуда "смотрит" каждый FoV.
            dataset_source_positions (List[Tuple[float, float]]): Список кортежей (lat, lon)
                координат источника для каждой сессии.
            aggregation_options (Optional[Dict[str, Literal["AVG", "MED"]]]): Словарь, указывающий
                тип агрегации. Если None, для всех метрик по умолчанию используется "AVG".

        Returns:
            Dict[str, float]: Словарь с агрегированными метриками.
        """
        num_sessions = len(dataset_predicted_fovs)
        if not (num_sessions == len(dataset_measurement_coords) == len(dataset_source_positions)):
            msg = "Размеры входных списков данных для датасета не совпадают!"
            logger.error(msg)
            raise ValueError(msg)

        if num_sessions == 0:
            logger.warning("Датасет для расчета метрик пуст.")
            return {}

        if aggregation_options is None:
            aggregation_options = {}  # Будет использован AVG по умолчанию ниже

        per_session_metric_values: Dict[str, List[float]] = {
            base_metric_name: [] for base_metric_name in self.metric_registry
        }

        for i in range(num_sessions):
            # Проверка, что для текущей сессии количество предсказанных FoV
            # совпадает с количеством координат PoV
            if len(dataset_predicted_fovs[i]) != len(dataset_measurement_coords[i]):
                logger.error(f"Для сессии {i}: количество предсказанных FoV "
                             f"({len(dataset_predicted_fovs[i])}) не совпадает с количеством "
                             f"координат PoV ({len(dataset_measurement_coords[i])}). Сессия пропущена.")
                continue  # Пропустить эту сессию

            session_data = self._prepare_session_data_dict(
                predicted_fovs=dataset_predicted_fovs[i],
                measurement_coords=dataset_measurement_coords[i],
                source_position=dataset_source_positions[i]
            )
            for base_metric_name, calculate_func in self.metric_registry.items():
                try:
                    value = calculate_func(session_data)
                    per_session_metric_values[base_metric_name].append(value)
                except Exception as e:
                    logger.error(f"Ошибка при вычислении метрики '{base_metric_name}' для сессии {i}: {e}",
                                 exc_info=True)

        final_aggregated_metrics: Dict[str, float] = {}
        for base_metric_name, values_list in per_session_metric_values.items():
            if not values_list:
                logger.warning(f"Нет значений для агрегации метрики '{base_metric_name}'.")
                continue

            aggregator = MetricAggregator(values_list)
            agg_type = aggregation_options.get(base_metric_name, "AVG").upper()

            if agg_type == "AVG":
                aggregated_value = aggregator.calculate_mean()
            elif agg_type == "MED":
                aggregated_value = aggregator.calculate_median()
            else:
                logger.warning(f"Неизвестный тип агрегации '{agg_type}' для метрики '{base_metric_name}'. "
                               f"Используется AVG по умолчанию.")
                aggregated_value = aggregator.calculate_mean()
                agg_type = "AVG"

            final_aggregated_metrics[f"{agg_type}_{base_metric_name}"] = aggregated_value

        return final_aggregated_metrics

    def add_custom_metric_calculation(self,
                                      base_metric_name: str,
                                      session_calculation_function: Callable[[Dict[str, Any]], float]):
        """
        Позволяет добавить новую функцию для расчета базовой метрики на уровне сессии.
        (Без изменений)
        """
        if base_metric_name in self.metric_registry:
            logger.warning(f"Метрика с именем '{base_metric_name}' уже существует и будет перезаписана.")
        self.metric_registry[base_metric_name] = session_calculation_function
        logger.info(f"Пользовательская метрика '{base_metric_name}' добавлена в реестр.")