# oculuz/src/data/preprocessing/preprocessor.py

import logging
from typing import Dict, Any, Callable, List, Union
import torch

from configuration.config_loader import DataPreprocessingConfig
# Импортируем напрямую, так как они здесь используются
from .features_preprocessing_rules.rssi import normalize_rssi
from .features_preprocessing_rules.coordinate import get_sinusoidal_embedding

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс-оркестратор для предобработки данных сессии.
    Применяет различные правила предобработки к признакам измерений.
    Возвращает данные, готовые для сборки в граф.
    """

    def __init__(self, config: DataPreprocessingConfig = None) -> None:
        if config is None:
            self.config = DataPreprocessingConfig.get_instance()
            # Путь к файлу должен быть определен глобально или передан
            default_config_path = "oculuz/configuration/data_preprocessing_config.yaml"
            # Загружаем или создаем дефолтный, если его нет
            DataPreprocessingConfig.load(default_config_path)
        else:
            self.config = config

        self.feature_rules: Dict[str, Callable[..., Any]] = self._register_rules()
        logger.info("DataPreprocessor initialized.")

    def _register_rules(self) -> Dict[str, Callable[..., Any]]:
        rules = {
            "rssi": lambda val: normalize_rssi(
                val, self.config.rssi_min_val, self.config.rssi_max_val
            ),
            "latitude": lambda val: get_sinusoidal_embedding(
                val, self.config.coordinate_embedding_dim
            ),
            "longitude": lambda val: get_sinusoidal_embedding(
                val, self.config.coordinate_embedding_dim
            ),
        }
        logger.info(f"Registered preprocessing rules for features: {list(rules.keys())}")
        return rules

    def preprocess_feature(self, feature_name: str, feature_value: Any) -> Any:
        if feature_name not in self.feature_rules:
            logger.error(f"No preprocessing rule found for feature: {feature_name}")
            raise ValueError(f"Неизвестное имя признака для предобработки: {feature_name}")

        try:
            processed_value = self.feature_rules[feature_name](feature_value)
            # logger.debug(f"Processed feature '{feature_name}' from {feature_value} to {processed_value}") # Может быть слишком много логов
            return processed_value
        except Exception as e:
            logger.error(f"Error processing feature {feature_name} with value {feature_value}: {e}")
            raise

    def preprocess_measurement_point(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет предобработку всех известных признаков в одной точке измерения.
        Сохраняет оригинальные значения координат для использования в KNN.

        Args:
            measurement: Словарь, представляющий одну точку измерения
                         (например, {'longitude': 10.0, 'latitude': 20.0, 'rssi': -50}).

        Returns:
            Словарь с обработанными признаками и сохраненными оригинальными координатами.
            Пример:
            {
                'rssi_norm': 0.7,
                'latitude_emb': torch.Tensor([...]),
                'longitude_emb': torch.Tensor([...]),
                'original_latitude': 20.0,
                'original_longitude': 10.0,
                'original_rssi': -50 # Можно также сохранять оригинальный RSSI если нужно
                # ... другие поля из measurement могут быть сохранены без изменений
            }
        """
        processed_measurement: Dict[str, Any] = {}
        # Сохраняем оригинальные значения, которые могут понадобиться для создания графа или других шагов
        if 'latitude' in measurement:
            processed_measurement['original_latitude'] = measurement['latitude']
        if 'longitude' in measurement:
            processed_measurement['original_longitude'] = measurement['longitude']
        if 'rssi' in measurement:  # Сохраняем оригинальный RSSI
            processed_measurement['original_rssi'] = measurement['rssi']

        for feature_name, feature_value in measurement.items():
            if feature_name in self.feature_rules:
                if feature_name == "rssi":
                    processed_measurement["rssi_norm"] = self.preprocess_feature(feature_name, feature_value)
                elif feature_name == "latitude":
                    processed_measurement["latitude_emb"] = self.preprocess_feature(feature_name, feature_value)
                elif feature_name == "longitude":
                    processed_measurement["longitude_emb"] = self.preprocess_feature(feature_name, feature_value)
                else:
                    # Если есть другие правила, но без специального именования ключа
                    processed_measurement[f"{feature_name}_processed"] = self.preprocess_feature(feature_name,
                                                                                                 feature_value)
            else:
                # Сохраняем нетронутыми признаки, для которых нет правил, если они еще не были скопированы
                if feature_name not in processed_measurement:
                    processed_measurement[feature_name] = feature_value
        return processed_measurement

    def preprocess_session_data(self, session_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Starting preprocessing for a session with {len(session_data)} measurement points.")
        processed_session = [self.preprocess_measurement_point(measurement) for measurement in session_data]
        logger.info(
            f"Session preprocessing completed. First processed point: {processed_session[0] if processed_session else 'No data'}")
        return processed_session