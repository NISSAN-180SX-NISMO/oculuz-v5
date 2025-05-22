# oculuz/src/data/preprocessing/preprocessor.py

import logging
from typing import Dict, Any, Callable, List, Union, Optional
import torch

# Используем новый ConfigLoader
from configuration.config_loader import ConfigLoader
from .features_preprocessing_rules.rssi import normalize_rssi
from .features_preprocessing_rules.coordinate import get_sinusoidal_embedding

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_loader: Optional[ConfigLoader] = None,
                 default_config_path: str = "configuration/data_preprocessing_config.yaml") -> None:
        """
        Инициализирует DataPreprocessor.

        Args:
            config_loader: Экземпляр ConfigLoader с загруженной конфигурацией предобработки.
                           Если None, будет загружен конфиг по default_config_path.
            default_config_path: Путь к файлу конфигурации по умолчанию.
        """
        if config_loader is None:
            self.config_loader: ConfigLoader = ConfigLoader(default_config_path)
        else:
            self.config_loader: ConfigLoader = config_loader

        self.feature_rules: Dict[str, Callable[..., Any]] = self._register_rules()
        logger.info("DataPreprocessor initialized.")
        # logger.debug(f"DataPreprocessor config: {self.config_loader.data if hasattr(self.config_loader, 'data') else self.config_loader}")


    def _get_config_param(self, keys: List[str], default: Optional[Any] = None) -> Any:
        """Вспомогательный метод для безопасного извлечения вложенных параметров из ConfigLoader."""
        current_level = self.config_loader.data # Предполагаем, что .data возвращает словарь
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                logger.warning(f"Config parameter {' -> '.join(keys)} not found. Using default: {default}")
                return default
        return current_level

    def _register_rules(self) -> Dict[str, Callable[..., Any]]:
        try:
            rssi_min = self._get_config_param(["rssi", "min_val"], -130.0)
            rssi_max = self._get_config_param(["rssi", "max_val"], -30.0)
            coord_emb_dim = self._get_config_param(["coordinate", "embedding_dim"], 64)

        except KeyError as e:
            logger.error(f"Missing critical key in data_preprocessing_config: {e}. Cannot register rules.", exc_info=True)
            raise ValueError(f"Missing critical key in data_preprocessing_config: {e}") from e


        rules = {
            "rssi": lambda val: normalize_rssi(val, rssi_min, rssi_max),
            "latitude": lambda val: get_sinusoidal_embedding(val, coord_emb_dim),
            "longitude": lambda val: get_sinusoidal_embedding(val, coord_emb_dim),
        }
        logger.info(f"Registered preprocessing rules for features: {list(rules.keys())} "
                    f"with params: RSSI_range=({rssi_min},{rssi_max}), CoordEmbDim={coord_emb_dim}")
        return rules

    def preprocess_feature(self, feature_name: str, feature_value: Any) -> Any:
        if feature_name not in self.feature_rules:
            logger.warning(f"No preprocessing rule found for feature: {feature_name}. Returning original value.")
            return feature_value # Возвращаем как есть, если нет правила

        try:
            processed_value = self.feature_rules[feature_name](feature_value)
            return processed_value
        except Exception as e:
            logger.error(f"Error processing feature {feature_name} with value {feature_value}: {e}", exc_info=True)
            raise

    def preprocess_measurement_point(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        processed_measurement: Dict[str, Any] = {}

        if 'latitude' in measurement:
            processed_measurement['original_latitude'] = measurement['latitude']
        if 'longitude' in measurement:
            processed_measurement['original_longitude'] = measurement['longitude']
        if 'rssi' in measurement:
            processed_measurement['original_rssi'] = measurement['rssi']

        for feature_name, feature_value in measurement.items():
            if feature_name in self.feature_rules:
                if feature_name == "rssi":
                    processed_measurement["rssi_norm"] = self.preprocess_feature(feature_name, feature_value)
                elif feature_name == "latitude":
                    processed_measurement["latitude_emb"] = self.preprocess_feature(feature_name, feature_value)
                elif feature_name == "longitude":
                    processed_measurement["longitude_emb"] = self.preprocess_feature(feature_name, feature_value)
                # Добавьте другие правила здесь, если они имеют специальные имена ключей
            else:
                if feature_name not in processed_measurement: # Копируем поля, не затронутые обработкой
                    processed_measurement[feature_name] = feature_value
        return processed_measurement

    def preprocess_session_data(self, session_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not session_data:
            logger.warning("Received empty session_data for preprocessing.")
            return []
        # logger.info(f"Starting preprocessing for a session with {len(session_data)} measurement points.")
        processed_session = [self.preprocess_measurement_point(measurement) for measurement in session_data]
        # logger.info(
        #     f"Session preprocessing completed. First processed point keys: {processed_session[0].keys() if processed_session else 'No data'}")
        return processed_session