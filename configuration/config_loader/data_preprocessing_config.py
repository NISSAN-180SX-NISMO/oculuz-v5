# oculuz/configuration/config_loader/data_preprocessing_config.py

import logging
from typing import Dict, Any
from .base_config import BaseConfig

logger = logging.getLogger(__name__)

class DataPreprocessingConfig(BaseConfig):
    """
    Конфигурация для предобработки данных.
    Является синглтоном.
    """

    # RSSI parameters
    rssi_min_val: float
    rssi_max_val: float

    # Coordinate parameters
    coordinate_embedding_dim: int

    def _set_defaults(self) -> None:
        """Устанавливает значения по умолчанию."""
        self.rssi_min_val = -130.0
        self.rssi_max_val = -30.0
        self.coordinate_embedding_dim = 64
        self._validate_config()

    def _validate_config(self) -> None:
        """Проверяет корректность значений конфигурации."""
        if not isinstance(self.rssi_min_val, (int, float)):
            raise ValueError("rssi_min_val должен быть числом.")
        if not isinstance(self.rssi_max_val, (int, float)):
            raise ValueError("rssi_max_val должен быть числом.")
        if self.rssi_min_val >= self.rssi_max_val:
            raise ValueError("rssi_min_val должен быть меньше rssi_max_val.")

        if not isinstance(self.coordinate_embedding_dim, int):
            raise ValueError("coordinate_embedding_dim должен быть целым числом.")
        if self.coordinate_embedding_dim <= 0:
            raise ValueError("coordinate_embedding_dim должен быть положительным.")
        if self.coordinate_embedding_dim % 2 != 0:
            raise ValueError("coordinate_embedding_dim должен быть четным числом для синусоидального кодирования.")
        logger.debug("DataPreprocessingConfig validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь."""
        return {
            "rssi": {
                "min_val": self.rssi_min_val,
                "max_val": self.rssi_max_val,
            },
            "coordinate": {
                "embedding_dim": self.coordinate_embedding_dim,
            }
        }

    def _from_dict(self, data: Dict[str, Any]) -> None:
        """Загружает конфигурацию из словаря."""
        rssi_data = data.get("rssi", {})
        self.rssi_min_val = float(rssi_data.get("min_val", self.rssi_min_val))
        self.rssi_max_val = float(rssi_data.get("max_val", self.rssi_max_val))

        coordinate_data = data.get("coordinate", {})
        self.coordinate_embedding_dim = int(coordinate_data.get("embedding_dim", self.coordinate_embedding_dim))
        self._validate_config()