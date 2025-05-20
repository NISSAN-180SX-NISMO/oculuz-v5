# oculuz/src/data/dataset/noise_generators/base_noise_generator.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, TypeVar, Optional

from configuration.config_loader.base_config import BaseConfig # Используем BaseConfig

logger = logging.getLogger(__name__)
NoiseConfigType = TypeVar('NoiseConfigType', bound=BaseConfig)


class BaseNoiseGenerator(ABC):
    def __init__(self, config: NoiseConfigType, config_path: Optional[str] = None):
        if config_path: # Если путь указан, загружаем конфиг из файла
             self.config = config.__class__.load(config_path) # type: ignore
        else: # Иначе используем переданный объект конфига
             self.config = config
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config._to_dict()}")

    @abstractmethod
    def apply_noise(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Применяет шум к списку измерений (в частности, к полю 'rssi').
        Модифицирует и возвращает тот же список.

        Args:
            measurements: Список словарей, каждый из которых содержит 'rssi'.

        Returns:
            Тот же список словарей с измененным 'rssi'.
        """
        pass