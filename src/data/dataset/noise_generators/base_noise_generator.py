# oculuz/src/data/dataset/noise_generators/base_noise_generator.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Предполагаем, что ConfigLoader находится здесь
from configuration.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class BaseNoiseGenerator(ABC):
    def __init__(self, config: ConfigLoader):
        """
        Инициализирует базовый генератор шума.

        Args:
            config: Экземпляр ConfigLoader с загруженной конфигурацией.
        """
        self.config: ConfigLoader = config
        # Используем config.data для получения словаря, если ConfigLoader его предоставляет
        # или сам объект ConfigLoader, если он поддерживает форматирование в строку.
        # Для примера, будем использовать .data, если оно есть, или сам объект.
        config_data_for_log = self.config.data if hasattr(self.config, 'data') else self.config
        logger.info(f"Initialized {self.__class__.__name__} with config: {config_data_for_log}")

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