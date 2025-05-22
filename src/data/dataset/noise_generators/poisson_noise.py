# oculuz/src/data/dataset/noise_generators/poisson_noise.py
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .base_noise_generator import BaseNoiseGenerator
from configuration.config_loader import ConfigLoader # Используем новый ConfigLoader

logger = logging.getLogger(__name__)

class PoissonNoiseGenerator(BaseNoiseGenerator):
    def __init__(self, config: Optional[Union[ConfigLoader, str]] = None):
        """
        Инициализирует генератор пуассоновского шума.

        Args:
            config: Экземпляр ConfigLoader, путь к файлу конфигурации, или None
                    чтобы использовать путь по умолчанию.
        """
        final_config: ConfigLoader
        if isinstance(config, ConfigLoader):
            final_config = config
        elif isinstance(config, str):
            final_config = ConfigLoader(config)
        else:
            default_config_path = "oculuz/configuration/noise_generators/poisson_noise_config.yaml"
            final_config = ConfigLoader(default_config_path)
        super().__init__(final_config)

    def apply_noise(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            lambda_param = self.config["parameters"]["lambda_param"]
            offset = self.config["parameters"]["offset"]
        except KeyError as e:
            logger.error(f"Missing key in PoissonNoiseGenerator config: {e}. Config data: {self.config.data if hasattr(self.config, 'data') else self.config}")
            return measurements # Не применяем шум, если конфиг неполный

        for m in measurements:
            if 'rssi' in m:
                if lambda_param <= 0:
                    logger.warning(f"Poisson lambda_param ({lambda_param}) <= 0. Skipping noise.")
                    continue

                poisson_sample = np.random.poisson(lambda_param)
                noise = poisson_sample - offset
                m['rssi'] += noise
                # logger.debug(f"Applied Poisson noise {noise:.2f} (sample: {poisson_sample}) to RSSI. New RSSI: {m['rssi']:.2f}")
            else:
                logger.warning("Measurement point missing 'rssi' field. Cannot apply Poisson noise.")
        return measurements