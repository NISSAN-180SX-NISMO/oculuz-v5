# oculuz/src/data/dataset/noise_generators/poisson_noise.py
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .base_noise_generator import BaseNoiseGenerator
from configuration.config_loader import PoissonNoiseConfig

logger = logging.getLogger(__name__)


class PoissonNoiseGenerator(BaseNoiseGenerator):
    config: PoissonNoiseConfig

    def __init__(self, config: Optional[PoissonNoiseConfig] = None, config_path: Optional[str] = None):
        if config is None:
            default_config_path = config_path or "oculuz/configuration/noise_generators/poisson_noise_config.yaml"
            final_config = PoissonNoiseConfig.load(default_config_path)
        else:
            final_config = config
        super().__init__(final_config)

    def apply_noise(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Применение Пуассоновского шума к RSSI (дБм) не является стандартным.
        # Обычно его применяют к счетным данным или к мощности в линейной шкале.
        # Здесь мы просто добавим значение (Poisson(lambda) - offset) к RSSI.
        for m in measurements:
            if 'rssi' in m:
                # np.random.poisson ожидает lambda > 0.
                if self.config.lambda_param <= 0:
                    logger.warning(f"Poisson lambda_param ({self.config.lambda_param}) <= 0. Skipping noise.")
                    continue

                poisson_sample = np.random.poisson(self.config.lambda_param)
                noise = poisson_sample - self.config.offset
                m['rssi'] += noise
                # logger.debug(f"Applied Poisson noise {noise:.2f} (sample: {poisson_sample}) to RSSI. New RSSI: {m['rssi']:.2f}")
            else:
                logger.warning("Measurement point missing 'rssi' field. Cannot apply Poisson noise.")
        return measurements