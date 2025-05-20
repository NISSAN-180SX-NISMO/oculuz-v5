# oculuz/src/data/dataset/noise_generators/gaussian_noise.py
import random
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .base_noise_generator import BaseNoiseGenerator
from configuration.config_loader import GaussianNoiseConfig

logger = logging.getLogger(__name__)

class GaussianNoiseGenerator(BaseNoiseGenerator):
    config: GaussianNoiseConfig

    def __init__(self, config: Optional[GaussianNoiseConfig] = None, config_path: Optional[str] = None):
        if config is None:
            default_config_path = config_path or "oculuz/configuration/noise_generators/gaussian_noise_config.yaml"
            final_config = GaussianNoiseConfig.load(default_config_path)
        else:
            final_config = config
        super().__init__(final_config)

    def apply_noise(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for m in measurements:
            if 'rssi' in m:
                noise = random.gauss(self.config.mean, self.config.std_dev)
                m['rssi'] += noise
                # logger.debug(f"Applied Gaussian noise {noise:.2f} to RSSI. New RSSI: {m['rssi']:.2f}")
            else:
                logger.warning("Measurement point missing 'rssi' field. Cannot apply Gaussian noise.")
        return measurements