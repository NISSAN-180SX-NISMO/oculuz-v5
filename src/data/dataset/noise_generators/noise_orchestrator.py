# oculuz/src/data/dataset/noise_generators/noise_orchestrator.py
import logging
from typing import List, Dict, Any, Optional

from configuration.config_loader import (
    CommonNoiseConfig,
    GaussianNoiseConfig,
    PoissonNoiseConfig
)
from .base_noise_generator import BaseNoiseGenerator
from .gaussian_noise import GaussianNoiseGenerator
from .poisson_noise import PoissonNoiseGenerator


logger = logging.getLogger(__name__)


class NoiseOrchestrator:
    def __init__(
            self,
            common_noise_config: Optional[CommonNoiseConfig] = None,
            gaussian_config: Optional[GaussianNoiseConfig] = None,
            poisson_config: Optional[PoissonNoiseConfig] = None,
            # Пути к файлам конфигов, если объекты не переданы
            common_noise_config_path: str = "oculuz/configuration/noise_generators/common_noise_config.yaml",
            gaussian_config_path: str = "oculuz/configuration/noise_generators/gaussian_noise_config.yaml",
            poisson_config_path: str = "oculuz/configuration/noise_generators/poisson_noise_config.yaml"
    ):
        self.common_config = common_noise_config or CommonNoiseConfig.load(common_noise_config_path)

        self.noise_generators: Dict[str, BaseNoiseGenerator] = {}

        if "gaussian" in self.common_config.enabled_noise_types:
            g_config = gaussian_config or GaussianNoiseConfig.load(gaussian_config_path)
            self.noise_generators["gaussian"] = GaussianNoiseGenerator(g_config)
            logger.info("Gaussian noise enabled.")

        if "poisson" in self.common_config.enabled_noise_types:
            p_config = poisson_config or PoissonNoiseConfig.load(poisson_config_path)
            self.noise_generators["poisson"] = PoissonNoiseGenerator(p_config)
            logger.info("Poisson noise enabled.")

        if not self.common_config.enabled_noise_types:
            logger.info("No noise types enabled in common_noise_config.")
        elif not self.noise_generators:
            logger.warning("Noise types enabled in config, but no generators were initialized. Check paths/configs.")

    def apply_all_enabled_noise(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Применяет все включенные типы шума последовательно к измерениям.
        """
        if not self.common_config.enabled_noise_types or not self.noise_generators:
            # logger.debug("No noise to apply.")
            return measurements

        # logger.debug(f"Applying noise types: {list(self.noise_generators.keys())}")
        for noise_type in self.common_config.enabled_noise_types:
            if noise_type in self.noise_generators:
                generator = self.noise_generators[noise_type]
                measurements = generator.apply_noise(measurements)  # Модифицирует 'rssi' inplace
                # logger.debug(f"Applied {noise_type} noise.")
            else:
                logger.warning(f"Noise type '{noise_type}' enabled in common config but generator not found.")
        return measurements