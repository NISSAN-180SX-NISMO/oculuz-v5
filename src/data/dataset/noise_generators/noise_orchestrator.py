# oculuz/src/data/dataset/noise_generators/noise_orchestrator.py
import logging
from typing import List, Dict, Any, Optional, Union

from configuration.config_loader import ConfigLoader  # Используем новый ConfigLoader
from .base_noise_generator import BaseNoiseGenerator
from .gaussian_noise import GaussianNoiseGenerator
from .poisson_noise import PoissonNoiseGenerator

logger = logging.getLogger(__name__)


class NoiseOrchestrator:
    def __init__(
            self,
            common_noise_config_source: Optional[Union[ConfigLoader, str]] = None,
            gaussian_config_source: Optional[Union[ConfigLoader, str]] = None,
            poisson_config_source: Optional[Union[ConfigLoader, str]] = None,
            # Пути по умолчанию, если источники не переданы
            default_common_noise_config_path: str = "configuration/noise_generators/common_noise_config.yaml",
            default_gaussian_config_path: str = "configuration/noise_generators/gaussian_noise_config.yaml",
            default_poisson_config_path: str = "configuration/noise_generators/poisson_noise_config.yaml"
    ):
        if isinstance(common_noise_config_source, ConfigLoader):
            self.common_config = common_noise_config_source
        elif isinstance(common_noise_config_source, str):
            self.common_config = ConfigLoader(common_noise_config_source)
        else:
            self.common_config = ConfigLoader(default_common_noise_config_path)

        self.noise_generators: Dict[str, BaseNoiseGenerator] = {}

        enabled_noise_types = []
        try:
            enabled_noise_types = self.common_config["enabled_noise_types"]
            if not isinstance(enabled_noise_types, list):
                logger.warning(
                    f"'enabled_noise_types' in common_config is not a list: {enabled_noise_types}. Disabling all noise.")
                enabled_noise_types = []
        except KeyError:
            logger.warning("'enabled_noise_types' key missing in common_noise_config. Disabling all noise.")
            # common_config.data if hasattr(self.common_config, 'data') else self.common_config

        if "gaussian" in enabled_noise_types:
            g_config_loader: Optional[ConfigLoader] = None
            if isinstance(gaussian_config_source, ConfigLoader):
                g_config_loader = gaussian_config_source
            elif isinstance(gaussian_config_source, str):
                g_config_loader = ConfigLoader(gaussian_config_source)
            else:
                g_config_loader = ConfigLoader(default_gaussian_config_path)

            if g_config_loader:
                self.noise_generators["gaussian"] = GaussianNoiseGenerator(g_config_loader)
                logger.info("Gaussian noise enabled.")

        if "poisson" in enabled_noise_types:
            p_config_loader: Optional[ConfigLoader] = None
            if isinstance(poisson_config_source, ConfigLoader):
                p_config_loader = poisson_config_source
            elif isinstance(poisson_config_source, str):
                p_config_loader = ConfigLoader(poisson_config_source)
            else:
                p_config_loader = ConfigLoader(default_poisson_config_path)

            if p_config_loader:
                self.noise_generators["poisson"] = PoissonNoiseGenerator(p_config_loader)
                logger.info("Poisson noise enabled.")

        if not enabled_noise_types:
            logger.info("No noise types specified in common_noise_config.")
        elif not self.noise_generators and any(n_type in ["gaussian", "poisson"] for n_type in
                                               enabled_noise_types):  # Check if any known types were expected
            logger.warning(
                "Noise types enabled in config, but no corresponding generators were initialized. Check paths/configs for these types.")

    def apply_all_enabled_noise(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Применяет все включенные типы шума последовательно к измерениям.
        """
        enabled_noise_types = []
        try:
            enabled_noise_types = self.common_config["enabled_noise_types"]
            if not isinstance(enabled_noise_types, list):  # Доп. проверка на случай если конфиг изменился после init
                enabled_noise_types = []
        except KeyError:
            enabled_noise_types = []

        if not enabled_noise_types or not self.noise_generators:
            # logger.debug("No noise to apply based on current configuration or initialized generators.")
            return measurements

        # logger.debug(f"Applying noise types: {list(self.noise_generators.keys())}")
        for noise_type in enabled_noise_types:  # Итерируемся по типам из конфига
            if noise_type in self.noise_generators:
                generator = self.noise_generators[noise_type]
                measurements = generator.apply_noise(measurements)
                # logger.debug(f"Applied {noise_type} noise.")
            else:
                # Логируем, только если тип был в общем конфиге, но для него нет генератора
                # (например, "custom" шум еще не реализован, но указан в enabled_noise_types)
                logger.warning(
                    f"Noise type '{noise_type}' enabled in common config but its generator was not found/initialized.")
        return measurements