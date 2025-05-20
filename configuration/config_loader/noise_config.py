# oculuz/configuration/config_loader/noise_config.py
import logging
from typing import Dict, Any, Optional, List
from .base_config import BaseConfig

logger = logging.getLogger(__name__)

class CommonNoiseConfig(BaseConfig):
    enabled_noise_types: List[str] # Список активных типов шума, например ["gaussian", "poisson"]

    def _set_defaults(self) -> None:
        self.enabled_noise_types = [] # По умолчанию шум выключен
        self._validate_config()

    def _validate_config(self) -> None:
        if not isinstance(self.enabled_noise_types, list):
            raise ValueError("enabled_noise_types должен быть списком.")
        # Можно добавить проверку на допустимые типы шума, если они известны заранее
        logger.debug(f"{self.__class__.__name__} validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        return {"enabled_noise_types": self.enabled_noise_types}

    def _from_dict(self, data: Dict[str, Any]) -> None:
        self.enabled_noise_types = data.get("enabled_noise_types", self.enabled_noise_types)
        self._validate_config()


class GaussianNoiseConfig(BaseConfig):
    mean: float
    std_dev: float

    def _set_defaults(self) -> None:
        self.mean = 0.0
        self.std_dev = 5.0 # дБ
        self._validate_config()

    def _validate_config(self) -> None:
        if not isinstance(self.mean, (float, int)):
            raise ValueError("Gaussian noise mean должен быть числом.")
        if not (isinstance(self.std_dev, (float, int)) and self.std_dev >= 0):
            raise ValueError("Gaussian noise std_dev должен быть неотрицательным числом.")
        logger.debug(f"{self.__class__.__name__} validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std_dev": self.std_dev}

    def _from_dict(self, data: Dict[str, Any]) -> None:
        self.mean = float(data.get("mean", self.mean))
        self.std_dev = float(data.get("std_dev", self.std_dev))
        self._validate_config()


class PoissonNoiseConfig(BaseConfig):
    # Пуассоновский шум обычно применяется к счетным данным.
    # Применение его напрямую к RSSI (дБм) не совсем стандартно.
    # Возможно, имелось в виду преобразование RSSI в некую "мощность" (линейную шкалу),
    # добавление пуассоновского шума, и обратное преобразование в дБм.
    # Для простоты, будем считать, что это некий параметр 'lambda',
    # который как-то влияет на RSSI, например, масштабируя его или добавляя случайное значение из Пуассона.
    # Это требует уточнения. Пока сделаем заглушку.
    # Предположим, что мы добавляем к RSSI значение, сэмплированное из Пуассона(lambda) - offset,
    # чтобы шум мог быть и отрицательным.
    lambda_param: float
    offset: float # Для центрирования шума вокруг 0

    def _set_defaults(self) -> None:
        self.lambda_param = 3.0
        self.offset = 3.0 # Чтобы E[Poisson(lambda) - offset] = 0
        self._validate_config()

    def _validate_config(self) -> None:
        if not (isinstance(self.lambda_param, (float, int)) and self.lambda_param > 0):
            raise ValueError("Poisson noise lambda_param должен быть положительным числом.")
        if not isinstance(self.offset, (float, int)):
            raise ValueError("Poisson noise offset должен быть числом.")
        logger.debug(f"{self.__class__.__name__} validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        return {"lambda": self.lambda_param, "offset": self.offset} # YAML не любит lambda как ключ

    def _from_dict(self, data: Dict[str, Any]) -> None:
        self.lambda_param = float(data.get("lambda", self.lambda_param))
        self.offset = float(data.get("offset", self.offset))
        self._validate_config()


class CustomNoiseConfig(BaseConfig):
    # Параметры для вашего кастомного шума
    # Пока оставим пустым
    param1: Any
    param2: Any

    def _set_defaults(self) -> None:
        self.param1 = "default_value1"
        self.param2 = 0
        self._validate_config()

    def _validate_config(self) -> None:
        # Добавьте проверки для ваших параметров
        logger.debug(f"{self.__class__.__name__} validated successfully.")
        pass

    def _to_dict(self) -> Dict[str, Any]:
        return {"param1": self.param1, "param2": self.param2}

    def _from_dict(self, data: Dict[str, Any]) -> None:
        self.param1 = data.get("param1", self.param1)
        self.param2 = data.get("param2", self.param2)
        self._validate_config()