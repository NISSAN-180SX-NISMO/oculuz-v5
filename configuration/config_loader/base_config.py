# oculuz/configuration/config_loader/base_config.py

import yaml
import os
import logging
from abc import ABC, abstractmethod, ABCMeta
from typing import Type, TypeVar, Dict, Any

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


class ConfigMeta(ABCMeta):
    _instances: Dict[Type['BaseConfig'], 'BaseConfig'] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in ConfigMeta._instances:
            instance = super().__call__(*args, **kwargs)
            ConfigMeta._instances[cls] = instance
            # Лог о создании будет здесь, __init__ вызовется один раз
            # logger.info(f"Created singleton instance of {cls.__name__}") # Можно закомментировать, если лог в __init__ достаточен
        return ConfigMeta._instances[cls]


class BaseConfig(ABC, metaclass=ConfigMeta):
    def __init__(self) -> None:
        self._set_defaults()  # Устанавливает Python-дефолты по всей иерархии при первом создании
        logger.info(
            f"[{self.__class__.__name__}] Instance created, Python defaults set. Current state: {self._to_dict_safe()}")

    @abstractmethod
    def _set_defaults(self) -> None:
        """Устанавливает значения по умолчанию для всех параметров конфигурации (Python-дефолты)."""
        pass

    @abstractmethod
    def _validate_config(self) -> None:
        """Проверяет корректность текущих значений конфигурации."""
        pass

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь для сохранения."""
        pass

    def _to_dict_safe(self) -> Dict[str, Any]:
        """Безопасный вызов _to_dict, используется для логирования до полной инициализации."""
        try:
            return self._to_dict()
        except AttributeError:
            # Если атрибуты еще не все установлены (маловероятно после _set_defaults, но на всякий случай)
            return {"error": "Attributes not fully set for _to_dict_safe"}

    @abstractmethod
    def _from_dict(self, data: Dict[str, Any]) -> None:
        """Загружает конфигурацию из словаря (данные из YAML)."""
        pass

    @classmethod
    def get_instance(cls: Type[T]) -> T:
        if cls not in ConfigMeta._instances:
            cls()  # Создаст экземпляр, если его нет, через ConfigMeta.__call__ -> cls.__init__
        return ConfigMeta._instances[cls]

    def save(self, filepath: str) -> None:
        """Сохраняет текущую конфигурацию в YAML файл."""
        logger.debug(f"[{self.__class__.__name__}] Attempting to save configuration to: {filepath}")
        try:
            # Перед сохранением, убедимся, что текущее состояние валидно
            self._validate_config()
            logger.debug(f"[{self.__class__.__name__}] Configuration validated successfully before saving.")

            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self._to_dict(), f, allow_unicode=True, sort_keys=False)
            logger.info(f"[{self.__class__.__name__}] Configuration saved to {filepath}")
        except ValueError as ve:  # Ошибка валидации
            logger.error(f"[{self.__class__.__name__}] Validation error, cannot save configuration to {filepath}: {ve}")
            # Не сохраняем невалидный конфиг, но и не падаем, если не критично
        except IOError as e:
            logger.error(f"[{self.__class__.__name__}] IOError saving configuration to {filepath}: {e}")
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Unexpected error saving configuration to {filepath}: {e}")

    @classmethod
    def load(cls: Type[T], filepath: str) -> T:
        logger.debug(f"[{cls.__name__}] Attempting to load configuration from: {filepath}")
        instance = cls.get_instance()  # Получает или создает экземпляр с Python-дефолтами
        logger.debug(
            f"[{cls.__name__}] Got instance. Values after _set_defaults (Python defaults): {instance._to_dict_safe()}")

        try:
            if not os.path.exists(filepath):
                logger.warning(
                    f"[{cls.__name__}] File {filepath} not found. "
                    f"Instance will use current values (Python defaults). Validating and saving these to a new file."
                )
                instance._validate_config()  # Валидируем Python-дефолты
                logger.debug(f"[{cls.__name__}] Python defaults validated: {instance._to_dict()}")
                instance.save(filepath)  # Сохраняем Python-дефолты
                return instance

            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
                logger.debug(f"[{cls.__name__}] Raw content of {filepath}:\n---\n{file_content}\n---")
                f.seek(0)
                data = yaml.safe_load(f)

            logger.debug(f"[{cls.__name__}] Parsed YAML data from {filepath}: {data}")

            # _from_dict теперь отвечает за применение базы от CommonConfig и специфичных YAML данных
            instance._from_dict(data if data is not None else {})  # Передаем пустой dict, если YAML пуст

            # Валидация ПОСЛЕ применения данных из файла (или базы от CommonConfig)
            instance._validate_config()
            logger.info(
                f"[{cls.__name__}] Configuration processed from {filepath} and validated. Final values: {instance._to_dict()}")

            # Если оригинальный YAML файл был пуст (data is None), но _from_dict отработал
            # (например, взяв базу из CommonConfig), то стоит пересохранить файл,
            # чтобы он отражал актуальное состояние конфига.
            if data is None and os.path.exists(filepath):  # Файл был пуст
                logger.info(f"[{cls.__name__}] Original YAML file {filepath} was empty. "
                            f"Re-saving with current (possibly inherited/default) validated configuration.")
                instance.save(filepath)  # Пересохраняем

        except ValueError as ve:  # Ошибки валидации из _validate_config или _from_dict
            logger.error(f"[{cls.__name__}] Validation error during processing of {filepath}: {ve}. "
                         f"Instance state may be based on Python defaults or partially loaded data (if _from_dict failed before validation).")
            # Можно попытаться откатиться к безопасным Python-дефолтам и сохранить их
            try:
                logger.warning(
                    f"[{cls.__name__}] Attempting to revert to Python defaults, validate, and save due to error.")
                instance._set_defaults()  # Сброс к Python-дефолтам
                instance._validate_config()  # Валидация Python-дефолтов
                instance.save(filepath)  # Сохраняем безопасные дефолты
                logger.info(f"[{cls.__name__}] Reverted to Python defaults, validated and saved to {filepath}.")
            except Exception as e_revert:
                logger.error(
                    f"[{cls.__name__}] Failed to revert to/validate/save Python defaults after load error: {e_revert}")

        except (IOError, yaml.YAMLError) as e_file:
            logger.error(
                f"[{cls.__name__}] File/YAML error loading {filepath}: {e_file}. Instance uses Python defaults.")
            # Аналогично, можно попытаться сохранить Python-дефолты
            try:
                instance._validate_config()  # Python-дефолты должны быть уже валидны
                instance.save(filepath)
            except Exception as e_save_default:
                logger.error(f"[{cls.__name__}] Error saving default config after file/YAML error: {e_save_default}")

        except Exception as e_unexpected:
            logger.error(
                f"[{cls.__name__}] Unexpected error loading {filepath}: {e_unexpected}. Instance uses Python defaults.")

        logger.debug(
            f"[{cls.__name__}] Returning instance. Final values for {cls.__name__}: {instance._to_dict_safe()}")
        return instance