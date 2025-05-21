import yaml
from pathlib import Path
from typing import Any, Dict, Union, List

class ConfigLoader:
    """
    Класс для загрузки и доступа к конфигурационным параметрам из YAML файла.

    Позволяет обращаться к загруженным данным как к словарю.
    Выбрасывает KeyError, если ключ не найден.
    """
    def __init__(self, config_path: Union[str, Path]):
        """
        Инициализирует ConfigLoader и загружает конфигурацию из указанного YAML файла.

        Args:
            config_path (Union[str, Path]): Путь к YAML файлу конфигурации.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            yaml.YAMLError: Если произошла ошибка при парсинге YAML файла.
            TypeError: Если загруженные данные не являются словарем.
        """
        self._config_path = Path(config_path)
        self._data: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Загружает конфигурацию из YAML файла.
        """
        if not self._config_path.is_file():
            raise FileNotFoundError(f"Файл конфигурации не найден: {self._config_path}")

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Ошибка парсинга YAML файла {self._config_path}: {e}")
        except Exception as e:
            raise Exception(f"Неожиданная ошибка при чтении файла {self._config_path}: {e}")

        if not isinstance(data, dict):
            raise TypeError(
                f"Содержимое файла конфигурации {self._config_path} должно быть словарем (dict), "
                f"получен {type(data)}."
            )
        return data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Возвращает значение по ключу. Если ключ не найден, возвращает значение по умолчанию.
        Аналогично методу get() для словарей.

        Args:
            key (str): Ключ для поиска.
            default (Any, optional): Значение, которое будет возвращено, если ключ не найден.
                                     По умолчанию None.

        Returns:
            Any: Значение, связанное с ключом, или значение по умолчанию.
        """
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Позволяет получать доступ к элементам конфигурации по ключу, как в словаре.

        Args:
            key (str): Ключ для доступа к элементу.

        Returns:
            Any: Значение, связанное с ключом.

        Raises:
            KeyError: Если ключ не найден в конфигурации.
        """
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"Ключ '{key}' не найден в конфигурации ({self._config_path}).")

    def __contains__(self, key: str) -> bool:
        """
        Проверяет наличие ключа в конфигурации.

        Args:
            key (str): Ключ для проверки.

        Returns:
            bool: True, если ключ существует, иначе False.
        """
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self._config_path}', loaded_keys={list(self._data.keys())})"

    @property
    def data(self) -> Dict[str, Any]:
        """
        Возвращает копию загруженных данных конфигурации.
        Это сделано для предотвращения случайного изменения внутреннего состояния.
        Если требуется модификация "на лету", ее нужно реализовывать отдельно.
        """
        return self._data.copy()

    @property
    def config_file_path(self) -> Path:
        """
        Возвращает путь к загруженному файлу конфигурации.
        """
        return self._config_path