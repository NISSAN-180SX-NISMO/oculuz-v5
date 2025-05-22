# oculuz/src/model/model_loader.py

import torch
import yaml  # Убедитесь, что PyYAML установлен (pip install PyYAML)
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Type, Tuple, Optional, Dict, Any


# Для аннотаций типов и примера, предположим, что GNNModel может быть импортирован.
# В реальном коде убедитесь, что GNNModel (или ваш класс модели) доступен.
# from src.model.model import GNNModel


# Предположим, у вас есть:
# - model: экземпляр вашей обученной модели (например, GNNModel)
# - MODEL_CONFIG_PATH: путь к вашему model_config.yaml
# - DATA_PREPROCESSING_CONFIG_PATH: путь к вашему data_preprocessing_config.yaml

# from src.model.model_loader import ModelLoader # или соответствующий импорт

# loader = ModelLoader(
#     model_config_source_path=MODEL_CONFIG_PATH,
#     data_preprocessing_config_source_path=DATA_PREPROCESSING_CONFIG_PATH
# )
# saved_session_path = loader.save_model(model, training_session_name="my_first_training")
# print(f"Сессия сохранена в: {saved_session_path}")




# Предположим, у вас есть:
# - GNNModel: класс вашей модели (импортированный)
# - SAVED_SESSION_PATH: путь к сохраненной сессии (например, "oculuz/assets/train_sessions/my_first_training")

# from src.model.model import GNNModel # Убедитесь, что класс модели импортирован
# from src.model.model_loader import ModelLoader

# loaded_model, model_cfg, data_cfg = ModelLoader.load_model(
#     model_class=GNNModel,
#     training_session_path=SAVED_SESSION_PATH
# )
# print("Модель загружена!")
# print("Конфигурация модели:", model_cfg)
# print("Конфигурация данных:", data_cfg)

# # Теперь loaded_model готова к использованию для предсказаний
# # loaded_model.eval() # Уже вызвано в load_model
# # predictions = loaded_model(some_input_data)

class ModelLoader:
    """
    Класс для сохранения и загрузки моделей PyTorch вместе с их конфигурациями.
    """

    def __init__(self,
                 model_config_source_path: str,
                 data_preprocessing_config_source_path: str):
        """
        Инициализирует ModelLoader путями к ИСХОДНЫМ файлам конфигурации.
        Эти файлы будут скопированы в папку сессии при сохранении модели.

        Args:
            model_config_source_path (str): Путь к исходному файлу конфигурации модели
                                            (например, 'oculuz/configuration/model_config.yaml').
            data_preprocessing_config_source_path (str): Путь к исходному файлу конфигурации
                                                         предобработки данных (например,
                                                         'oculuz/configuration/data_preprocessing_config.yaml').
        """
        self.model_config_source_path = Path(model_config_source_path)
        self.data_preprocessing_config_source_path = Path(data_preprocessing_config_source_path)

        if not self.model_config_source_path.is_file():
            raise FileNotFoundError(
                f"Исходный файл конфигурации модели не найден: {self.model_config_source_path.resolve()}"
            )
        if not self.data_preprocessing_config_source_path.is_file():
            raise FileNotFoundError(
                f"Исходный файл конфигурации предобработки данных не найден: {self.data_preprocessing_config_source_path.resolve()}"
            )

    def save_model(self,
                   model: torch.nn.Module,
                   training_session_name: Optional[str] = None,
                   base_save_path: str = "oculuz/assets/train_sessions",
                   model_state_filename: str = "model_state.pt",
                   target_model_config_filename: str = "model_config.yaml",
                   target_data_prep_config_filename: str = "data_preprocessing_config.yaml"
                   ) -> str:
        """
        Сохраняет состояние модели и ее конфигурационные файлы в указанную директорию.

        Args:
            model (torch.nn.Module): Обученная модель PyTorch для сохранения.
            training_session_name (Optional[str]): Имя для папки сессии обучения.
                                                   Если None, генерируется имя 'train_session_<дата>_<время>'.
            base_save_path (str): Базовый путь для сохранения всех сессий обучения.
            model_state_filename (str): Имя файла для сохранения состояния модели.
            target_model_config_filename (str): Имя файла для сохранения конфигурации модели в папке сессии.
            target_data_prep_config_filename (str): Имя файла для сохранения конфигурации предобработки
                                                    в папке сессии.

        Returns:
            str: Абсолютный путь к созданной папке сессии обучения.
        """
        now = datetime.now()
        if training_session_name is None:
            training_session_name = f"train_session_{now.strftime('%Y%m%d_%H%M%S')}"

        session_dir = Path(base_save_path) / training_session_name

        # Перезаписать папку, если она уже существует
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=False)  # exist_ok=False чтобы убедиться, что папка только что создана

        # 1. Сохранить состояние модели
        model_state_file_path = session_dir / model_state_filename
        torch.save(model.state_dict(), model_state_file_path)

        # 2. Скопировать конфигурационные файлы с целевыми именами
        target_model_config_path = session_dir / target_model_config_filename
        target_data_prep_config_path = session_dir / target_data_prep_config_filename

        shutil.copy2(self.model_config_source_path, target_model_config_path)
        shutil.copy2(self.data_preprocessing_config_source_path, target_data_prep_config_path)

        # Логирование или вывод пути
        # print(f"Модель и конфигурации сохранены в: {session_dir.resolve()}")
        return str(session_dir.resolve())

    @staticmethod
    def _calculate_model_in_channels_from_config(data_prep_config: Dict[str, Any]) -> int:
        """
        Вспомогательный статический метод для вычисления `model_in_channels`
        на основе конфигурации предобработки данных.

        Эта реализация является примером и должна быть адаптирована
        к фактической структуре вашего `data_preprocessing_config.yaml` и логике
        формирования входных признаков модели.

        Args:
            data_prep_config (Dict[str, Any]): Загруженная конфигурация предобработки данных.

        Returns:
            int: Рассчитанное количество входных каналов для модели.

        Raises:
            ValueError: Если не удается определить `model_in_channels` или ключевые
                        параметры отсутствуют в конфигурации.
        """
        model_in_channels = 0

        # Пример: RSSI всегда дает 1 признак после нормализации
        # Это предположение; если RSSI не всегда используется или его размерность другая,
        # эту логику нужно будет адаптировать.
        model_in_channels += 1  # За RSSI

        # Пример: координатные эмбеддинги
        # Предполагаем, что в data_prep_config есть раздел 'coordinate_embedding'
        # и в нем ключ 'embedding_dim'.
        coordinate_embedding_config = data_prep_config.get('coordinate_embedding')
        if coordinate_embedding_config and isinstance(coordinate_embedding_config, dict):
            embedding_dim = coordinate_embedding_config.get('embedding_dim')
            if isinstance(embedding_dim, int) and embedding_dim > 0:
                model_in_channels += embedding_dim
            else:
                # Можно выдать предупреждение или ошибку, если 'embedding_dim' обязателен
                raise ValueError(
                    f"Ключ 'embedding_dim' в 'coordinate_embedding' некорректен или отсутствует "
                    f"в конфигурации предобработки: {embedding_dim}"
                )
        else:
            # Если координатные эмбеддинги являются неотъемлемой частью модели,
            # здесь следует вызывать исключение.
            raise ValueError(
                "Раздел 'coordinate_embedding' отсутствует или некорректен "
                "в конфигурации предобработки данных."
            )

        if model_in_channels <= 0:  # Должен быть строго больше 0
            raise ValueError(
                f"Рассчитанное значение model_in_channels ({model_in_channels}) некорректно. "
                "Проверьте конфигурацию предобработки и логику расчета."
            )
        return model_in_channels

    @staticmethod
    def load_model(model_class: Type[torch.nn.Module],
                   training_session_path: str,
                   model_config_filename: str = "model_config.yaml",
                   data_preprocessing_config_filename: str = "data_preprocessing_config.yaml",
                   model_state_filename: str = "model_state.pt"
                   ) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
        """
        Загружает модель и ее конфигурации из указанной папки сессии.

        Args:
            model_class (Type[torch.nn.Module]): Класс модели для инстанцирования (например, GNNModel).
            training_session_path (str): Путь к сохраненной папке сессии обучения.
            model_config_filename (str): Имя файла конфигурации модели внутри папки сессии.
            data_preprocessing_config_filename (str): Имя файла конфигурации предобработки данных
                                                      внутри папки сессии.
            model_state_filename (str): Имя файла состояния модели внутри папки сессии.

        Returns:
            Tuple[torch.nn.Module, dict, dict]: Кортеж, содержащий:
                - инстанс загруженной модели PyTorch,
                - загруженную конфигурацию модели (словарь),
                - загруженную конфигурацию предобработки данных (словарь).

        Raises:
            FileNotFoundError: Если необходимые файлы или директория сессии не найдены.
            ValueError: Если возникают проблемы с парсингом конфигураций или созданием модели.
        """
        session_dir = Path(training_session_path)
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Папка сессии обучения не найдена: {session_dir.resolve()}")

        # 1. Загрузить конфигурационные файлы
        model_cfg_file_path = session_dir / model_config_filename
        data_prep_cfg_file_path = session_dir / data_preprocessing_config_filename

        if not model_cfg_file_path.is_file():
            raise FileNotFoundError(
                f"Файл конфигурации модели не найден в папке сессии: {model_cfg_file_path.resolve()}"
            )
        if not data_prep_cfg_file_path.is_file():
            raise FileNotFoundError(
                f"Файл конфигурации предобработки данных не найден: {data_prep_cfg_file_path.resolve()}"
            )

        try:
            with open(model_cfg_file_path, 'r', encoding='utf-8') as f:
                loaded_model_config = yaml.safe_load(f)
            if not isinstance(loaded_model_config, dict):
                raise ValueError(f"Содержимое файла {model_cfg_file_path} не является словарем.")
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка парсинга YAML для файла конфигурации модели {model_cfg_file_path}: {e}")

        try:
            with open(data_prep_cfg_file_path, 'r', encoding='utf-8') as f:
                loaded_data_prep_config = yaml.safe_load(f)
            if not isinstance(loaded_data_prep_config, dict):
                raise ValueError(f"Содержимое файла {data_prep_cfg_file_path} не является словарем.")
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка парсинга YAML для файла предобработки {data_prep_cfg_file_path}: {e}")

        # 2. Вычислить model_in_channels на основе загруженного конфига предобработки
        try:
            model_in_channels = ModelLoader._calculate_model_in_channels_from_config(loaded_data_prep_config)
        except ValueError as e:
            raise ValueError(f"Ошибка при вычислении model_in_channels: {e}")

        # 3. Инстанциировать модель
        # Предполагается, что конструктор model_class принимает 'model_in_channels' и 'config'
        try:
            # Убедитесь, что ваш класс модели (например, GNNModel) импортирован в файле,
            # где вы вызываете ModelLoader.load_model
            model_instance = model_class(model_in_channels=model_in_channels, config=loaded_model_config)
        except Exception as e:
            # Логирование ошибки может быть полезно здесь
            raise ValueError(
                f"Ошибка при инстанцировании класса модели '{model_class.__name__}' с "
                f"model_in_channels={model_in_channels}. Проверьте совместимость конструктора. Ошибка: {e}"
            )

        # 4. Загрузить состояние модели
        model_state_file_path = session_dir / model_state_filename
        if not model_state_file_path.is_file():
            raise FileNotFoundError(f"Файл состояния модели не найден: {model_state_file_path.resolve()}")

        # Определить устройство для загрузки (CPU по умолчанию, можно передать как параметр)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            model_instance.load_state_dict(torch.load(model_state_file_path, map_location=device))
        except Exception as e:
            # Логирование ошибки
            raise RuntimeError(f"Ошибка при загрузке state_dict в модель из {model_state_file_path}: {e}")

        model_instance.to(device)  # Переместить модель на соответствующее устройство
        model_instance.eval()  # Перевести модель в режим оценки по умолчанию после загрузки

        # print(f"Модель успешно загружена с {model_state_file_path.resolve()} на устройство {device}.")

        return model_instance, loaded_model_config, loaded_data_prep_config