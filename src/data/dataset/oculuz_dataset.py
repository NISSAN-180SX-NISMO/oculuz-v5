# oculuz/src/data/dataset/oculuz_dataset.py
import torch
import torch_geometric
from torch.utils.data import Dataset, DataLoader
import random
import math
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional, Type

from configuration.config_loader import (
    DataPreprocessingConfig, GraphConfig,
    CommonRouteConfig, DirectRouteConfig, CircleRouteConfig, ArcRouteConfig, RandomWalkRouteConfig,
    CommonNoiseConfig, GaussianNoiseConfig, PoissonNoiseConfig, CustomNoiseConfig
)
from src.data.preprocessing import DataPreprocessor, CoderGraph
from src.data.dataset.route_generators import (
    BaseRouteGenerator, DirectRouteGenerator, CircleRouteGenerator, ArcRouteGenerator, RandomWalkRouteGenerator
)
from src.data.dataset.noise_generators import NoiseOrchestrator
import sigma_fov as sigma_fov  # Импорт модуля sigma_fov

logger = logging.getLogger(__name__)


class OculuzDataset(Dataset):
    def __init__(
            self,
            dataset_size: int,
            route_generator_specs: List[Dict[str, Any]],
            # e.g., [{"type": "direct", "weight": 0.5, "config_path": "..."}]
            data_prep_config: Optional[DataPreprocessingConfig] = None,
            graph_config: Optional[GraphConfig] = None,
            noise_common_config: Optional[CommonNoiseConfig] = None,
            noise_configs: Optional[Dict[str, Any]] = None,  # {"gaussian": GaussianNoiseConfig(), ...}
            # Пути для авто-загрузки конфигов, если они не переданы как объекты
            data_prep_config_path: str = "oculuz/configuration/data_preprocessing_config.yaml",
            graph_config_path: str = "oculuz/configuration/graph_config.yaml",
            noise_common_config_path: str = "oculuz/configuration/noise_generators/common_noise_config.yaml"
            # noise_configs_paths можно будет добавить для каждого типа шума
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.route_generator_specs = route_generator_specs

        self.data_prep_config = data_prep_config or DataPreprocessingConfig.load(data_prep_config_path)
        self.graph_config = graph_config or GraphConfig.load(graph_config_path)

        self.preprocessor = DataPreprocessor(config=self.data_prep_config)
        self.coder_graph = CoderGraph(graph_config=self.graph_config, data_prep_config=self.data_prep_config)

        self._initialize_route_generators()

        # Инициализация шума
        # Если noise_configs не передан, загрузчики конфигов шума сами найдут свои файлы по умолчанию
        g_conf = noise_configs.get("gaussian") if noise_configs else None
        p_conf = noise_configs.get("poisson") if noise_configs else None
        c_conf = noise_configs.get("custom") if noise_configs else None

        self.noise_orchestrator = NoiseOrchestrator(
            common_noise_config=noise_common_config,  # может быть None, тогда загрузится из файла
            gaussian_config=g_conf,
            poisson_config=p_conf,
            common_noise_config_path=noise_common_config_path
        )

        logger.info(f"OculuzDataset initialized. Size: {self.dataset_size}, "
                    f"Num generator types: {len(self.route_generators)}. "
                    f"Noise enabled: {self.noise_orchestrator.common_config.enabled_noise_types}")

    def _initialize_route_generators(self):
        self.route_generators: List[BaseRouteGenerator] = []
        self.route_generator_weights: List[float] = []

        gen_map: Dict[str, Type[BaseRouteGenerator]] = {
            "direct": DirectRouteGenerator,
            "circle": CircleRouteGenerator,
            "arc": ArcRouteGenerator,
            "random_walk": RandomWalkRouteGenerator,
        }
        config_map: Dict[str, Type[CommonRouteConfig]] = {
            "direct": DirectRouteConfig,
            "circle": CircleRouteConfig,
            "arc": ArcRouteConfig,
            "random_walk": RandomWalkRouteConfig,
        }

        for spec in self.route_generator_specs:
            gen_type = spec["type"]
            weight = spec.get("weight", 1.0)
            config_path = spec.get(
                "config_path")  # e.g., "oculuz/configuration/route_generators/direct_route_config.yaml"

            if gen_type not in gen_map:
                logger.error(f"Unknown route generator type: {gen_type}")
                raise ValueError(f"Unknown route generator type: {gen_type}")

            # Загрузка или создание конфига для конкретного генератора
            # Если config_path не указан, конструктор генератора загрузит конфиг по умолчанию
            # Если config_path указан, он будет использован.
            # Конструкторы генераторов ожидают либо объект конфига, либо config_path

            generator_class = gen_map[gen_type]
            # Мы передаем config_path, конструктор генератора сам загрузит/создаст конфиг
            generator_instance = generator_class(config_path=config_path)

            self.route_generators.append(generator_instance)
            self.route_generator_weights.append(weight)
            logger.info(f"Initialized route generator: {gen_type} with weight {weight} "
                        f" (config path: {config_path or 'default'})")

        if not self.route_generators:
            logger.error("No route generators were initialized. Check route_generator_specs.")
            raise ValueError("No route generators specified or initialized.")

    def _generate_single_session_data(self, session_idx: int) -> Tuple[
        List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]], str]:
        """
        Генерирует "сырые" данные для одной сессии: измерения, FoV, источник.
        Returns:
            - measurements_with_noisy_rssi_and_fov (List[Dict]):
                Каждый элемент: {'latitude': float, 'longitude': float, 'rssi': float (зашумленный),
                                 'fov_dir_sin': float, 'fov_dir_cos': float, 'fov_width_deg': float,
                                 'clean_rssi': float (чистый RSSI)}
            - source_coords (Dict): {'latitude': float, 'longitude': float}
            - fov_data_for_csv (List[Dict]): Данные FoV для CSV в нужном формате.
            - session_id (str): Уникальный ID сессии.
        """
        session_id = str(uuid.uuid4())

        # 1. Выбор генератора маршрута и генерация чистого маршрута
        generator = random.choices(self.route_generators, weights=self.route_generator_weights, k=1)[0]
        # clean_measurements: [{'latitude', 'longitude', 'rssi' (чистый)}, ...]
        clean_measurements, source_coords = generator.generate_route()

        if not clean_measurements:  # Если генерация маршрута не удалась
            logger.warning(f"Session {session_id} (idx {session_idx}): Route generation failed. Returning empty data.")
            # Возвращаем пустые структуры, чтобы избежать падений дальше
            return [], {"latitude": 0, "longitude": 0}, [], session_id

        # Копируем для сохранения чистых RSSI
        clean_rssi_values = [m['rssi'] for m in clean_measurements]

        # 2. Применение шума (модифицирует 'rssi' в clean_measurements)
        # Теперь clean_measurements содержит зашумленный RSSI, если шум включен
        noisy_measurements = self.noise_orchestrator.apply_all_enabled_noise([m.copy() for m in clean_measurements])
        # logger.debug(f"Session {session_id} (idx {session_idx}): Generated {len(noisy_measurements)} noisy measurements. "
        #              f"First noisy RSSI: {noisy_measurements[0]['rssi'] if noisy_measurements else 'N/A'}. "
        #              f"First clean RSSI: {clean_rssi_values[0] if clean_rssi_values else 'N/A'}")

        # 3. Расчет FoV для каждой точки
        # FoV рассчитывается на основе ЧИСТЫХ данных (по ТЗ - "истинный FoV")
        # но функция sigma_fov.calculate_fov_for_point может принимать и другие данные,
        # в ТЗ указано, что она принимает "набор из N точек". Будем передавать чистые.

        session_fov_data: List[Dict[str, Any]] = []  # Для CSV
        measurements_with_fov_and_clean_rssi: List[Dict[str, Any]] = []

        for i in range(len(noisy_measurements)):
            # sigma_fov.calculate_fov_for_point ожидает список точек до текущей включительно
            # и использует ИХ данные (включая RSSI) для расчета FoV.
            # По ТЗ, "истинный FoV" должен считаться по чистым данным.
            # Значит, передаем clean_measurements (с чистым RSSI).
            sub_route_clean = clean_measurements[0: i + 1]  # Срез с чистыми RSSI

            fov_params = sigma_fov.calculate_fov_for_point(sub_route_clean, source_coords)
            # fov_params: {"fov_dir_deg": float, "fov_width_deg": float}

            fov_dir_rad = math.radians(fov_params["fov_dir_deg"])
            fov_info_for_csv = {
                "fov_dir_sin": math.sin(fov_dir_rad),
                "fov_dir_cos": math.cos(fov_dir_rad),
                "fov_width_deg": fov_params["fov_width_deg"]
            }
            session_fov_data.append(fov_info_for_csv)

            # Собираем данные для DataPreprocessor
            # Важно: 'rssi' здесь должен быть зашумленным для входа в модель
            # 'clean_rssi' - это таргет для предобучения
            point_data_for_preproc = noisy_measurements[i].copy()  # содержит noisy 'rssi'
            point_data_for_preproc.update(fov_info_for_csv)
            point_data_for_preproc['clean_rssi'] = clean_rssi_values[i]
            measurements_with_fov_and_clean_rssi.append(point_data_for_preproc)

        # logger.debug(f"Session {session_id} (idx {session_idx}): Calculated FoV for {len(session_fov_data)} points. "
        #              f"First FoV: {session_fov_data[0] if session_fov_data else 'N/A'}")

        return measurements_with_fov_and_clean_rssi, source_coords, session_fov_data, session_id

    def get_raw_data_for_csv(self, index: int) -> Dict[str, Any]:
        """
        Генерирует и возвращает данные одной сессии в формате, подходящем для CSV.
        """
        # measurements_data: [{'latitude', 'longitude', 'rssi' (noisy), 'fov_...', 'clean_rssi'}, ...]
        measurements_data, source_coords, fov_data_csv, session_id = self._generate_single_session_data(index)

        # Для CSV нам нужны оригинальные (не нормализованные, не эмбеддинги) lon, lat, noisy_rssi
        # fov_dir_sin, fov_dir_cos, fov_width_deg, source_lon, source_lat

        # measurements_data уже содержит все, что нужно для CSV, кроме session_id и source_coords, которые есть отдельно
        # 'rssi' в measurements_data уже зашумленный.

        # Собираем в требуемую структуру для CSVSaver
        csv_points = []
        for m_point in measurements_data:
            csv_points.append({
                "latitude": m_point["latitude"],
                "longitude": m_point["longitude"],
                "rssi": m_point["rssi"]  # Это уже зашумленный RSSI
            })

        return {
            "session_id": session_id,
            "measurements": csv_points,  # [{lat, lon, noisy_rssi}, ...]
            "source_coords": source_coords,
            "fov_data": fov_data_csv  # [{fov_dir_sin, fov_dir_cos, fov_width_deg}, ...]
        }

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> torch_geometric.data.Data:
        # measurements_data: [{'latitude', 'longitude', 'rssi' (noisy), 'fov_...', 'clean_rssi'}, ...]
        measurements_data, source_coords, _, session_id = self._generate_single_session_data(index)

        if not measurements_data:  # Если генерация не удалась
            logger.error(f"Failed to generate data for sample {index}. Returning empty graph.")
            # Возвращаем пустой, но валидный Data объект
            num_features = 1 + 2 * self.data_prep_config.coordinate_embedding_dim
            return torch_geometric.data.Data(
                x=torch.empty((0, num_features), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=0,
                session_id="error_empty_session",
                y_clean_rssi=torch.empty((0, 1), dtype=torch.float32),
                y_fov=torch.empty((0, 3), dtype=torch.float32),  # sin, cos, width
                source_coords=torch.empty((1, 2), dtype=torch.float32)
            )

        # 4. Предобработка данных (нормализация RSSI, эмбеддинги координат)
        # DataPreprocessor ожидает список словарей, где каждый словарь - точка измерения.
        # Он извлечет 'rssi', 'latitude', 'longitude' для обработки.
        # Остальные поля ('fov_...', 'clean_rssi') будут сохранены как есть.
        processed_measurements = self.preprocessor.preprocess_session_data(measurements_data)
        # processed_measurements теперь содержит:
        # 'rssi_norm' (из зашумленного), 'latitude_emb', 'longitude_emb'
        # 'original_latitude', 'original_longitude', 'original_rssi' (зашумленный до нормализации)
        # а также 'fov_dir_sin', 'fov_dir_cos', 'fov_width_deg', 'clean_rssi' (чистый, ненормализованный)

        # 5. Формирование графа
        # Извлекаем таргеты (чистый RSSI и FoV) перед передачей в CoderGraph,
        # так как CoderGraph формирует только 'x'.
        # Таргет для предобучения: чистый RSSI (нормализованный)
        # Правило нормализации для 'clean_rssi' должно быть таким же, как для 'rssi'.
        y_clean_rssi_list = []
        for p_meas in processed_measurements:
            # Нормализуем чистый RSSI так же, как и входной RSSI
            # DataPreprocessor не обрабатывает 'clean_rssi' автоматически.
            # Мы можем это сделать здесь или добавить правило в DataPreprocessor.
            # Проще здесь:
            normalized_clean_rssi = self.preprocessor.feature_rules['rssi'](p_meas['clean_rssi'])
            y_clean_rssi_list.append(normalized_clean_rssi)

        y_clean_rssi_tensor = torch.tensor(y_clean_rssi_list, dtype=torch.float32).unsqueeze(1)  # (num_nodes, 1)

        # Таргет для основного обучения: FoV (sin, cos, width_deg)
        y_fov_list = [[pm['fov_dir_sin'], pm['fov_dir_cos'], pm['fov_width_deg']] for pm in processed_measurements]
        y_fov_tensor = torch.tensor(y_fov_list, dtype=torch.float32)  # (num_nodes, 3)

        # Координаты источника
        source_coords_tensor = torch.tensor(
            [source_coords['latitude'], source_coords['longitude']], dtype=torch.float32
        ).unsqueeze(0)  # (1, 2)

        # CoderGraph собирает 'x' из 'rssi_norm', 'latitude_emb', 'longitude_emb'
        # и добавляет 'original_coords', 'original_rssi' (зашумленный, ненормализованный)
        graph_data = self.coder_graph.create_graph_data(processed_measurements)

        # Добавляем таргеты и ID сессии в объект Data
        graph_data.y_clean_rssi = y_clean_rssi_tensor
        graph_data.y_fov = y_fov_tensor
        graph_data.source_coords = source_coords_tensor  # Координаты источника для этой сессии
        graph_data.session_id = session_id

        # logger.debug(f"Generated graph for session {session_id} (idx {index}): "
        #              f"Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}. "
        #              f"X shape: {graph_data.x.shape}, "
        #              f"Y_clean_rssi shape: {graph_data.y_clean_rssi.shape}, "
        #              f"Y_fov shape: {graph_data.y_fov.shape}")

        return graph_data


def OculuzDataLoader(dataset: OculuzDataset, batch_size: int, shuffle: bool = True, **kwargs) -> DataLoader:
    # PyTorch Geometric DataLoader
    from torch_geometric.loader import DataLoader as PyGDataLoader
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)





# **Примечание по `OculuzDataset` и `CSVSaver`**:
# Я добавил метод `get_raw_data_for_csv` в `OculuzDataset`, который генерирует данные сессии в формате, более удобном для сохранения в CSV,
# не прогоняя их через полное преобразование в граф. `CSVSaver` теперь использует этот метод.
# Также важно, что `rssi` в `measurements_data` (и, следовательно, в `processed_measurements`, и в `graph_data.x`) - это **зашумленный** RSSI.
# `clean_rssi` сохраняется отдельно и используется для формирования `graph_data.y_clean_rssi` (таргет для предобучения).