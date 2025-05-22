# oculuz/src/data/dataset/oculuz_dataset.py
import torch
import torch_geometric
from torch.utils.data import Dataset  # Не DataLoader, он импортируется ниже как PyGDataLoader
import random
import math
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional, Type, Union

# Используем ConfigLoader для всех конфигураций
from configuration.config_loader import ConfigLoader
from src.data.preprocessing import DataPreprocessor, CoderGraph
from src.data.dataset.route_generators import (
    BaseRouteGenerator, DirectRouteGenerator, CircleRouteGenerator, ArcRouteGenerator, RandomWalkRouteGenerator
)
from src.data.dataset.noise_generators import NoiseOrchestrator
import sigma_fov as sigma_fov

logger = logging.getLogger(__name__)


class OculuzDataset(Dataset):
    def __init__(
            self,
            dataset_size: int,
            route_generator_specs: List[Dict[str, Any]],
            data_prep_config_source: Optional[Union[ConfigLoader, str]] = None,
            graph_config_source: Optional[Union[ConfigLoader, str]] = None,
            noise_common_config_source: Optional[Union[ConfigLoader, str]] = None,
            # Если specific_noise_configs_sources переданы, они переопределят пути по умолчанию в NoiseOrchestrator
            specific_noise_configs_sources: Optional[Dict[str, Union[ConfigLoader, str]]] = None,
            # Пути по умолчанию, если источники не переданы
            default_data_prep_config_path: str = "configuration/data_preprocessing_config.yaml",
            default_graph_config_path: str = "configuration/graph_config.yaml",
            default_noise_common_config_path: str = "configuration/noise_generators/common_noise_config.yaml",
            default_common_route_config_path: str = "configuration/route_generators/common_config.yaml"
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.route_generator_specs = route_generator_specs
        self.default_common_route_config_path = default_common_route_config_path

        # Загрузка конфигураций
        self.data_prep_config: ConfigLoader
        if isinstance(data_prep_config_source, ConfigLoader):
            self.data_prep_config = data_prep_config_source
        elif isinstance(data_prep_config_source, str):
            self.data_prep_config = ConfigLoader(data_prep_config_source)
        else:
            self.data_prep_config = ConfigLoader(default_data_prep_config_path)

        self.graph_config: ConfigLoader
        if isinstance(graph_config_source, ConfigLoader):
            self.graph_config = graph_config_source
        elif isinstance(graph_config_source, str):
            self.graph_config = ConfigLoader(graph_config_source)
        else:
            self.graph_config = ConfigLoader(default_graph_config_path)

        noise_common_cfg: ConfigLoader
        if isinstance(noise_common_config_source, ConfigLoader):
            noise_common_cfg = noise_common_config_source
        elif isinstance(noise_common_config_source, str):
            noise_common_cfg = ConfigLoader(noise_common_config_source)
        else:
            noise_common_cfg = ConfigLoader(default_noise_common_config_path)

        self.preprocessor = DataPreprocessor(
            config_loader=self.data_prep_config)  # DataPreprocessor ожидает ConfigLoader
        self.coder_graph = CoderGraph(graph_config_loader=self.graph_config,
                                      data_prep_config_loader=self.data_prep_config)  # Аналогично

        self._initialize_route_generators()

        # Инициализация шума
        # NoiseOrchestrator был обновлен для приема ConfigLoader или путей
        g_conf_src = specific_noise_configs_sources.get("gaussian") if specific_noise_configs_sources else None
        p_conf_src = specific_noise_configs_sources.get("poisson") if specific_noise_configs_sources else None
        # c_conf_src = specific_noise_configs_sources.get("custom") if specific_noise_configs_sources else None # Если будет кастомный

        self.noise_orchestrator = NoiseOrchestrator(
            common_noise_config_source=noise_common_cfg,
            gaussian_config_source=g_conf_src,
            poisson_config_source=p_conf_src,
            # Пути по умолчанию для специфичных конфигов шума уже есть в NoiseOrchestrator
        )
        enabled_noise_log = []
        try:
            enabled_noise_log = self.noise_orchestrator.common_config["enabled_noise_types"]
        except:
            pass

        logger.info(f"OculuzDataset initialized. Size: {self.dataset_size}, "
                    f"Num generator types: {len(self.route_generators)}. "
                    f"Noise enabled: {enabled_noise_log}")

    def _initialize_route_generators(self):
        self.route_generators: List[BaseRouteGenerator] = []
        self.route_generator_weights: List[float] = []

        gen_map: Dict[str, Type[BaseRouteGenerator]] = {
            "direct": DirectRouteGenerator,
            "circle": CircleRouteGenerator,
            "arc": ArcRouteGenerator,
            "random_walk": RandomWalkRouteGenerator,
        }

        # Загружаем общий конфиг для маршрутов один раз
        common_route_cfg_loader = ConfigLoader(self.default_common_route_config_path)

        for spec in self.route_generator_specs:
            gen_type = spec["type"]
            weight = spec.get("weight", 1.0)
            # Путь к специфичному конфигу для этого типа генератора
            specific_config_path = spec.get("config_path")

            if gen_type not in gen_map:
                logger.error(f"Unknown route generator type: {gen_type}")
                raise ValueError(f"Unknown route generator type: {gen_type}")

            if not specific_config_path:
                logger.error(f"Specific config_path not provided for route generator type: {gen_type}")
                raise ValueError(f"Specific config_path missing for route generator type: {gen_type}")

            generator_class = gen_map[gen_type]
            specific_route_cfg_loader = ConfigLoader(specific_config_path)

            try:
                generator_instance = generator_class(
                    common_config=common_route_cfg_loader,
                    specific_config=specific_route_cfg_loader
                )
                self.route_generators.append(generator_instance)
                self.route_generator_weights.append(weight)
                logger.info(f"Initialized route generator: {gen_type} with weight {weight} "
                            f"(common_cfg: '{self.default_common_route_config_path}', specific_cfg: '{specific_config_path}')")
            except Exception as e:
                logger.error(f"Failed to initialize route generator {gen_type} with config {specific_config_path}: {e}",
                             exc_info=True)
                raise

        if not self.route_generators:
            logger.error("No route generators were initialized. Check route_generator_specs.")
            raise ValueError("No route generators specified or initialized.")

    def _generate_single_session_data(self, session_idx: int) -> Tuple[
        List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]], str]:
        session_id = str(uuid.uuid4())
        generator = random.choices(self.route_generators, weights=self.route_generator_weights, k=1)[0]
        clean_measurements, source_coords = generator.generate_route()

        if not clean_measurements:
            logger.warning(f"Session {session_id} (idx {session_idx}): Route generation failed. Returning empty data.")
            return [], {"latitude": 0.0, "longitude": 0.0}, [], session_id

        clean_rssi_values = [m['rssi'] for m in clean_measurements]
        noisy_measurements = self.noise_orchestrator.apply_all_enabled_noise([m.copy() for m in clean_measurements])

        session_fov_data: List[Dict[str, Any]] = []
        measurements_with_fov_and_clean_rssi: List[Dict[str, Any]] = []

        for i in range(len(noisy_measurements)):
            sub_route_clean = clean_measurements[0: i + 1]
            # Предполагаем, что sigma_fov.calculate_fov_for_point ожидает данные с чистым RSSI
            # и использует их. Если ему нужны зашумленные, нужно передавать noisy_measurements.
            # По ТЗ "истинный FoV" считается по чистым данным.
            fov_params = sigma_fov.calculate_fov_for_point(sub_route_clean, source_coords)
            fov_dir_rad = math.radians(fov_params["fov_dir_deg"])
            fov_info_for_csv = {
                "fov_dir_sin": math.sin(fov_dir_rad),
                "fov_dir_cos": math.cos(fov_dir_rad),
                "fov_width_deg": fov_params["fov_width_deg"]
            }
            session_fov_data.append(fov_info_for_csv)

            point_data_for_preproc = noisy_measurements[i].copy()
            point_data_for_preproc.update(fov_info_for_csv)
            point_data_for_preproc['clean_rssi'] = clean_rssi_values[i]
            measurements_with_fov_and_clean_rssi.append(point_data_for_preproc)

        return measurements_with_fov_and_clean_rssi, source_coords, session_fov_data, session_id

    def get_raw_data_for_csv(self, index: int) -> Dict[str, Any]:
        measurements_data, source_coords, fov_data_csv, session_id = self._generate_single_session_data(index)
        csv_points = []
        for m_point in measurements_data:
            csv_points.append({
                "latitude": m_point["latitude"],
                "longitude": m_point["longitude"],
                "rssi": m_point["rssi"]
            })
        return {
            "session_id": session_id,
            "measurements": csv_points,
            "source_coords": source_coords,
            "fov_data": fov_data_csv
        }

    def __len__(self) -> int:
        return self.dataset_size

    def generate_one_sample_fully(self, index: int) -> Tuple[
        Optional[torch_geometric.data.Data], Optional[Dict[str, Any]]]:
        """
        Генерирует один полный сэмпл: и граф, и сырые данные для CSV.
        Используется Orchestrator'ом.
        """
        measurements_data, source_coords, fov_data_for_csv, session_id = self._generate_single_session_data(index)

        if not measurements_data:
            logger.error(f"Failed to generate base data for sample {index}. Cannot create graph or raw CSV data.")
            return None, None

        # Собираем raw_data для CSV
        csv_points = []
        for m_point in measurements_data:  # measurements_data уже содержит зашумленный RSSI
            csv_points.append({
                "latitude": m_point["latitude"],
                "longitude": m_point["longitude"],
                "rssi": m_point["rssi"]
            })
        raw_data_for_csv = {
            "session_id": session_id,
            "measurements": csv_points,
            "source_coords": source_coords,
            "fov_data": fov_data_for_csv
        }

        # Продолжаем с созданием графа
        processed_measurements = self.preprocessor.preprocess_session_data(measurements_data)

        y_clean_rssi_list = []
        for p_meas in processed_measurements:
            # Нормализация чистого RSSI должна использовать то же правило, что и для входного RSSI.
            # DataPreprocessor.feature_rules['rssi'] теперь должен быть методом, а не словарем функций.
            # Либо получаем правило из preprocessor.
            # Предположим, что preprocessor.get_feature_rule('rssi') возвращает функцию нормализации.
            # Или, если preprocessor.normalize_feature(value, feature_name) есть:
            # normalized_clean_rssi = self.preprocessor.normalize_feature(p_meas['clean_rssi'], 'rssi')
            # В DataPreprocessor config ожидает Dict[str, Any], а не ConfigLoader. Надо это исправить в Preprocessor.
            # Пока что предположим, что self.preprocessor.rssi_normalizer - это нужная функция.
            # Это требует доработки в DataPreprocessor, чтобы он создавал/хранил нормализаторы.
            # Допустим, preprocessor.normalize_rssi(value) существует
            try:
                # Это примерная логика, DataPreprocessor должен предоставлять метод для нормализации отдельного значения
                # по правилу для 'rssi'.
                # Сейчас DataPreprocessor.feature_rules - это словарь функций.
                # Должно быть self.data_prep_config["feature_rules"]["rssi"] и оттуда параметры.
                # А сам DataPreprocessor должен иметь методы типа:
                # preprocessor.normalize_rssi(value)
                # preprocessor.embed_coordinates(lat, lon)
                # Для простоты, пока оставим как было, но это точка для улучшения.
                # В текущей реализации DataPreprocessor, он ожидает config: Dict.
                # А мы передаем ConfigLoader. Это несовместимо.
                # DataPreprocessor должен быть адаптирован под ConfigLoader.
                min_rssi = self.data_prep_config["rssi"]["min_val"]
                max_rssi = self.data_prep_config["rssi"]["max_val"]
                normalized_clean_rssi = (p_meas['clean_rssi'] - min_rssi) / (max_rssi - min_rssi)
                normalized_clean_rssi = max(0.0, min(1.0, normalized_clean_rssi))  # Клиппинг
                y_clean_rssi_list.append(normalized_clean_rssi)

            except KeyError as e:
                logger.error(
                    f"KeyError during clean_rssi normalization for session {session_id}: {e}. Check data_prep_config structure.")
                y_clean_rssi_list.append(0.0)  # Заглушка
            except Exception as e:
                logger.error(f"Error during clean_rssi normalization for session {session_id}: {e}", exc_info=True)
                y_clean_rssi_list.append(0.0)

        y_clean_rssi_tensor = torch.tensor(y_clean_rssi_list, dtype=torch.float32).unsqueeze(1)
        y_fov_list = [[pm['fov_dir_sin'], pm['fov_dir_cos'], pm['fov_width_deg']] for pm in processed_measurements]
        y_fov_tensor = torch.tensor(y_fov_list, dtype=torch.float32)
        source_coords_tensor = torch.tensor(
            [source_coords['latitude'], source_coords['longitude']], dtype=torch.float32
        ).unsqueeze(0)

        graph_data = self.coder_graph.create_graph_data(processed_measurements)
        graph_data.y_clean_rssi = y_clean_rssi_tensor
        graph_data.y_fov = y_fov_tensor
        graph_data.source_coords = source_coords_tensor
        graph_data.session_id = session_id

        return graph_data, raw_data_for_csv

    def __getitem__(self, index: int) -> torch_geometric.data.Data:
        graph_data, _ = self.generate_one_sample_fully(index)
        if graph_data is None:  # Если генерация не удалась
            logger.error(f"Failed to generate graph data for sample {index}. Returning empty graph.")
            # Возвращаем пустой, но валидный Data объект
            try:
                coord_emb_dim = self.data_prep_config["coordinate"]["embedding_dim"]
            except KeyError:
                coord_emb_dim = 64  # Default fallback
            num_features = 1 + 2 * coord_emb_dim  # rssi_norm + 2 * coord_emb
            return torch_geometric.data.Data(
                x=torch.empty((0, num_features), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=0,
                session_id="error_empty_session",
                y_clean_rssi=torch.empty((0, 1), dtype=torch.float32),
                y_fov=torch.empty((0, 3), dtype=torch.float32),
                source_coords=torch.empty((1, 2), dtype=torch.float32)
            )
        return graph_data


def OculuzDataLoader(dataset: OculuzDataset, batch_size: int, shuffle: bool = True,
                     **kwargs) -> torch_geometric.loader.DataLoader:
    from torch_geometric.loader import DataLoader as PyGDataLoader
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)