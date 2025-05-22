# oculuz/src/data/dataset/dataset_orchestrator.py
import math
import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import torch_geometric  # Импортируем весь модуль
from src.data.dataset.oculuz_dataset import OculuzDataset
from configuration.config_loader import ConfigLoader  # Используем новый ConfigLoader

logger = logging.getLogger(__name__)


class DatasetOrchestrator:
    def __init__(
            self,
            total_dataset_size: int,
            route_type_distribution: Dict[str, float],
            config_paths_dict: Dict[str, Any],
            # Пример config_paths_dict:
            # {
            #     "route_generators": { # Специфичные конфиги для каждого типа маршрута
            #         "direct": "configuration/route_generators/direct_route_config.yaml",
            #         "circle": "configuration/route_generators/circle_route_config.yaml",
            #     },
            #     "common_route_config": "configuration/route_generators/common_config.yaml", # Общий для всех маршрутов
            #     "data_preprocessing_config": "configuration/data_preprocessing_config.yaml",
            #     "graph_config": "configuration/graph_config.yaml",
            #     "noise_common_config": "configuration/noise_generators/common_noise_config.yaml",
            #     # Опционально: пути к специфичным конфигам шума, если нужны не дефолтные
            #     "specific_noise_configs": {
            #         "gaussian": "configuration/noise_generators/gaussian_noise_config.yaml",
            #         # "poisson": "..."
            #     }
            # }
    ):
        self.total_dataset_size = total_dataset_size
        self.route_type_distribution = route_type_distribution
        self.config_paths_dict = config_paths_dict

        if not math.isclose(sum(self.route_type_distribution.values()), 1.0):
            logger.warning(
                f"Sum of route type proportions is {sum(self.route_type_distribution.values())}, not 1.0. "
                f"Counts will be normalized."
            )
        logger.info(f"DatasetOrchestrator initialized. Target size: {total_dataset_size}, "
                    f"Distribution: {route_type_distribution}")

    def _calculate_target_counts(self) -> Dict[str, int]:
        target_counts: Dict[str, int] = {}
        initial_proportional_counts: Dict[str, float] = {}
        current_sum_proportions = sum(self.route_type_distribution.values())
        if current_sum_proportions == 0:
            logger.error("Sum of route type proportions is zero. Cannot distribute.")
            return {rt: 0 for rt in self.route_type_distribution}

        for rt, prop in self.route_type_distribution.items():
            normalized_prop = prop / current_sum_proportions
            initial_proportional_counts[rt] = normalized_prop * self.total_dataset_size

        for rt, count in initial_proportional_counts.items():
            target_counts[rt] = math.floor(count)

        remainders = sorted(
            [(rt, initial_proportional_counts[rt] - target_counts[rt]) for rt in target_counts],
            key=lambda x: x[1],
            reverse=True
        )
        current_integer_sum = sum(target_counts.values())
        num_to_distribute = self.total_dataset_size - current_integer_sum

        for i in range(abs(int(num_to_distribute))):
            if not remainders: break
            route_type_to_adjust = remainders[i % len(remainders)][0]
            target_counts[route_type_to_adjust] += 1 if num_to_distribute > 0 else -1
            target_counts[route_type_to_adjust] = max(0, target_counts[route_type_to_adjust])

        final_sum = sum(target_counts.values())
        if final_sum != self.total_dataset_size and remainders:  # Простая корректировка
            discrepancy = self.total_dataset_size - final_sum
            target_counts[remainders[0][0]] += discrepancy
            target_counts[remainders[0][0]] = max(0, target_counts[remainders[0][0]])
            logger.warning(f"Adjusted counts due to discrepancy. New sum: {sum(target_counts.values())}")

        logger.info(f"Target route counts for dataset generation: {target_counts}")
        return target_counts

    def create_dataset(self) -> Tuple[List[torch_geometric.data.Data], List[Dict[str, Any]]]:
        target_counts = self._calculate_target_counts()
        actual_counts: Dict[str, int] = {rt: 0 for rt in target_counts}
        total_generated_count = 0

        all_graph_data_samples: List[torch_geometric.data.Data] = []
        all_raw_data_for_csv: List[Dict[str, Any]] = []

        # Загрузка общих конфигов один раз
        try:
            data_prep_cfg_loader = ConfigLoader(self.config_paths_dict["data_preprocessing_config"])
            graph_cfg_loader = ConfigLoader(self.config_paths_dict["graph_config"])
            noise_common_cfg_loader = ConfigLoader(self.config_paths_dict["noise_common_config"])
            common_route_cfg_path = self.config_paths_dict["common_route_config"]  # Путь, OculuzDataset загрузит
        except KeyError as e:
            logger.error(f"Missing a common config path in config_paths_dict: {e}", exc_info=True)
            return [], []

        specific_noise_cfg_sources: Dict[str, Union[ConfigLoader, str]] = {}
        if "specific_noise_configs" in self.config_paths_dict:
            for noise_type, path in self.config_paths_dict["specific_noise_configs"].items():
                specific_noise_cfg_sources[noise_type] = ConfigLoader(path)  # Загружаем сразу

        for route_type, num_samples_for_type in target_counts.items():
            if num_samples_for_type <= 0:
                logger.info(f"Skipping route type {route_type} as 0 samples are targeted.")
                actual_counts[route_type] = 0
                continue

            logger.info(f"Generating {num_samples_for_type} samples for route type: {route_type}...")

            specific_route_config_path = self.config_paths_dict.get("route_generators", {}).get(route_type)
            if not specific_route_config_path:
                logger.error(
                    f"Specific configuration path for route type '{route_type}' not found in 'route_generators'. Skipping.")
                continue

            # OculuzDataset ожидает список спецификаций, но для одной генерации мы передаем только один тип
            route_gen_spec_for_temp_dataset = [{
                "type": route_type,
                "weight": 1.0,  # Вес не важен, т.к. только один генератор
                "config_path": specific_route_config_path  # OculuzDataset загрузит этот специфичный конфиг
            }]

            try:
                # Передаем уже загруженные ConfigLoader объекты для общих конфигов
                temp_dataset = OculuzDataset(
                    dataset_size=num_samples_for_type,  # Генерируем нужное количество для этого типа
                    route_generator_specs=route_gen_spec_for_temp_dataset,
                    data_prep_config_source=data_prep_cfg_loader,
                    graph_config_source=graph_cfg_loader,
                    noise_common_config_source=noise_common_cfg_loader,
                    specific_noise_configs_sources=specific_noise_cfg_sources,  # Передаем загруженные
                    default_common_route_config_path=common_route_cfg_path  # Путь к общему конфигу маршрутов
                )
            except Exception as e:
                logger.error(f"Failed to initialize OculuzDataset for route type {route_type}: {e}", exc_info=True)
                continue

            current_type_graph_samples = []
            current_type_raw_samples = []
            for i in range(num_samples_for_type):  # Используем num_samples_for_type, а не temp_dataset.dataset_size
                try:
                    graph_sample, raw_sample_for_csv = temp_dataset.generate_one_sample_fully(i)
                    if graph_sample and raw_sample_for_csv:
                        current_type_graph_samples.append(graph_sample)
                        current_type_raw_samples.append(raw_sample_for_csv)
                    else:
                        logger.warning(f"Failed to generate sample {i + 1}/{num_samples_for_type} "
                                       f"for route type {route_type}. Skipping.")
                except Exception as e:
                    logger.error(f"Error generating sample {i + 1}/{num_samples_for_type} "
                                 f"for route type {route_type}: {e}", exc_info=True)
                    continue

            all_graph_data_samples.extend(current_type_graph_samples)
            all_raw_data_for_csv.extend(current_type_raw_samples)
            actual_counts[route_type] = len(current_type_graph_samples)
            total_generated_count += len(current_type_graph_samples)
            logger.info(f"Finished generating for {route_type}. "
                        f"Actual samples: {len(current_type_graph_samples)}.")

        logger.info(f"--- Dataset Generation Summary ---")
        logger.info(f"Target total samples: {self.total_dataset_size}")
        logger.info(f"Actual total samples generated: {total_generated_count}")
        logger.info(f"Target distribution: {self.route_type_distribution}")
        logger.info(f"Target counts per type: {target_counts}")
        logger.info(f"Actual counts per type generated: {actual_counts}")
        logger.info(f"------------------------------------")

        if not all_graph_data_samples:
            logger.warning("No samples were generated for any route type.")
            return [], []

        combined = list(zip(all_graph_data_samples, all_raw_data_for_csv))
        random.shuffle(combined)
        shuffled_graphs, shuffled_raws = zip(*combined) if combined else ([], [])

        logger.info(f"Total {len(shuffled_graphs)} samples collected and shuffled.")
        return list(shuffled_graphs), list(shuffled_raws)