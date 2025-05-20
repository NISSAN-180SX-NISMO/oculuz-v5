# oculuz/src/data/dataset/dataset_orchestrator.py
import math
import random
import logging
from typing import List, Dict, Any, Tuple, Optional

import torch_geometric
from src.data.dataset.oculuz_dataset import OculuzDataset
# Assuming config loaders are accessible, adjust import if necessary
from configuration.config_loader import (
    DataPreprocessingConfig, GraphConfig, CommonNoiseConfig
)

logger = logging.getLogger(__name__)


class DatasetOrchestrator:
    def __init__(
            self,
            total_dataset_size: int,
            route_type_distribution: Dict[str, float],
            config_paths: Dict[str, Any],
            # Example config_paths:
            # {
            #     "route_configs": {
            #         "direct": "configuration/route_generators/direct_route_config.yaml",
            #         "circle": "configuration/route_generators/circle_route_config.yaml",
            #         # ...
            #     },
            #     "common_configs": {
            #         "data_prep": "configuration/data_preprocessing_config.yaml",
            #         "graph": "configuration/graph_config.yaml",
            #         "noise_common": "configuration/noise_generators/common_noise_config.yaml",
            #         # Optional: specific noise config paths if not relying on OculuzDataset defaults
            #         # "gaussian_noise": "...",
            #         # "poisson_noise": "...",
            #     }
            # }
    ):
        self.total_dataset_size = total_dataset_size
        self.route_type_distribution = route_type_distribution
        self.config_paths = config_paths

        if not math.isclose(sum(self.route_type_distribution.values()), 1.0):
            logger.warning(
                f"Sum of route type proportions is {sum(self.route_type_distribution.values())}, not 1.0. "
                f"Counts will be normalized based on these proportions for the total size."
            )
        logger.info(f"DatasetOrchestrator initialized. Target size: {total_dataset_size}, "
                    f"Distribution: {route_type_distribution}")

    def _calculate_target_counts(self) -> Dict[str, int]:
        """Calculates the number of samples for each route type using Largest Remainder Method."""
        target_counts: Dict[str, int] = {}
        initial_proportional_counts: Dict[str, float] = {}

        # Normalize proportions if they don't sum to 1
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
            reverse=True  # Sort by remainder descending
        )

        current_integer_sum = sum(target_counts.values())
        num_to_distribute = self.total_dataset_size - current_integer_sum

        if num_to_distribute < 0:
            logger.warning(
                f"Initial floor sum {current_integer_sum} is greater than total size {self.total_dataset_size}. "
                f"Adjusting counts downwards. This might happen with very small proportions/total_size.")
        # Distribute positive or negative difference
        for i in range(abs(int(num_to_distribute))):
            if not remainders: break  # Should not happen if route_type_distribution is not empty
            route_type_to_adjust = remainders[i % len(remainders)][0]
            target_counts[route_type_to_adjust] += 1 if num_to_distribute > 0 else -1
            target_counts[route_type_to_adjust] = max(0, target_counts[route_type_to_adjust])  # Ensure not negative

        # Final check for sum
        final_sum = sum(target_counts.values())
        if final_sum != self.total_dataset_size:
            logger.warning(
                f"Final calculated sum of counts {final_sum} does not match total_dataset_size {self.total_dataset_size}. "
                f"Manual adjustment might be needed or check distribution logic for edge cases.")
            # Basic fix: adjust the first type or largest type if discrepancy is small
            if final_sum < self.total_dataset_size and remainders:
                target_counts[remainders[0][0]] += (self.total_dataset_size - final_sum)
            elif final_sum > self.total_dataset_size and remainders:
                target_counts[remainders[0][0]] -= (final_sum - self.total_dataset_size)

        logger.info(f"Target route counts for dataset generation: {target_counts}")
        return target_counts

    def create_dataset(self) -> Tuple[List[torch_geometric.data.Data], List[Dict[str, Any]]]:
        """
        Generates the dataset with the specified distribution of route types.

        Returns:
            A tuple containing:
                - List[torch_geometric.data.Data]: Shuffled list of generated graph samples.
                - List[Dict[str, Any]]: Corresponding list of raw data dictionaries for CSV saving.
        """
        target_counts = self._calculate_target_counts()
        actual_counts: Dict[str, int] = {rt: 0 for rt in target_counts}
        total_generated_count = 0

        all_graph_data_samples: List[torch_geometric.data.Data] = []
        all_raw_data_for_csv: List[Dict[str, Any]] = []

        # Load common configs once
        # These will be passed as objects to OculuzDataset to avoid repeated file loading
        common_cfg_paths = self.config_paths.get("common_configs", {})
        data_prep_config = DataPreprocessingConfig.load(common_cfg_paths["data_prep"])
        graph_config = GraphConfig.load(common_cfg_paths["graph"])
        noise_common_config = CommonNoiseConfig.load(common_cfg_paths["noise_common"])
        # Specific noise configs (Gaussian, Poisson) will be loaded by OculuzDataset
        # based on noise_common_config.enabled_noise_types and their default paths,
        # unless overridden by passing noise_configs dict to OculuzDataset.

        for route_type, num_samples_for_type in target_counts.items():
            if num_samples_for_type <= 0:
                logger.info(f"Skipping route type {route_type} as 0 samples are targeted.")
                actual_counts[route_type] = 0
                continue

            logger.info(f"Generating {num_samples_for_type} samples for route type: {route_type}...")

            route_config_path = self.config_paths.get("route_configs", {}).get(route_type)
            if not route_config_path:
                logger.error(f"Configuration path for route type '{route_type}' not found. Skipping.")
                continue

            specific_route_gen_spec = [{
                "type": route_type,
                "weight": 1.0,
                "config_path": route_config_path
            }]

            # Instantiate OculuzDataset configured for this specific route type
            try:
                temp_dataset = OculuzDataset(
                    dataset_size=num_samples_for_type,
                    route_generator_specs=specific_route_gen_spec,
                    data_prep_config=data_prep_config,  # Pass loaded object
                    graph_config=graph_config,  # Pass loaded object
                    noise_common_config=noise_common_config,  # Pass loaded object
                    # noise_configs can be passed here if specific noise configs need to be overridden
                    # from paths different than their defaults. For now, rely on OculuzDataset's internal loading.
                )
            except Exception as e:
                logger.error(f"Failed to initialize OculuzDataset for route type {route_type}: {e}", exc_info=True)
                continue

            current_type_graph_samples = []
            current_type_raw_samples = []
            for i in range(num_samples_for_type):
                try:
                    # Use the new method to get both graph and raw data consistently
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

        # Shuffle the collected samples, ensuring graph and raw data remain paired
        combined = list(zip(all_graph_data_samples, all_raw_data_for_csv))
        random.shuffle(combined)
        shuffled_graphs, shuffled_raws = zip(*combined)

        logger.info(f"Total {len(shuffled_graphs)} samples collected and shuffled.")

        return list(shuffled_graphs), list(shuffled_raws)