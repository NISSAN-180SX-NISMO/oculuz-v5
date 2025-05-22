# oculuz/check_dataset.py
import logging
import os
import sys
from typing import List, Dict, Any

import pandas as pd
from datetime import datetime

project_root_script = os.path.abspath(os.path.dirname(__file__))
if project_root_script not in sys.path:
    # Если check_dataset.py находится в корне проекта oculuz,
    # то project_root_script это /path/to/oculuz
    # импорты типа from src... будут работать
    sys.path.insert(0, project_root_script)

# DatasetOrchestrator уже адаптирован для работы с ConfigLoader (он сам их загружает по путям)
from src.data.dataset.dataset_orchestrator import DatasetOrchestrator
# CSVSaver был изменен для приема raw_data_list
from src.data.dataset.csv_saver import CSVSaver


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("src.data.dataset.dataset_orchestrator").setLevel(logging.DEBUG)
    logging.getLogger("src.data.dataset.oculuz_dataset").setLevel(logging.INFO)
    logging.getLogger("src.data.dataset.route_generators.base_route_generator").setLevel(logging.INFO) # Уменьшаем избыточность логов отсюда


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Dataset Orchestration Test Script...")

    total_samples = 50
    route_distribution = {
        "direct": 0.25, # Убедимся, что используется такой же ключ, как в config_paths
        "circle": 0.25,
        "arc": 0.25,
        "random_walk": 0.25
    }

    # Пути к конфигурациям относительно корня проекта (где лежит этот скрипт)
    # Если скрипт в oculuz/, а конфиги в oculuz/configuration/
    config_base = "configuration"

    config_paths_dict = {
        "route_generators": {
            "direct": os.path.join(config_base, "route_generators/direct_route_config.yaml"),
            "circle": os.path.join(config_base, "route_generators/circle_route_config.yaml"),
            "arc": os.path.join(config_base, "route_generators/arc_route_config.yaml"),
            "random_walk": os.path.join(config_base, "route_generators/random_walk_route_config.yaml"),
        },
        "common_route_config": os.path.join(config_base, "route_generators/common_route_config.yaml"),
        "data_preprocessing_config": os.path.join(config_base, "data_preprocessing_config.yaml"),
        "graph_config": os.path.join(config_base, "graph_config.yaml"),
        "noise_common_config": os.path.join(config_base, "noise_generators/common_noise_config.yaml"),
        "specific_noise_configs": { # Опционально, если хотим переопределить пути по умолчанию
             "gaussian": os.path.join(config_base, "noise_generators/gaussian_noise_config.yaml"),
             "poisson": os.path.join(config_base, "noise_generators/poisson_noise_config.yaml"),
        }
    }

    # Проверка существования файлов (важно для отладки)
    paths_to_check = [
        config_paths_dict["common_route_config"],
        config_paths_dict["data_preprocessing_config"],
        config_paths_dict["graph_config"],
        config_paths_dict["noise_common_config"]
    ]
    for gen_type_path_map in config_paths_dict["route_generators"].values():
        paths_to_check.append(gen_type_path_map)
    if "specific_noise_configs" in config_paths_dict:
        for noise_type_path_map in config_paths_dict["specific_noise_configs"].values():
            paths_to_check.append(noise_type_path_map)

    all_paths_exist = True
    for p_to_check in paths_to_check:
        if not os.path.exists(p_to_check):
            logger.error(f"CRITICAL: Config file not found: {p_to_check}")
            all_paths_exist = False
    if not all_paths_exist:
        logger.error("One or more critical configuration files are missing. Exiting.")
        return


    logger.info("Instantiating DatasetOrchestrator...")
    try:
        orchestrator = DatasetOrchestrator(
            total_dataset_size=total_samples,
            route_type_distribution=route_distribution,
            config_paths_dict=config_paths_dict # Orchestrator сам загрузит конфиги по этим путям
        )
    except Exception as e:
        logger.error(f"Failed to instantiate DatasetOrchestrator: {e}", exc_info=True)
        return

    logger.info("Creating dataset using orchestrator...")
    graph_samples_list, raw_data_list_for_csv = orchestrator.create_dataset()

    logger.info(f"Dataset creation complete.")
    logger.info(f"Number of graph samples generated: {len(graph_samples_list)}")
    logger.info(f"Number of raw data entries for CSV: {len(raw_data_list_for_csv)}")

    if not graph_samples_list and not raw_data_list_for_csv:
        logger.warning("No data (graphs or raw for CSV) was generated. Ending test.")
        return
    elif not graph_samples_list:
        logger.warning("No graph samples were generated.")
    elif not raw_data_list_for_csv:
         logger.warning("No raw data for CSV was generated.")


    if graph_samples_list:
        logger.info("First graph sample details (example):")
        logger.info(f"  Session ID: {graph_samples_list[0].session_id if hasattr(graph_samples_list[0], 'session_id') else 'N/A'}")
        logger.info(f"  Num nodes: {graph_samples_list[0].num_nodes}")
        logger.info(f"  Num edges: {graph_samples_list[0].num_edges}")
        logger.info(f"  X shape: {graph_samples_list[0].x.shape}")


    # Сохраняем CSV с помощью обновленного CSVSaver
    if raw_data_list_for_csv:
        output_dir = "oculuz_orchestrated_datasets_test" # Будет создана в корне проекта oculuz
        # os.makedirs(output_dir, exist_ok=True) # CSVSaver теперь делает это сам
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename_prefix = f"orchestrated_dataset_{timestamp}" # CSVSaver добавит .csv

        logger.info(f"Saving raw data to CSV in directory: {output_dir} with prefix: {csv_filename_prefix}")
        try:
            saved_csv_path = CSVSaver.save_dataset_to_csv(
                raw_data_list=raw_data_list_for_csv,
                num_sessions_in_data=len(raw_data_list_for_csv), # Передаем количество уникальных сессий
                filepath_prefix=csv_filename_prefix,
                output_dir=output_dir
            )
            logger.info(f"Raw data saved to: {saved_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset to CSV: {e}", exc_info=True)
    else:
        logger.warning("No raw data available to save to CSV.")

    logger.info("Dataset Orchestration Test Script finished.")

if __name__ == "__main__":
    main()