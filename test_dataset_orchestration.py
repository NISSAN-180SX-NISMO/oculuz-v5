# oculuz/test_dataset_orchestration.py
import logging
import os
import sys
from typing import List, Dict, Any

import pandas as pd
from datetime import datetime

# Adjust Python path to include the oculuz project directory
# This allows imports like `from src...`
project_root = os.path.abspath(os.path.dirname(__file__))
# If 'oculuz' is the project root and this script is inside 'oculuz',
# then project_root is '.../oculuz'.
# We need the parent of 'oculuz' if 'oculuz' is meant to be a package itself,
# or ensure CWD is parent of 'oculuz'
# Assuming script is run from 'oculuz' directory or 'oculuz' is in PYTHONPATH.
# If oculuz is the current directory:
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Allows `from src...` if oculuz is the current folder
# If oculuz is a subfolder and we run from its parent:
# parent_dir = os.path.dirname(project_root)
# if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir) # Allows `from src...`

from src.data.dataset.dataset_orchestrator import DatasetOrchestrator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
            # You can also add a FileHandler here if needed
        ]
    )
    # Optionally, set lower levels for specific loggers
    logging.getLogger("src.data.dataset.dataset_orchestrator").setLevel(logging.DEBUG)
    logging.getLogger("src.data.dataset.oculuz_dataset").setLevel(logging.INFO)  # or DEBUG for more verbosity


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Dataset Orchestration Test Script...")

    # 1. Define Dataset Parameters
    total_samples = 50  # Small number for quick testing
    route_distribution = {
        "direct": 0.4,
        "circle": 0.3,
        "arc": 0.2,
        "random_walk": 0.1
    }
    # Ensure sum is 1.0 for clarity, though orchestrator normalizes
    if not sum(route_distribution.values()) == 1.0:
        logger.warning("Route distribution proportions do not sum to 1.0.")

    # 2. Define Paths to Configuration Files
    # All paths should be relative to the oculuz project root
    base_config_path = "oculuz/configuration"  # Adjusted to be relative from where script is likely run

    config_paths = {
        "route_configs": {
            "direct": os.path.join(base_config_path, "route_generators/direct_route_config.yaml"),
            "circle": os.path.join(base_config_path, "route_generators/circle_route_config.yaml"),
            "arc": os.path.join(base_config_path, "route_generators/arc_route_config.yaml"),
            "random_walk": os.path.join(base_config_path, "route_generators/random_walk_route_config.yaml"),
        },
        "common_configs": {
            "data_prep": os.path.join(base_config_path, "data_preprocessing_config.yaml"),
            "graph": os.path.join(base_config_path, "graph_config.yaml"),
            "noise_common": os.path.join(base_config_path, "noise_generators/common_noise_config.yaml"),
        }
    }

    # Verify paths (optional but good for debugging)
    for cat, paths in config_paths.items():
        for name, path in paths.items():
            if not os.path.exists(path):
                logger.error(f"Config file not found: {path} (for {cat} - {name})")
                # return # Exit if critical configs are missing

    # 3. Instantiate DatasetOrchestrator
    logger.info("Instantiating DatasetOrchestrator...")
    orchestrator = DatasetOrchestrator(
        total_dataset_size=total_samples,
        route_type_distribution=route_distribution,
        config_paths=config_paths
    )

    # 4. Create the Dataset
    logger.info("Creating dataset using orchestrator...")
    # This might take some time depending on total_samples
    graph_samples_list, raw_data_list_for_csv = orchestrator.create_dataset()

    # 5. Log Results
    logger.info(f"Dataset creation complete.")
    logger.info(f"Number of graph samples generated: {len(graph_samples_list)}")
    logger.info(f"Number of raw data entries for CSV: {len(raw_data_list_for_csv)}")

    if not graph_samples_list:
        logger.warning("No graph samples were generated. Ending test.")
        return
    if not raw_data_list_for_csv:
        logger.warning("No raw data for CSV was generated. Ending test.")
        return

    logger.info("First graph sample details (example):")
    logger.info(f"  Session ID: {graph_samples_list[0].session_id}")
    logger.info(f"  Num nodes: {graph_samples_list[0].num_nodes}")
    logger.info(f"  Num edges: {graph_samples_list[0].num_edges}")
    logger.info(f"  X shape: {graph_samples_list[0].x.shape}")

    # 6. Save a portion of the raw data to CSV (using pandas directly)
    output_dir = "oculuz_orchestrated_datasets_test"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"orchestrated_dataset_{timestamp}.csv"
    full_csv_path = os.path.join(output_dir, csv_filename)

    logger.info(f"Preparing to save raw data to CSV: {full_csv_path}")

    all_rows_for_df: List[Dict[str, Any]] = []
    # raw_data_list_for_csv is a list of session dictionaries
    for session_data in raw_data_list_for_csv:
        session_id = session_data["session_id"]
        source_lon = session_data["source_coords"]["longitude"]
        source_lat = session_data["source_coords"]["latitude"]

        # Each session_data["measurements"] is a list of points
        # Each session_data["fov_data"] is a list of fov info for those points
        for point_idx, meas_point in enumerate(session_data["measurements"]):
            if point_idx < len(session_data["fov_data"]):  # Ensure fov_data exists for the point
                fov_info = session_data["fov_data"][point_idx]
            else:  # Should not happen if data is consistent
                fov_info = {"fov_dir_sin": 0, "fov_dir_cos": 1, "fov_width_deg": 0}

            row = {
                "session_id": session_id,
                "longitude": meas_point["longitude"],
                "latitude": meas_point["latitude"],
                "rssi": meas_point["rssi"],
                "fov_dir_sin": fov_info["fov_dir_sin"],
                "fov_dir_cos": fov_info["fov_dir_cos"],
                "fov_width_deg": fov_info["fov_width_deg"],
                "source_longitude": source_lon,
                "source_latitude": source_lat,
            }
            all_rows_for_df.append(row)

    if all_rows_for_df:
        df = pd.DataFrame(all_rows_for_df)
        df.to_csv(full_csv_path, index=False, encoding='utf-8')
        logger.info(
            f"Successfully saved {len(df)} data points from {len(raw_data_list_for_csv)} sessions to {full_csv_path}")
    else:
        logger.warning("No data rows to save to CSV.")

    logger.info("Dataset Orchestration Test Script finished.")


if __name__ == "__main__":
    main()