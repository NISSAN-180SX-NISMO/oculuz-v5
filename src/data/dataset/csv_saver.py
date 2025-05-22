# oculuz/src/data/dataset/csv_saver.py
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Это изменение важно, чтобы было понятно, какой dataset ожидается
    from .oculuz_dataset import OculuzDataset
    from .dataset_orchestrator import DatasetOrchestrator

logger = logging.getLogger(__name__)

class CSVSaver:
    @staticmethod
    def save_dataset_to_csv(
            # Можно принимать либо готовый список сырых данных, либо оркестратор/датасет
            # Если принимаем Orchestrator, то он должен сгенерировать и вернуть сырые данные
            raw_data_list: List[Dict[str, Any]], # Изменение: принимаем напрямую список сырых данных
            num_sessions_in_data: int, # Количество сессий, которое привело к raw_data_list
            filepath_prefix: Optional[str] = None,
            output_dir: str = "oculuz_datasets" # Директория по умолчанию внутри корня проекта
    ) -> str:
        """
        Сохраняет уже сгенерированные "сырые" данные в CSV.
        Формат CSV: "session_id,longitude,latitude,rssi,fov_dir_sin,fov_dir_cos,fov_width_deg,source_longitude,source_latitude"

        Args:
            raw_data_list: Список словарей, где каждый словарь - это результат dataset.get_raw_data_for_csv().
                           Структура каждого элемента:
                           {
                               "session_id": str,
                               "measurements": [{'latitude', 'longitude', 'rssi'(noisy)}, ...],
                               "source_coords": {'latitude', 'longitude'},
                               "fov_data": [{'fov_dir_sin', 'fov_dir_cos', 'fov_width_deg'}, ...]
                           }
            num_sessions_in_data: Количество уникальных сессий в raw_data_list.
            filepath_prefix: Префикс для имени файла. Если None, используется "dataset_<date>_<time>".
            output_dir: Директория для сохранения CSV файла.

        Returns:
            Полный путь к сохраненному файлу.
        """
        os.makedirs(output_dir, exist_ok=True)

        if filepath_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_{timestamp}.csv"
        elif not filepath_prefix.endswith(".csv"):
             filename = f"{filepath_prefix}.csv"
        else:
            filename = filepath_prefix


        full_filepath = os.path.join(output_dir, filename)

        if os.path.exists(full_filepath):
            logger.warning(f"File {full_filepath} already exists and will be overwritten.")

        all_rows_for_df: List[Dict[str, Any]] = []
        processed_sessions_count = 0

        if not raw_data_list:
            logger.warning("Received empty raw_data_list to save to CSV.")
        else:
            logger.info(f"Preparing {len(raw_data_list)} raw session data entries for CSV saving (expected {num_sessions_in_data} sessions)...")

        for data_sample in raw_data_list:
            try:
                session_id = data_sample["session_id"]
                source_lon = data_sample["source_coords"]["longitude"]
                source_lat = data_sample["source_coords"]["latitude"]

                if not data_sample["measurements"] or not data_sample["fov_data"] or \
                   len(data_sample["measurements"]) != len(data_sample["fov_data"]):
                    logger.warning(f"Inconsistent measurements/fov_data for session {session_id}. Skipping this session for CSV.")
                    continue

                for point_idx, meas_point in enumerate(data_sample["measurements"]):
                    fov_info = data_sample["fov_data"][point_idx]
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
                processed_sessions_count +=1
            except KeyError as e:
                logger.error(f"Missing expected key in raw_data_list item: {e}. Skipping this item. Data: {data_sample}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing raw_data_list item: {e}. Skipping this item. Data: {data_sample}", exc_info=True)
                continue


        if not all_rows_for_df:
            logger.warning("No data rows generated to save to CSV after processing raw_data_list.")
            # Тем не менее, создадим пустой файл, если имя было задано, чтобы показать, что процесс прошел
            if filepath_prefix:
                 pd.DataFrame([]).to_csv(full_filepath, index=False, encoding='utf-8')
            return full_filepath

        df = pd.DataFrame(all_rows_for_df)
        try:
            df.to_csv(full_filepath, index=False, encoding='utf-8')
            logger.info(
                f"Dataset successfully saved to {full_filepath} with {len(df)} rows from {processed_sessions_count} processed sessions.")
        except Exception as e:
            logger.error(f"Failed to write DataFrame to CSV {full_filepath}: {e}", exc_info=True)
            # Можно попытаться сохранить с другим именем или просто вернуть ошибку
            raise # Перевыбрасываем ошибку, чтобы вызывающий код знал о проблеме

        return full_filepath