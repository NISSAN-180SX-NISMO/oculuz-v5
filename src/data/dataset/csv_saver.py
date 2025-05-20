# oculuz/src/data/dataset/csv_saver.py
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .oculuz_dataset import OculuzDataset  # Для type hinting, избегаем циклического импорта

logger = logging.getLogger(__name__)


class CSVSaver:
    @staticmethod
    def save_dataset_to_csv(
            dataset: 'OculuzDataset',  # Используем строку для избежания циклического импорта во время выполнения
            num_samples_to_save: int,
            filepath_prefix: Optional[str] = None,
            output_dir: str = "oculuz_datasets"
    ) -> str:
        """
        Генерирует указанное количество сэмплов из датасета и сохраняет их в CSV.
        Формат CSV: "session_id,longitude,latitude,rssi,fov_dir_sin,fov_dir_cos,fov_width_deg,source_longitude,source_latitude"

        Args:
            dataset: Экземпляр OculuzDataset для генерации данных.
            num_samples_to_save: Количество сессий для генерации и сохранения.
            filepath_prefix: Префикс для имени файла. Если None, используется "dataset_<date>_<time>".
            output_dir: Директория для сохранения CSV файла.

        Returns:
            Полный путь к сохраненному файлу.
        """
        os.makedirs(output_dir, exist_ok=True)

        if filepath_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_{timestamp}.csv"
        else:
            filename = f"{filepath_prefix}.csv"

        full_filepath = os.path.join(output_dir, filename)

        if os.path.exists(full_filepath):
            logger.warning(f"File {full_filepath} already exists and will be overwritten.")

        all_rows: List[Dict[str, Any]] = []

        logger.info(f"Starting dataset generation for CSV saving. Target samples: {num_samples_to_save}")

        for i in range(num_samples_to_save):
            if (i + 1) % max(1, num_samples_to_save // 10) == 0:  # Логировать каждые 10%
                logger.info(f"Generating sample {i + 1}/{num_samples_to_save} for CSV...")

            # Получаем сгенерированную сессию (НЕ граф, а сырые данные до графа)
            # Метод __getitem__ в OculuzDataset возвращает Data объект.
            # Нам нужны данные *до* преобразования в граф и до полного препроцессинга.
            # OculuzDataset должен иметь метод для генерации "сырых" данных сессии.
            # Либо мы здесь воспроизводим часть логики генерации.
            # Проще, если OculuzDataset может вернуть нужные данные.

            # Предположим, OculuzDataset.generate_raw_session_data() существует и возвращает
            # (session_measurements, source_coords, session_fov_data)
            # где session_measurements это [{lat, lon, noisy_rssi}, ...],
            # session_fov_data это [{fov_dir_sin, fov_dir_cos, fov_width_deg}, ...]

            # В текущей структуре __getitem__ делает всю работу.
            # Мы можем вызвать его и извлечь нужные данные из объекта Data.
            # НО! longitude, latitude в Data объекте УЖЕ преобразованы в эмбеддинги.
            # Нам нужны оригинальные. Они хранятся в Data.original_coords.
            # RSSI в Data.x[0] - нормализованный и зашумленный. Нам нужен "оригинальный" зашумленный.
            # Data.original_rssi содержит чистый RSSI.
            # Это усложняет.
            # Лучше, если OculuzDataset будет иметь метод, возвращающий данные в нужном формате для CSV.

            # Пока что, для простоты, будем доставать из Data объекта, но это не идеально.
            # Модифицируем OculuzDataset, чтобы он сохранял нужные данные для CSV.
            data_sample = dataset.get_raw_data_for_csv(i)  # Этот метод нужно добавить в OculuzDataset

            session_id = data_sample["session_id"]
            source_lon = data_sample["source_coords"]["longitude"]
            source_lat = data_sample["source_coords"]["latitude"]

            for point_idx, meas_point in enumerate(data_sample["measurements"]):
                fov_info = data_sample["fov_data"][point_idx]
                row = {
                    "session_id": session_id,
                    "longitude": meas_point["longitude"],
                    "latitude": meas_point["latitude"],
                    "rssi": meas_point["rssi"],  # Это должен быть зашумленный RSSI
                    "fov_dir_sin": fov_info["fov_dir_sin"],
                    "fov_dir_cos": fov_info["fov_dir_cos"],
                    "fov_width_deg": fov_info["fov_width_deg"],
                    "source_longitude": source_lon,
                    "source_latitude": source_lat,
                }
                all_rows.append(row)

        if not all_rows:
            logger.warning("No data generated to save to CSV.")
            return full_filepath  # Возвращаем путь, даже если файл пустой или не создан

        df = pd.DataFrame(all_rows)
        df.to_csv(full_filepath, index=False, encoding='utf-8')
        logger.info(
            f"Dataset successfully saved to {full_filepath} with {len(df)} rows from {num_samples_to_save} sessions.")

        return full_filepath