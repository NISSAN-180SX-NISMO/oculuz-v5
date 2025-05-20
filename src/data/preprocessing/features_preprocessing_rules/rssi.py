# oculuz/src/data/preprocessing/features_preprocessing_rules/rssi.py

import logging
from typing import Union

logger = logging.getLogger(__name__)

def normalize_rssi(value: Union[float, int], min_val: float, max_val: float) -> float:
    """
    Нормализует значение RSSI в диапазон [0, 1].

    Args:
        value: Исходное значение RSSI.
        min_val: Минимальное значение в истинном диапазоне RSSI.
        max_val: Максимальное значение в истинном диапазоне RSSI.

    Returns:
        Нормализованное значение RSSI.
    """
    if min_val >= max_val:
        logger.error(f"Min_val ({min_val}) должен быть меньше max_val ({max_val}) для нормализации RSSI.")
        raise ValueError("min_val должен быть меньше max_val для нормализации RSSI.")

    if value < min_val:
        logger.warning(f"RSSI value {value} is less than min_val {min_val}. Clamping to min_val.")
        value = min_val
    elif value > max_val:
        logger.warning(f"RSSI value {value} is greater than max_val {max_val}. Clamping to max_val.")
        value = max_val

    normalized_value = (value - min_val) / (max_val - min_val)
    logger.debug(f"Normalized RSSI {value} to {normalized_value} (min: {min_val}, max: {max_val})")
    return normalized_value