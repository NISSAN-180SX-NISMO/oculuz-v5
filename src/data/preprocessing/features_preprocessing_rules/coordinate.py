# oculuz/src/data/preprocessing/features_preprocessing_rules/coordinate.py

import torch
import math
import logging

logger = logging.getLogger(__name__)


def get_sinusoidal_embedding(value: float, embedding_dim: int) -> torch.Tensor:
    """
    Создает синусоидальный пространственный эмбеддинг для одного координатного значения.

    Args:
        value: Значение координаты (например, широта или долгота).
        embedding_dim: Размерность эмбеддинга (должна быть четной).

    Returns:
        torch.Tensor: Вектор эмбеддинга размерности `embedding_dim`.
    """
    if embedding_dim % 2 != 0:
        logger.error(f"Embedding dimension {embedding_dim} must be even for sinusoidal encoding.")
        raise ValueError("Embedding dimension must be even.")

    position = torch.tensor([[value]], dtype=torch.float32)  # Shape (1, 1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * \
                         -(math.log(10000.0) / embedding_dim))  # Shape (embedding_dim / 2)

    embedding = torch.zeros(1, embedding_dim, dtype=torch.float32)  # Shape (1, embedding_dim)

    # PE(pos, 2i) = sin(pos * div_term)
    embedding[0, 0::2] = torch.sin(position * div_term)
    # PE(pos, 2i+1) = cos(pos * div_term)
    embedding[0, 1::2] = torch.cos(position * div_term)

    logger.debug(f"Generated sinusoidal embedding for value {value} with dim {embedding_dim}")
    return embedding.squeeze(0)  # Return shape (embedding_dim)