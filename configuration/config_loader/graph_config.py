# oculuz/configuration/config_loader/graph_config.py

import logging
from typing import Dict, Any
from .base_config import BaseConfig

logger = logging.getLogger(__name__)

class GraphConfig(BaseConfig):
    """
    Конфигурация для генерации графа.
    Является синглтоном.
    """
    edge_creation_method: str
    k_scale: float
    min_k_for_large_graphs: int
    large_graph_threshold: int

    def _set_defaults(self) -> None:
        """Устанавливает значения по умолчанию."""
        self.edge_creation_method = "knn"
        self.k_scale = 0.07
        self.min_k_for_large_graphs = 7
        self.large_graph_threshold = 100
        self._validate_config()

    def _validate_config(self) -> None:
        """Проверяет корректность значений конфигурации."""
        if self.edge_creation_method not in ["knn", "sequential"]:
            raise ValueError("edge_creation_method должен быть 'knn' или 'sequential'.")
        if not isinstance(self.k_scale, float) or not (0 < self.k_scale < 1):
            # k_scale может быть и больше 1, но обычно это малая доля
            logger.warning("k_scale обычно находится в диапазоне (0, 1). Текущее значение: %s", self.k_scale)
        if not isinstance(self.min_k_for_large_graphs, int) or self.min_k_for_large_graphs < 1:
            raise ValueError("min_k_for_large_graphs должен быть целым числом >= 1.")
        if not isinstance(self.large_graph_threshold, int) or self.large_graph_threshold < 0:
            raise ValueError("large_graph_threshold должен быть целым числом >= 0.")
        logger.debug("GraphConfig validated successfully.")

    def _to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь."""
        return {
            "edge_creation_method": self.edge_creation_method,
            "k_scale": self.k_scale,
            "min_k_for_large_graphs": self.min_k_for_large_graphs,
            "large_graph_threshold": self.large_graph_threshold,
        }

    def _from_dict(self, data: Dict[str, Any]) -> None:
        """Загружает конфигурацию из словаря."""
        self.edge_creation_method = str(data.get("edge_creation_method", self.edge_creation_method))
        self.k_scale = float(data.get("k_scale", self.k_scale))
        self.min_k_for_large_graphs = int(data.get("min_k_for_large_graphs", self.min_k_for_large_graphs))
        self.large_graph_threshold = int(data.get("large_graph_threshold", self.large_graph_threshold))
        self._validate_config()