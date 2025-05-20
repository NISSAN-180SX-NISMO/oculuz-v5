# oculuz/src/data/preprocessing/coder_graph.py

import logging
import torch
from torch_geometric.data import Data
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

from configuration.config_loader import GraphConfig, DataPreprocessingConfig

logger = logging.getLogger(__name__)


class CoderGraph:
    """
    Кодирует структуру маршрута (сессии измерений) в объект torch_geometric.data.Data.
    Собирает тензор признаков узлов и создает ребра графа.
    """

    def __init__(self, graph_config: GraphConfig = None, data_prep_config: DataPreprocessingConfig = None):
        """
        Инициализирует кодировщик графа.

        Args:
            graph_config: Экземпляр GraphConfig. Если None, будет загружен
                          или создан экземпляр по умолчанию.
            data_prep_config: Экземпляр DataPreprocessingConfig. Нужен для получения
                              размерности эмбеддингов при сборке тензора признаков.
        """
        if graph_config is None:
            self.graph_config = GraphConfig.get_instance()
            GraphConfig.load("oculuz/configuration/graph_config.yaml")
        else:
            self.graph_config = graph_config

        if data_prep_config is None:
            self.data_prep_config = DataPreprocessingConfig.get_instance()
            DataPreprocessingConfig.load("oculuz/configuration/data_preprocessing_config.yaml")
        else:
            self.data_prep_config = data_prep_config
        logger.info("CoderGraph initialized.")

    def _assemble_node_features(self, processed_session_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Собирает матрицу признаков X для всех узлов (точек измерений) в сессии.
        Порядок конкатенации: [rssi_norm, latitude_emb, longitude_emb].

        Args:
            processed_session_data: Список словарей с предобработанными признаками
                                   из DataPreprocessor.

        Returns:
            torch.Tensor: Матрица признаков [num_nodes, num_node_features].
        """
        if not processed_session_data:
            logger.warning("Received empty processed_session_data for feature assembly.")
            # Определяем количество фич на основе конфига, если данных нет
            # rssi (1) + lat_emb (dim) + lon_emb (dim)
            num_features = 1 + 2 * self.data_prep_config.coordinate_embedding_dim
            return torch.empty((0, num_features), dtype=torch.float32)

        node_features_list: List[torch.Tensor] = []
        for point_data in processed_session_data:
            rssi_norm = torch.tensor([point_data.get("rssi_norm", 0.0)], dtype=torch.float32)
            lat_emb = point_data.get("latitude_emb")
            lon_emb = point_data.get("longitude_emb")

            coord_dim = self.data_prep_config.coordinate_embedding_dim
            if lat_emb is None:
                logger.warning("Latitude embedding is missing for a point. Using zero embedding.")
                lat_emb = torch.zeros(coord_dim, dtype=torch.float32)
            if lon_emb is None:
                logger.warning("Longitude embedding is missing for a point. Using zero embedding.")
                lon_emb = torch.zeros(coord_dim, dtype=torch.float32)

            # Убедимся, что эмбеддинги имеют правильную форму (1D)
            feature_parts = [rssi_norm, lat_emb.view(-1), lon_emb.view(-1)]
            node_features_list.append(torch.cat(feature_parts))

        if not node_features_list:  # Если вдруг все точки были без нужных данных
            num_features = 1 + 2 * self.data_prep_config.coordinate_embedding_dim
            return torch.empty((0, num_features), dtype=torch.float32)

        return torch.stack(node_features_list)

    def _calculate_k_for_knn(self, num_points: int) -> int:
        """
        Рассчитывает параметр k для KNN согласно правилу из ТЗ.
        k = max(1, round(min(min_k_for_large_graphs, k_scale * num_points)))
        Если num_points > large_graph_threshold, то k = min_k_for_large_graphs
        (исправленная логика согласно ТЗ: k = max(1, round(min(7, k_scale * num_points))))
        "но если точек больше 100 то k = 7 (k = max(1, round(min(7, k_scale * num_points))))"
        Эта формулировка немного двусмысленна. Интерпретируем:
        1. Базовое k = round(k_scale * num_points)
        2. Ограничиваем сверху: min(self.graph_config.min_k_for_large_graphs, k_from_scale)
        3. Ограничиваем снизу: max(1, k_limited_above)
        ТЗ: "k = max(1, round(min(7, k_scale * num_points)))"
        и "но если точек больше 100 то k = 7" - это условие НЕ переопределяет предыдущее,
        а скорее min(7, ...) уже учитывает это.
        То есть, если k_scale * num_points дает, например, 10, то min(7, 10) = 7.
        Если k_scale * num_points дает 3.4, то min(7, 3.4) = 3.4.
        """
        if num_points <= 1:  # Нет соседей для одной или нуля точек
            return 0

        k_val = self.graph_config.k_scale * num_points
        k_val = min(self.graph_config.min_k_for_large_graphs, k_val)  # min(7, k_val)
        k_val = round(k_val)
        k_val = max(1, k_val)

        logger.info(
            f"Calculated k={k_val} for {num_points} points (k_scale={self.graph_config.k_scale}, min_k_large={self.graph_config.min_k_for_large_graphs})")
        return int(k_val)

    def _create_edges_knn(self, processed_session_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Создает ребра графа на основе k-ближайших соседей.
        Использует оригинальные координаты (широта, долгота).
        """
        num_points = len(processed_session_data)
        if num_points <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        coordinates = np.array(
            [[p.get('original_longitude', 0.0), p.get('original_latitude', 0.0)] for p in processed_session_data]
        )
        if coordinates.shape[0] == 0:  # Все точки без координат
            return torch.empty((2, 0), dtype=torch.long)

        # Преобразуем градусы в радианы для haversine
        coordinates_rad = np.radians(coordinates)

        k = self._calculate_k_for_knn(num_points)
        if k == 0:  # Например, если num_points = 1
            return torch.empty((2, 0), dtype=torch.long)

        # n_neighbors = k+1, так как точка является своим соседом
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='haversine')
        nn.fit(coordinates_rad)
        # distances, indices = nn.kneighbors(coordinates_rad)
        indices = nn.kneighbors(coordinates_rad, return_distance=False)

        edge_list_src: List[int] = []
        edge_list_dst: List[int] = []

        for i in range(num_points):
            for j_idx in range(1, indices.shape[1]):  # Начинаем с 1, чтобы пропустить ребро к себе
                neighbor_idx = indices[i, j_idx]
                # Добавляем ребра в обе стороны для неориентированного графа
                edge_list_src.append(i)
                edge_list_dst.append(neighbor_idx)
                # Можно было бы добавить и (neighbor_idx, i), но PyG обычно сам обрабатывает
                # неориентированность через message passing или можно использовать to_undirected()

        if not edge_list_src:
            return torch.empty((2, 0), dtype=torch.long)

        # Создаем тензор ребер и делаем его неориентированным симметричным
        # и удаляем дубликаты и петли (хотя петель быть не должно из-за j_idx=1)
        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)

        # Для GAT обычно ребра должны быть неориентированными (т.е. если есть (i,j), то и (j,i))
        # Можно было бы добавить здесь from torch_geometric.utils import to_undirected
        # edge_index = to_undirected(edge_index, num_nodes=num_points)
        # Но пока оставим так, т.к. GAT и без этого справится, если данные есть.
        # Важнее, чтобы не было дубликатов, если k очень большое.
        # Сортировка и unique для чистоты
        edge_index_T = edge_index.t().contiguous()
        edge_index_T = torch.unique(edge_index_T, dim=0)
        edge_index = edge_index_T.t().contiguous()

        logger.info(f"Created KNN edge_index with {edge_index.shape[1]} edges for {num_points} nodes using k={k}.")
        return edge_index

    def _create_edges_sequential(self, num_points: int) -> torch.Tensor:
        """
        Создает ребра графа, соединяя последовательные точки в маршруте.
        Создает двунаправленные ребра (i -> i+1 и i+1 -> i).
        """
        if num_points <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        edge_list_src: List[int] = []
        edge_list_dst: List[int] = []
        for i in range(num_points - 1):
            edge_list_src.extend([i, i + 1])
            edge_list_dst.extend([i + 1, i])

        if not edge_list_src:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        logger.info(f"Created sequential edge_index with {edge_index.shape[1]} edges for {num_points} nodes.")
        return edge_index

    def create_graph_data(self, processed_session_data: List[Dict[str, Any]], **kwargs) -> Data:
        """
        Создает объект torch_geometric.data.Data из предобработанных данных сессии.

        Args:
            processed_session_data: Список словарей с предобработанными данными
                                   для каждой точки измерения.
            **kwargs: Дополнительные атрибуты, которые нужно добавить в объект Data
                      (например, y, fov_data, source_coords).

        Returns:
            torch_geometric.data.Data: Объект графа.
        """
        num_points = len(processed_session_data)
        if num_points == 0:
            logger.warning("Attempting to create a graph from empty session data.")
            # Возвращаем пустой Data объект с корректными, но пустыми тензорами
            num_features = 1 + 2 * self.data_prep_config.coordinate_embedding_dim
            return Data(x=torch.empty((0, num_features), dtype=torch.float32),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        num_nodes=0,
                        **kwargs)

        node_features = self._assemble_node_features(processed_session_data)

        if self.graph_config.edge_creation_method == "knn":
            edge_index = self._create_edges_knn(processed_session_data)
        elif self.graph_config.edge_creation_method == "sequential":
            edge_index = self._create_edges_sequential(num_points)
        else:
            logger.error(f"Unknown edge creation method: {self.graph_config.edge_creation_method}")
            raise ValueError(f"Неизвестный метод создания ребер: {self.graph_config.edge_creation_method}")

        # Сохраняем оригинальные координаты в объекте Data, если они могут понадобиться
        original_coords = torch.tensor(
            [[p.get('original_longitude', 0.0), p.get('original_latitude', 0.0)] for p in processed_session_data],
            dtype=torch.float32
        )
        original_rssi = torch.tensor(
            [p.get('original_rssi', self.data_prep_config.rssi_min_val) for p in processed_session_data],
            dtype=torch.float32
        ).unsqueeze(1)  # (num_nodes, 1)

        graph = Data(x=node_features, edge_index=edge_index, num_nodes=num_points,
                     original_coords=original_coords, original_rssi=original_rssi, **kwargs)

        logger.info(f"Created graph data object with {graph.num_nodes} nodes and {graph.num_edges} edges.")
        logger.debug(f"Graph details: {graph}")
        return graph