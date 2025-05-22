# oculuz/src/data/preprocessing/coder_graph.py

import logging
import torch
from torch_geometric.data import Data
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

# Используем новый ConfigLoader
from configuration.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class CoderGraph:
    def __init__(self,
                 graph_config_loader: Optional[ConfigLoader] = None,
                 data_prep_config_loader: Optional[ConfigLoader] = None,
                 default_graph_config_path: str = "configuration/graph_config.yaml",
                 default_data_prep_config_path: str = "configuration/data_preprocessing_config.yaml"):
        """
        Инициализирует кодировщик графа.

        Args:
            graph_config_loader: ConfigLoader для graph_config.yaml.
            data_prep_config_loader: ConfigLoader для data_preprocessing_config.yaml.
        """
        if graph_config_loader is None:
            self.graph_config: ConfigLoader = ConfigLoader(default_graph_config_path)
        else:
            self.graph_config: ConfigLoader = graph_config_loader

        if data_prep_config_loader is None:
            self.data_prep_config: ConfigLoader = ConfigLoader(default_data_prep_config_path)
        else:
            self.data_prep_config: ConfigLoader = data_prep_config_loader
        logger.info("CoderGraph initialized.")
        # logger.debug(f"CoderGraph graph_config: {self.graph_config.data if hasattr(self.graph_config, 'data') else self.graph_config}")
        # logger.debug(f"CoderGraph data_prep_config: {self.data_prep_config.data if hasattr(self.data_prep_config, 'data') else self.data_prep_config}")

    def _get_config_param(self, config_loader: ConfigLoader, keys: List[str], default: Optional[Any] = None) -> Any:
        """Вспомогательный метод для безопасного извлечения вложенных параметров."""
        current_level = config_loader.data  # Предполагаем, что .data возвращает словарь
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                # logger.warning(f"Config parameter {' -> '.join(keys)} not found in {config_loader.config_file_path}. Using default: {default}")
                return default
        return current_level

    def _assemble_node_features(self, processed_session_data: List[Dict[str, Any]]) -> torch.Tensor:
        # rssi (1) + lat_emb (dim) + lon_emb (dim)
        coord_dim = self._get_config_param(self.data_prep_config,["coordinate", "embedding_dim"], 64)
        num_features_expected = 1 + 2 * coord_dim

        if not processed_session_data:
            logger.warning("Received empty processed_session_data for feature assembly.")
            return torch.empty((0, num_features_expected), dtype=torch.float32)

        node_features_list: List[torch.Tensor] = []
        for point_data in processed_session_data:
            rssi_norm = torch.tensor([point_data.get("rssi_norm", 0.0)], dtype=torch.float32)  # Default 0.0 if missing
            lat_emb = point_data.get("latitude_emb")
            lon_emb = point_data.get("longitude_emb")

            if lat_emb is None:
                # logger.warning("Latitude embedding is missing for a point. Using zero embedding.")
                lat_emb = torch.zeros(coord_dim, dtype=torch.float32)
            if lon_emb is None:
                # logger.warning("Longitude embedding is missing for a point. Using zero embedding.")
                lon_emb = torch.zeros(coord_dim, dtype=torch.float32)

            # Обеспечиваем, что эмбеддинги имеют правильную форму (1D) перед конкатенацией
            feature_parts = [rssi_norm, lat_emb.view(-1), lon_emb.view(-1)]
            node_features_list.append(torch.cat(feature_parts))

        if not node_features_list:
            return torch.empty((0, num_features_expected), dtype=torch.float32)

        stacked_features = torch.stack(node_features_list)
        if stacked_features.shape[1] != num_features_expected:
            logger.error(f"Assembled node features have {stacked_features.shape[1]} dimensions, "
                         f"but expected {num_features_expected} based on config (coord_dim={coord_dim}). "
                         f"Check embedding dimensions and feature assembly logic.")
            # Можно либо выбросить ошибку, либо попытаться исправить (например, паддингом/обрезкой, но это опасно)
            # raise ValueError("Mismatch in assembled node feature dimensions.")
            # Если нужно вернуть хоть что-то:
            # return torch.empty((stacked_features.shape[0], num_features_expected), dtype=torch.float32) # Возвращает пустой тензор правильной формы
            # Или обрезать/дополнить нулями, но это скроет проблему. Лучше ошибка или логирование выше.

        return stacked_features

    def _calculate_k_for_knn(self, num_points: int) -> int:
        if num_points <= 1: return 0
        try:
            rules = self.graph_config["graph_creation_rules"]["knn"]
            k_scale = rules["k_scale"]
            min_k_large = rules["min_k_for_large_graphs"]  # ТЗ: это 7
            # large_graph_threshold = rules.get("large_graph_threshold", 100) # Не используется в формуле из ТЗ
        except KeyError as e:
            logger.error(f"Missing key in graph_config for KNN k calculation: {e}. Using default k=3.", exc_info=True)
            return min(3, num_points - 1) if num_points > 1 else 0

        # ТЗ: k = max(1, round(min(7, k_scale * num_points)))
        k_val = k_scale * num_points
        k_val = min(min_k_large, k_val)  # min(7, k_val)
        k_val = round(k_val)
        k_val = max(1, k_val)  # k должно быть как минимум 1, если есть хотя бы 2 точки

        # k не может быть больше, чем num_points - 1 (количество других точек)
        k_val = min(k_val, num_points - 1)

        # logger.info(
        #     f"Calculated k={k_val} for {num_points} points (k_scale={k_scale}, min_k_large={min_k_large})")
        return int(k_val)

    def _create_edges_knn(self, processed_session_data: List[Dict[str, Any]]) -> torch.Tensor:
        num_points = len(processed_session_data)
        if num_points <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        coordinates = np.array(
            [[p.get('original_longitude', 0.0), p.get('original_latitude', 0.0)] for p in processed_session_data]
        )
        if coordinates.shape[0] == 0:  # Все точки без координат
            return torch.empty((2, 0), dtype=torch.long)

        coordinates_rad = np.radians(coordinates)
        k = self._calculate_k_for_knn(num_points)
        if k == 0: return torch.empty((2, 0), dtype=torch.long)

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='haversine')
        try:
            nn.fit(coordinates_rad)
            indices = nn.kneighbors(coordinates_rad, return_distance=False)
        except ValueError as e:  # Может возникнуть, если k > num_points (хотя _calculate_k_for_knn это предотвращает)
            logger.error(f"Error in NearestNeighbors fitting/querying for k={k}, n_points={num_points}: {e}",
                         exc_info=True)
            return torch.empty((2, 0), dtype=torch.long)

        edge_list_src: List[int] = []
        edge_list_dst: List[int] = []

        for i in range(num_points):
            for j_idx in range(1, indices.shape[1]):  # indices.shape[1] должно быть k+1
                if j_idx < indices.shape[
                    1]:  # Доп. проверка на случай если k было скорректировано до очень малого значения
                    neighbor_idx = indices[i, j_idx]
                    edge_list_src.append(i)
                    edge_list_dst.append(neighbor_idx)

        if not edge_list_src:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        # logger.info(f"Created KNN edge_index with {edge_index.shape[1]} edges for {num_points} nodes using k={k}.")
        return edge_index

    def _create_edges_sequential(self, num_points: int) -> torch.Tensor:
        if num_points <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        edge_list_src: List[int] = []
        edge_list_dst: List[int] = []
        for i in range(num_points - 1):
            edge_list_src.extend([i, i + 1])  # Ребро от i к i+1
            edge_list_dst.extend([i + 1, i])  # Ребро от i+1 к i (для неориентированного)

        if not edge_list_src:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        # logger.info(f"Created sequential edge_index with {edge_index.shape[1]} edges for {num_points} nodes.")
        return edge_index

    def create_graph_data(self, processed_session_data: List[Dict[str, Any]], **kwargs) -> Data:
        num_points = len(processed_session_data)
        coord_emb_dim = self._get_config_param(self.data_prep_config,
                                               ["coordinate", "embedding_dim"], 64)
        num_features_expected = 1 + 2 * coord_emb_dim

        if num_points == 0:
            # logger.warning("Attempting to create a graph from empty session data.")
            return Data(x=torch.empty((0, num_features_expected), dtype=torch.float32),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        num_nodes=0,
                        **kwargs)

        node_features = self._assemble_node_features(processed_session_data)

        edge_creation_method = self._get_config_param(self.graph_config, ["edge_creation_method"], "sequential")

        if edge_creation_method == "knn":
            edge_index = self._create_edges_knn(processed_session_data)
        elif edge_creation_method == "sequential":
            edge_index = self._create_edges_sequential(num_points)
        else:
            logger.error(
                f"Unknown edge creation method: {edge_creation_method} from graph_config. Defaulting to sequential.")
            edge_index = self._create_edges_sequential(num_points)  # Fallback

        original_coords_list = []
        original_rssi_list = []
        default_rssi_min = self._get_config_param(self.data_prep_config,["rssi", "min_val"], -130.0)

        for p in processed_session_data:
            original_coords_list.append([p.get('original_longitude', 0.0), p.get('original_latitude', 0.0)])
            original_rssi_list.append(p.get('original_rssi', default_rssi_min))

        original_coords = torch.tensor(original_coords_list, dtype=torch.float32)
        original_rssi = torch.tensor(original_rssi_list, dtype=torch.float32).unsqueeze(1)

        graph = Data(x=node_features, edge_index=edge_index, num_nodes=num_points,
                     original_coords=original_coords, original_rssi=original_rssi, **kwargs)

        # logger.info(f"Created graph data object with {graph.num_nodes} nodes and {graph.num_edges} edges using '{edge_creation_method}'.")
        # logger.debug(f"Graph details: X shape {graph.x.shape if hasattr(graph, 'x') else 'N/A'}, "
        #              f"edge_index shape {graph.edge_index.shape if hasattr(graph, 'edge_index') else 'N/A'}")
        return graph