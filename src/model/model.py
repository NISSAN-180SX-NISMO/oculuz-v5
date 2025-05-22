# oculuz/src/model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # F is not used directly here but often useful
from torch_geometric.nn import GATConv  # Make sure torch_geometric is installed


def get_activation_module(activation_name: str) -> nn.Module:
    """
    Returns a torch.nn activation module based on its name.
    """
    if activation_name is None or activation_name.lower() == "none":
        return nn.Identity()
    elif activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    # Add other activations like 'leaky_relu', 'tanh', 'sigmoid' if needed
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")



# Example usage (assuming config is loaded into a dict called `loaded_model_config`
# and model_in_channels is determined)
# from src.model.model import GNNModel # Adjust import path as needed

# model_in_channels = 1 + 64 # Example: RSSI + 64-dim coord embedding
# model = GNNModel(model_in_channels=model_in_channels, config=loaded_model_config)
# print(model)

class GNNModel(nn.Module):
    """
    Graph Neural Network model using GAT layers followed by an MLP per node.
    The model predicts a single continuous value (e.g., cleaned RSSI) for each node.
    """

    def __init__(self, model_in_channels: int, config: dict):
        """
        Initializes the GNNModel.

        Args:
            model_in_channels (int): Dimensionality of input node features.
                                     This is determined by the preprocessing steps.
            config (dict): Configuration dictionary, typically loaded from model_config.yaml.
                           Expected keys: 'gat_layer_configs', 'mlp_head_config', 'output_channels_mlp'.
        """
        super().__init__()

        self.model_in_channels = model_in_channels
        self.config = config  # Store config for potential later reference

        # Initialize GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_activations = nn.ModuleList()

        current_channels = self.model_in_channels
        for layer_conf in config['gat_layer_configs']:
            gat_conv = GATConv(
                in_channels=current_channels,
                out_channels=layer_conf['out_channels'],
                heads=layer_conf['heads'],
                concat=layer_conf['concat'],
                dropout=layer_conf['dropout'],  # GATConv applies dropout on attention and features
                # add_self_loops=True, # Default and generally recommended
                # negative_slope=0.2   # Default for LeakyReLU in attention
            )
            self.gat_layers.append(gat_conv)
            self.gat_activations.append(get_activation_module(layer_conf['activation']))

            if layer_conf['concat']:
                current_channels = layer_conf['out_channels'] * layer_conf['heads']
            else:
                current_channels = layer_conf['out_channels']

        # MLP Head (applied per node)
        mlp_input_dim = current_channels  # This is the 'node_dim' from GAT layers output
        mlp_config = config['mlp_head_config']

        if mlp_config['hidden_to_input_ratio'] <= 0:
            raise ValueError("MLP 'hidden_to_input_ratio' must be positive and non-zero.")

        # Calculate MLP hidden dimension based on the ratio (node_dim // ratio)
        mlp_hidden_dim_float = mlp_input_dim / mlp_config['hidden_to_input_ratio']
        mlp_hidden_dim = int(mlp_hidden_dim_float)

        # Ensure mlp_hidden_dim is at least 1 if mlp_input_dim is positive,
        # otherwise Linear layer might be invalid.
        if mlp_hidden_dim == 0 and mlp_input_dim > 0:
            mlp_hidden_dim = 1
        elif mlp_input_dim == 0 and config['output_channels_mlp'] > 0:
            # This case (mlp_input_dim = 0) implies a potentially problematic GAT configuration
            # or an issue where current_channels became 0.
            # If mlp_hidden_dim also becomes 0, the MLP structure is (0,0) -> (0, output_mlp_channels).
            # This is valid but likely not intended. Consider adding a warning or stricter check.
            pass

        self.mlp_fc1 = nn.Linear(mlp_input_dim, mlp_hidden_dim)
        self.mlp_activation = get_activation_module(mlp_config['activation'])
        self.mlp_dropout_layer = nn.Dropout(p=mlp_config['dropout'])
        self.mlp_fc2 = nn.Linear(mlp_hidden_dim, config['output_channels_mlp'])

    def forward(self, data):
        """
        Forward pass of the GNN model.

        Args:
            data (torch_geometric.data.Data or torch_geometric.data.Batch):
                  A PyG Data or Batch object containing:
                  - data.x: Node features [num_nodes, model_in_channels]
                  - data.edge_index: Graph connectivity [2, num_edges]

        Returns:
            torch.Tensor: Predicted values for each node (e.g., cleaned RSSI)
                          Shape: [num_nodes, output_channels_mlp]
        """
        x, edge_index = data.x, data.edge_index

        if x is None:
            raise ValueError("Input data.x is None. Node features are required.")
        if x.shape[1] != self.model_in_channels:
            raise ValueError(
                f"Input feature dimension mismatch. Model configured for {self.model_in_channels} "
                f"input features, but received data.x with {x.shape[1]} features."
            )

        # Pass through GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            x = self.gat_activations[i](x)
            # GATConv's internal dropout is applied on attention weights and features.
            # If additional dropout *between* layers is desired, it would be added here.

        # Pass through MLP head (applied to each node's features)
        x = self.mlp_fc1(x)
        x = self.mlp_activation(x)
        x = self.mlp_dropout_layer(x)
        x_predicted = self.mlp_fc2(x)  # Output for regression (no final activation)

        # As per prompt: "из forward должен вернуться тот же набор точек, только с очищенным rssi."
        # This means returning a tensor of predicted RSSI values for each node.
        return x_predicted

    def __repr__(self):
        # A simple representation string
        return (f"{self.__class__.__name__}("
                f"in_channels={self.model_in_channels}, "
                f"out_channels_mlp={self.config['output_channels_mlp']}, "
                f"num_gat_layers={len(self.gat_layers)})")