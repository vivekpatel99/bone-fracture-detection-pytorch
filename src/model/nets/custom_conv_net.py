import torch
from torch import nn


class CustomConvNet(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        conv_layers: list[int],
        num_classes: int,
        dropout_rate: float,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        input_dim = input_shape[0]
        self.output_dims = conv_layers
        layers: list[nn.Module] = []
        # --- Convolutional layers
        for out_dim in conv_layers:
            layers.append(nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = out_dim

        self.conv_layers = nn.Sequential(*layers)

        # --- To determine the input size for the linear layer
        self.flattener = nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_dim = self.flattener(dummy_output).shape[1]
        # --- Classification head
        cls_layers = []
        current_fc_input_features = self.flatten_dim
        neuron_per_layer = 32
        for _ in range(num_hidden_layers):
            cls_layers.append(nn.Linear(current_fc_input_features, neuron_per_layer, bias=False))
            cls_layers.append(nn.BatchNorm1d(neuron_per_layer))
            cls_layers.append(nn.ReLU())
            cls_layers.append(nn.Dropout(dropout_rate))

            current_fc_input_features = neuron_per_layer
            neuron_per_layer = neuron_per_layer * 2

        cls_layers.append(nn.Linear(current_fc_input_features, num_classes))

        self.classification_head = nn.Sequential(*cls_layers)
        self.model = nn.Sequential(self.conv_layers, self.flattener, self.classification_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
