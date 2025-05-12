import pytest
import torch
from torch import nn

from src.model.net.custom_conv_net import CustomConvNet

# Default parameters for tests
DEFAULT_INPUT_SHAPE = (3, 64, 64)  # C, H, W
DEFAULT_CONV_LAYERS_PARAM = 3  # Results in 2 conv blocks
DEFAULT_NUM_CLASSES = 5
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_NUM_HIDDEN_LAYERS = 2
BATCH_SIZE = 4


class TestCustomConvNet:
    def test_initialization_basic(self):
        """Test basic initialization of the CustomConvNet model."""
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=DEFAULT_CONV_LAYERS_PARAM,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        )
        assert isinstance(model, nn.Module)
        assert isinstance(model.conv_layers, nn.Sequential)
        assert isinstance(model.flattener, nn.Flatten)
        assert isinstance(model.classification_head, nn.Sequential)
        assert isinstance(model.model, nn.Sequential)  # top-level sequential model

    def test_forward_pass_output_shape(self):
        """Test the output shape after a forward pass."""
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=DEFAULT_CONV_LAYERS_PARAM,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        )
        dummy_input = torch.randn(BATCH_SIZE, *DEFAULT_INPUT_SHAPE)
        output = model(dummy_input)
        assert output.shape == (BATCH_SIZE, DEFAULT_NUM_CLASSES)

    @pytest.mark.parametrize(
        "conv_layers_count, num_hidden_layers_count, num_classes_val",
        [
            (1, 0, 2),  # Min conv_layers (0 conv blocks), 0 hidden layers, 2 classes
            (2, 1, 3),  # 1 conv block, 1 hidden layer, 3 classes
            (4, 3, 10),  # 3 conv blocks, 3 hidden layers, 10 classes
        ],
    )
    def test_varying_configurations(self, conv_layers_count, num_hidden_layers_count, num_classes_val):
        """Test model with varying numbers of layers and classes."""
        input_shape = (3, 32, 32)  # Use a smaller H, W for this test
        model = CustomConvNet(
            input_shape=input_shape,
            conv_layers=conv_layers_count,
            num_classes=num_classes_val,
            dropout_rate=0.1,
            num_hidden_layers=num_hidden_layers_count,
        )
        dummy_input = torch.randn(BATCH_SIZE, *input_shape)
        output = model(dummy_input)
        assert output.shape == (BATCH_SIZE, num_classes_val)

        # Number of conv blocks = conv_layers_count - 1. Each block has 5 layers.
        expected_conv_sub_modules = max(0, conv_layers_count - 1) * 5
        assert len(model.conv_layers) == expected_conv_sub_modules

        # Each hidden layer block has 4 layers, plus one final Linear layer.
        expected_cls_head_sub_modules = num_hidden_layers_count * 4 + 1
        assert len(model.classification_head) == expected_cls_head_sub_modules

        final_fc_layer = model.classification_head[-1]
        assert isinstance(final_fc_layer, nn.Linear)
        assert final_fc_layer.out_features == num_classes_val

    def test_conv_layers_structure_detailed(self):
        """Test the detailed structure of convolutional layers."""
        conv_layers_param = 3  # This will create 2 conv blocks
        input_channels = DEFAULT_INPUT_SHAPE[0]
        dropout_rate = 0.25
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=conv_layers_param,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=dropout_rate,
            num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        )

        assert len(model.conv_layers) == (conv_layers_param - 1) * 5  # 2 blocks * 5 layers = 10 layers

        # Block 1
        assert isinstance(model.conv_layers[0], nn.Conv2d)
        assert model.conv_layers[0].in_channels == input_channels
        assert model.conv_layers[0].out_channels == 32
        assert isinstance(model.conv_layers[1], nn.BatchNorm2d)
        assert model.conv_layers[1].num_features == 32
        assert isinstance(model.conv_layers[2], nn.ReLU)
        assert isinstance(model.conv_layers[3], nn.MaxPool2d)
        assert isinstance(model.conv_layers[4], nn.Dropout)
        assert model.conv_layers[4].p == dropout_rate

        # Block 2
        assert isinstance(model.conv_layers[5], nn.Conv2d)
        assert model.conv_layers[5].in_channels == 32
        assert model.conv_layers[5].out_channels == 64  # 2 * 32
        assert isinstance(model.conv_layers[6], nn.BatchNorm2d)
        assert model.conv_layers[6].num_features == 64
        assert isinstance(model.conv_layers[7], nn.ReLU)
        assert isinstance(model.conv_layers[8], nn.MaxPool2d)
        assert isinstance(model.conv_layers[9], nn.Dropout)
        assert model.conv_layers[9].p == dropout_rate

    def test_classification_head_structure_detailed(self):
        """Test the detailed structure of the classification head."""
        num_hidden_layers_param = 2
        dropout_rate = 0.3
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=DEFAULT_CONV_LAYERS_PARAM,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers_param,
        )

        assert len(model.classification_head) == num_hidden_layers_param * 4 + 1  # 2*4 + 1 = 9 layers

        # Layer 0: Linear (input to first hidden)
        assert isinstance(model.classification_head[0], nn.Linear)
        assert model.classification_head[0].in_features == model.flatten_dim
        assert model.classification_head[0].out_features == 32  # neuron_per_layer starts at 32
        assert isinstance(model.classification_head[1], nn.BatchNorm1d)  # BN
        assert model.classification_head[1].num_features == 32
        assert isinstance(model.classification_head[2], nn.ReLU)  # ReLU
        assert isinstance(model.classification_head[3], nn.Dropout)  # Dropout
        assert model.classification_head[3].p == dropout_rate

        # Layer 4: Linear (first hidden to second hidden)
        assert isinstance(model.classification_head[4], nn.Linear)
        assert model.classification_head[4].in_features == 32
        assert model.classification_head[4].out_features == 64  # neuron_per_layer * 2
        assert isinstance(model.classification_head[5], nn.BatchNorm1d)  # BN
        assert model.classification_head[5].num_features == 64
        assert isinstance(model.classification_head[6], nn.ReLU)  # ReLU
        assert isinstance(model.classification_head[7], nn.Dropout)  # Dropout
        assert model.classification_head[7].p == dropout_rate

        # Layer 8: Final Linear layer
        assert isinstance(model.classification_head[8], nn.Linear)
        assert model.classification_head[8].in_features == 64
        assert model.classification_head[8].out_features == DEFAULT_NUM_CLASSES

    def test_no_hidden_layers(self):
        """Test model configuration with zero hidden layers."""
        num_hidden_layers_param = 0
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=DEFAULT_CONV_LAYERS_PARAM,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            num_hidden_layers=num_hidden_layers_param,
        )
        assert len(model.classification_head) == 1  # Only the final Linear layer
        final_layer = model.classification_head[0]
        assert isinstance(final_layer, nn.Linear)
        assert final_layer.in_features == model.flatten_dim
        assert final_layer.out_features == DEFAULT_NUM_CLASSES

        dummy_input = torch.randn(BATCH_SIZE, *DEFAULT_INPUT_SHAPE)
        output = model(dummy_input)
        assert output.shape == (BATCH_SIZE, DEFAULT_NUM_CLASSES)

    def test_min_conv_layers(self):
        """Test model with conv_layers=1 (no actual conv blocks created)."""
        conv_layers_param = 1
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=conv_layers_param,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        )
        assert len(model.conv_layers) == 0  # No conv blocks if conv_layers=1

        expected_flatten_dim = DEFAULT_INPUT_SHAPE[0] * DEFAULT_INPUT_SHAPE[1] * DEFAULT_INPUT_SHAPE[2]
        assert model.flatten_dim == expected_flatten_dim

        dummy_input = torch.randn(BATCH_SIZE, *DEFAULT_INPUT_SHAPE)
        output = model(dummy_input)
        assert output.shape == (BATCH_SIZE, DEFAULT_NUM_CLASSES)

    def test_dropout_eval_mode(self):
        """Test that dropout layers are active during training and inactive during eval."""
        dropout_rate = 0.5
        model = CustomConvNet(
            input_shape=DEFAULT_INPUT_SHAPE,
            conv_layers=DEFAULT_CONV_LAYERS_PARAM,  # Ensures conv dropout layers
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=dropout_rate,
            num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,  # Ensures FC dropout layers
        )
        dummy_input = torch.randn(BATCH_SIZE, *DEFAULT_INPUT_SHAPE)

        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training
                assert module.p == dropout_rate

        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert not module.training

        output_eval1 = model(dummy_input)
        output_eval2 = model(dummy_input)
        assert torch.equal(output_eval1, output_eval2), "Outputs should be identical in eval mode"

    def test_different_input_hw(self):
        """Test model with different input height and width."""
        input_shape = (3, 128, 128)  # Different H, W
        model = CustomConvNet(
            input_shape=input_shape,
            conv_layers=DEFAULT_CONV_LAYERS_PARAM,
            num_classes=DEFAULT_NUM_CLASSES,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        )
        dummy_input = torch.randn(BATCH_SIZE, *input_shape)
        output = model(dummy_input)
        assert output.shape == (BATCH_SIZE, DEFAULT_NUM_CLASSES)
