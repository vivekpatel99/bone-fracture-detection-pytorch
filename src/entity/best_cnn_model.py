from pydantic import BaseModel


class ModelConfig(BaseModel):
    input_shape: list | tuple
    batch_size: int
    conv_layers: int
    dropout_rate: float
    num_classes: int
    num_hidden_layers: int


class BestCNNModel(BaseModel):
    check_point_path: str
    f1_score: float
    accuracy: float
    loss: float
    params: ModelConfig
