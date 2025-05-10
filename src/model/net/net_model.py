from typing import Protocol

import torch


class Net(Protocol):
    def forward(self, x) -> torch.Tensor: ...
