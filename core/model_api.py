from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ForecastModel(nn.Module):
    """Uniform interface expected by trainers and evaluation code."""

    def forward(self, tec: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_forward(
        self,
        tec: torch.Tensor,
        aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self(tec, aux), None


class TecAuxForecastModel(ForecastModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tec: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        output = self.model(tec, aux)
        if isinstance(output, tuple):
            return output[0]
        return output

    def train_forward(
        self,
        tec: torch.Tensor,
        aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        train_forward = getattr(self.model, "train_forward", None)
        if train_forward is not None:
            output = train_forward(tec, aux)
            if isinstance(output, tuple):
                return output[0], output[1] if len(output) > 1 else None
            return output, None
        output = self.model(tec, aux)
        if isinstance(output, tuple):
            return output[0], output[1] if len(output) > 1 else None
        return output, None


class EDCGConvLSTMAdapter(TecAuxForecastModel):
    def forward(self, tec: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        del aux
        output = self.model(tec.unsqueeze(2))
        return output.squeeze(2)

    def train_forward(
        self,
        tec: torch.Tensor,
        aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward(tec, aux), None

