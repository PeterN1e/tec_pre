import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelCanonConfig, TrainConfig, DatasetConfig
from common.DataFusion import FilmFusion

cfg_model = ModelCanonConfig()
cfg_train = TrainConfig()
cfg_dataset = DatasetConfig()


class ModelCanon(nn.Module):
    def __init__(
        self,
        input_length=cfg_train.input_length,
        output_length=cfg_train.output_length,
        aux_dim=cfg_dataset.aux_dim,
        height=71,
        width=73,
        d_model=cfg_model.d_model,
        n_heads=cfg_model.n_heads,
        e_layers=cfg_model.e_layers,
        decoder_layers=cfg_model.decoder_layers,
        d_ff=cfg_model.d_ff,
        dropout=cfg_model.dropout,
        patch_size=cfg_model.patch_size,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.height = height
        self.width = width
        self.d_model = d_model
        self.patch_size = patch_size

        self.pad_h = (-height) % patch_size
        self.pad_w = (-width) % patch_size
        self.grid_h = (height + self.pad_h) // patch_size
        self.grid_w = (width + self.pad_w) // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.film = FilmFusion(aux_dim=aux_dim, channel=d_model, out_dim=3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.pred_queries = nn.Parameter(torch.randn(output_length, d_model) * (d_model ** -0.5))

        self.head = nn.Linear(d_model, height * width)

    def _tokenize(self, tec, aux):
        B, T, H, W = tec.shape
        assert (H, W) == (self.height, self.width), (
            f"Expected input grid ({self.height}, {self.width}), got ({H}, {W})"
        )
        x = F.pad(tec, (0, self.pad_w, 0, self.pad_h))
        x = x.reshape(B * T, 1, self.grid_h * self.patch_size, self.grid_w * self.patch_size)
        x = self.patch_embed(x)
        x = x.reshape(B, T, self.d_model, self.grid_h, self.grid_w)
        x = self.film(x, aux)
        x = x.reshape(B, T, self.num_patches, self.d_model)
        x = x + self.pos_embed
        return self.dropout(x)

    def _temporal_encode(self, x):
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = self.temporal_encoder(x)
        return x.reshape(B, N, T, D).permute(0, 2, 1, 3)

    def forward_delta(self, tec, aux):
        B = tec.shape[0]
        x = self._tokenize(tec, aux)
        x = self._temporal_encode(x)
        queries = self.pred_queries.unsqueeze(0).expand(B, -1, -1)
        memory = x.reshape(B, self.input_length * self.num_patches, self.d_model)
        dec = self.decoder(queries, memory)
        delta = self.head(dec).view(B, self.output_length, self.height, self.width)
        return delta

    def forward(self, tec, aux):
        """Reconstruct day 4 as day 3 same-hour baseline + predicted delta."""
        delta = self.forward_delta(tec, aux)
        return tec[:, -12:, :, :] + delta


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ModelCanon().to(cfg_train.device)
    tec = torch.randn(2, cfg_train.input_length, 71, 73, device=cfg_train.device)
    aux = torch.randn(2, cfg_train.input_length, cfg_dataset.aux_dim, device=cfg_train.device)

    pred = model(tec, aux)
    delta = model.forward_delta(tec, aux)
    print("forward       ->", tuple(pred.shape))
    print("forward_delta ->", tuple(delta.shape))
    print("params        ->", f"{sum(p.numel() for p in model.parameters()):,}")

    loss = pred.mean()
    loss.backward()
    print("backward      -> OK")
