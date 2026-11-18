"""
ConvLSTM backbone + multi-task forecasting head (NDVI map/mean + stress ternario).

Stessa interfaccia del modello UTAE forecasting:
  forward(x, batch_positions=..., forecast=True) -> (ndvi_map, ndvi_mean, stress_global, stress_map)
"""
from src.backbones import convlstm
import torch
import torch.nn as nn


class MultitaskForecastingHead(nn.Module):
    """Testa multi-task allineata a model_utils_minimal (senza MSA: il tempo è nel ConvLSTM)."""

    def __init__(self, in_channels: int, forecast_steps: int, ndvi_stress_threshold: float = 0.4):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.num_stress_classes = 3
        self.ndvi_stress_threshold = ndvi_stress_threshold

        self.ndvi_map_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, forecast_steps, 1),
            nn.Tanh(),
        )
        self.ndvi_map_scale = nn.Parameter(torch.tensor(1.0))
        self.ndvi_mean_backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, forecast_steps),
            nn.Tanh(),
        )
        self.ndvi_mean_scale = nn.Parameter(torch.tensor(1.0))
        self.stress_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, forecast_steps * self.num_stress_classes),
        )
        self.stress_map_head = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, forecast_steps * self.num_stress_classes, 1),
        )

    def forward(self, x: torch.Tensor):
        ndvi_map = self.ndvi_map_backbone(x) * torch.clamp(self.ndvi_map_scale, 0.7, 1.3)
        ndvi_mean = (
            self.ndvi_mean_backbone(x) * torch.clamp(self.ndvi_mean_scale, 0.7, 1.3)
        ).unsqueeze(-1)

        stress_global_logits = self.stress_global(x).view(
            x.shape[0], self.forecast_steps, self.num_stress_classes
        )
        stress_map_logits = self.stress_map_head(x)
        b, _, h, w = stress_map_logits.shape
        stress_map = stress_map_logits.view(
            b, self.forecast_steps, self.num_stress_classes, h, w
        ).permute(0, 1, 3, 4, 2).contiguous()

        return ndvi_map, ndvi_mean, stress_global_logits, stress_map


class ConvLSTMForecastEncoder(nn.Module):
    """Serie S2 (B,T,C,H,W) -> mappa spaziale di feature (ultimo stato ConvLSTM)."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 160,
        input_size=(128, 128),
        kernel_size=(3, 3),
        pad_value: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        if bidirectional:
            self.temporal_encoder = convlstm.BConvLSTM(
                input_size=input_size,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
            )
            self.out_channels = 2 * hidden_dim
        else:
            self.temporal_encoder = convlstm.ConvLSTM(
                input_size=input_size,
                input_dim=input_dim,
                hidden_dim=[hidden_dim],
                kernel_size=[kernel_size],
                return_all_layers=False,
            )
            self.out_channels = hidden_dim

    def _pad_mask(self, x: torch.Tensor):
        pad_mask = (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        return pad_mask if pad_mask.any() else None

    def forward(self, x: torch.Tensor, batch_positions=None) -> torch.Tensor:
        pad_mask = self._pad_mask(x)
        if self.bidirectional:
            return self.temporal_encoder(x, pad_mask=pad_mask)
        _, states = self.temporal_encoder(x, pad_mask=pad_mask)
        return states[0][1]


class ConvLSTMForecastingModel(nn.Module):
    """BConvLSTM (o ConvLSTM) + testa NDVI/stress."""

    def __init__(self, config):
        super().__init__()
        hidden_dim = int(getattr(config, "convlstm_hidden_dim", 160))
        bidirectional = bool(getattr(config, "convlstm_bidirectional", True))
        head_channels = int(getattr(config, "convlstm_head_channels", 128))

        self.encoder = ConvLSTMForecastEncoder(
            input_dim=10,
            hidden_dim=hidden_dim,
            input_size=(128, 128),
            kernel_size=(3, 3),
            pad_value=getattr(config, "pad_value", 0.0),
            bidirectional=bidirectional,
        )
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels, head_channels, 3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
        )
        self.head = MultitaskForecastingHead(
            head_channels,
            config.forecast_steps,
            ndvi_stress_threshold=getattr(config, "ndvi_stress_threshold", 0.4),
        )

    def forward(self, x, batch_positions=None, forecast=True):
        features = self.encoder(x, batch_positions=batch_positions)
        features = self.feat_proj(features)
        return self.head(features)


def get_model(config, mode="forecasting"):
    if mode == "forecasting":
        return ConvLSTMForecastingModel(config)
    raise ValueError(
        f"model_utils_convlstm supports only mode='forecasting', got '{mode}'. "
        "Use model_utils_minimal for panoptic/semantic modes."
    )
