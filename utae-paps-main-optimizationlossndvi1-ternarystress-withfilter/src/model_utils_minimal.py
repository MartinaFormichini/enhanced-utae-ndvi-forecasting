from src.backbones import utae, unet3d, convlstm, convgru, fpn


def get_model(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.model == "unet3d":
            model = unet3d.UNet3D(
                in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=10,
                num_classes=config.num_classes,
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        else:
            raise ValueError("Unknown model {}.".format(config.model))
    elif mode == "panoptic":
        from src.panoptic import paps
        model = paps.PaPs(
            encoder=config.encoder,
            decoder=config.decoder,
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=config.out_conv,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            num_classes=config.num_classes,
            num_groups=config.num_groups,
            padding_mode=config.padding_mode,
            pad_value=config.pad_value,
            mask_threshold=config.mask_threshold,
        )
        return model
    elif mode == "forecasting":
        from src.backbones.utae import UTAE
        import torch.nn as nn
        import torch
        import torch.nn.functional as F

        class MinimalSpatialAttention(nn.Module):
            """Minimal Spatial Attention (MSA): attention spaziale ottimizzata per migliori RMSE/MAE"""
            def __init__(self, in_channels, n_heads=4, d_model=64, dropout=0.1, downsample_factor=4):
                super(MinimalSpatialAttention, self).__init__()
                self.n_heads = n_heads
                self.d_model = d_model
                self.downsample_factor = downsample_factor
                
                # Downsampling per ridurre memoria
                self.downsample = nn.AdaptiveAvgPool2d(
                    (128 // downsample_factor, 128 // downsample_factor)
                )
                self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
                
                # Progetti leggeri per Q, K, V con regolarizzazione
                self.qkv_proj = nn.Linear(in_channels, d_model * 3)
                
                # Multi-head attention con dropout maggiore per regolarizzazione
                self.multihead_attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,  # Aumentato da 0.05 a 0.1
                    batch_first=True
                )
                
                # Output projection con dropout
                self.out_proj = nn.Linear(d_model, in_channels)
                self.dropout = nn.Dropout(dropout)
                self.layer_norm = nn.LayerNorm(in_channels)
                
            def forward(self, x):
                # x: (B, C, H, W)
                B, C, H, W = x.shape
                
                # Downsample per ridurre memoria: 128x128 -> 32x32
                x_down = self.downsample(x)  # (B, C, H//4, W//4)
                _, _, H_down, W_down = x_down.shape
                
                # Flatten spatial dimensions: (B, C, H_down*W_down) -> (B, H_down*W_down, C)
                x_flat = x_down.view(B, C, H_down*W_down).transpose(1, 2)  # (B, H_down*W_down, C)
                
                # Proiezioni Q, K, V
                qkv = self.qkv_proj(x_flat)  # (B, H_down*W_down, 3*d_model)
                q, k, v = qkv.chunk(3, dim=-1)  # Ognuno (B, H_down*W_down, d_model)
                
                # Multi-head attention (ora su 32x32=1024 invece di 128x128=16384)
                attn_output, _ = self.multihead_attn(q, k, v)  # (B, H_down*W_down, d_model)
                
                # Output projection con dropout
                output = self.out_proj(attn_output)  # (B, H_down*W_down, C)
                output = self.dropout(output)
                
                # Residual connection + Layer norm
                output = self.layer_norm(output + x_flat)
                
                # Reshape back: (B, H_down*W_down, C) -> (B, C, H_down, W_down)
                output = output.transpose(1, 2).view(B, C, H_down, W_down)
                
                # Upsample back to original size
                output = self.upsample(output)  # (B, C, H, W)
                
                # Residual connection with original input
                return output + x

        class OptimizedForecastingHead(nn.Module):
            """Forecasting head ottimizzato per RMSE/MAE migliorati"""
            def __init__(self, in_channels, forecast_steps, ndvi_stress_threshold=0.4, l1_reg=0.001):
                super(OptimizedForecastingHead, self).__init__()
                
                self.l1_reg = l1_reg  # Fattore di regolarizzazione L1
                self.forecast_steps = forecast_steps
                self.num_stress_classes = 3
                
                # Minimal Spatial Attention (MSA) con dropout maggiore e ottimizzazione memoria
                # (attributo chiamato ancora temporal_attention per compatibilità con i checkpoint salvati)
                self.temporal_attention = MinimalSpatialAttention(in_channels, n_heads=4, d_model=64, dropout=0.1, downsample_factor=4)
                
                # NDVI Map Branch - con regolarizzazione e scaling learnable
                self.ndvi_map_backbone = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),  # Aggiungi dropout spaziale
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Dropout2d(0.05),  # Dropout più leggero
                    nn.Conv2d(32, forecast_steps, 1),
                    nn.Tanh()  # Output tra -1 e 1 per NDVI
                )
                
                # Learnable scaling per NDVI map:
                # inizializzare a 1.0 evita compressione artificiale dell'ampiezza NDVI.
                self.ndvi_map_scale = nn.Parameter(torch.tensor(1.0))
                
                # NDVI Mean Branch - con regolarizzazione
                self.ndvi_mean_backbone = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # Aggiungi dropout
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.05),  # Dropout più leggero
                    nn.Linear(32, forecast_steps),
                    nn.Tanh()  # Output tra -1 e 1 per NDVI
                )
                
                # Learnable scaling per NDVI mean (stesso principio della mappa).
                self.ndvi_mean_scale = nn.Parameter(torch.tensor(1.0))
                
                # Stress Global Branch - ternario (healthy / warning / critical)
                self.stress_global = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, forecast_steps * self.num_stress_classes),
                )

                # Stress Map Branch - ternario pixel-wise
                self.stress_map_head = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, forecast_steps * self.num_stress_classes, 1),
                )
                
                self.ndvi_stress_threshold = ndvi_stress_threshold
                
            def get_l1_regularization(self):
                """Calcola la regolarizzazione L1 sui pesi"""
                l1_loss = 0
                for param in self.parameters():
                    if param.requires_grad and len(param.shape) > 1:  # Solo pesi, non bias
                        l1_loss += torch.sum(torch.abs(param))
                return self.l1_reg * l1_loss
                
            def forward(self, x):
                # Applica la Minimal Spatial Attention (MSA)
                x_attended = self.temporal_attention(x)
                
                # NDVI map con scaling learnable
                ndvi_map_raw = self.ndvi_map_backbone(x_attended)  # (B, forecast_steps, H, W)
                ndvi_map = ndvi_map_raw * torch.clamp(self.ndvi_map_scale, 0.7, 1.3)
                
                # NDVI mean con scaling learnable
                ndvi_mean_raw = self.ndvi_mean_backbone(x_attended)  # (B, forecast_steps)
                ndvi_mean = ndvi_mean_raw * torch.clamp(self.ndvi_mean_scale, 0.7, 1.3)
                ndvi_mean = ndvi_mean.unsqueeze(-1)  # (B, forecast_steps, 1)
                
                # Stress globale ternario: (B, forecast_steps, 3)
                stress_global_logits = self.stress_global(x_attended).view(
                    x_attended.shape[0], self.forecast_steps, self.num_stress_classes
                )
                # Ritorna logits: la loss applica softmax/cross_entropy in modo numericamente stabile.
                stress_global = stress_global_logits

                # Stress map ternario: (B, forecast_steps, H, W, 3)
                stress_map_logits = self.stress_map_head(x_attended)
                B, _, H, W = stress_map_logits.shape
                stress_map = stress_map_logits.view(
                    B, self.forecast_steps, self.num_stress_classes, H, W
                ).permute(0, 1, 3, 4, 2).contiguous()
                
                return ndvi_map, ndvi_mean, stress_global, stress_map

        class FullModel(nn.Module):
            """Modello completo con UTAE backbone + Optimized Forecasting Head"""
            def __init__(self, config):
                super(FullModel, self).__init__()
                
                # UTAE Backbone (mantenuto identico)
                self.backbone = UTAE(
                    input_dim=10,
                    encoder_widths=config.encoder_widths,
                    decoder_widths=config.decoder_widths,
                    out_conv=config.out_conv,
                    str_conv_k=config.str_conv_k,
                    str_conv_s=config.str_conv_s,
                    str_conv_p=config.str_conv_p,
                    agg_mode=config.agg_mode,
                    encoder_norm=config.encoder_norm,
                    n_head=config.n_head,
                    d_model=config.d_model,
                    d_k=config.d_k,
                    encoder=True,
                    return_maps=True,
                    pad_value=config.pad_value,
                    padding_mode=config.padding_mode,
                )
                
                # Optimized Forecasting Head
                in_channels = config.decoder_widths[0]  # 32
                self.head = OptimizedForecastingHead(
                    in_channels=in_channels,
                    forecast_steps=config.forecast_steps,
                    ndvi_stress_threshold=config.ndvi_stress_threshold,
                    l1_reg=0.001  # Regolarizzazione L1 delicata
                )
                
            def forward(self, x, batch_positions=None, forecast=True):
                # Extract features with UTAE backbone
                backbone_output = self.backbone(x, batch_positions=batch_positions, return_att=False)
                
                # Handle potential tuple output from backbone
                if isinstance(backbone_output, tuple):
                    features = backbone_output[0]  # Take the first element (features)
                else:
                    features = backbone_output
                
                # Apply forecasting head
                ndvi_map, ndvi_mean, stress_global, stress_map = self.head(features)
                
                return ndvi_map, ndvi_mean, stress_global, stress_map
                
            def get_regularization_loss(self):
                """Restituisce la loss di regolarizzazione"""
                return self.head.get_l1_regularization()

        # Create and return the model
        model = FullModel(config)
        return model
    else:
        raise ValueError("Unknown mode {}.".format(mode)) 