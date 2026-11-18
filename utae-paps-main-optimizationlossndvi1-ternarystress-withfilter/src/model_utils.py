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
        elif config.model == "uconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        elif config.model == "buconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
    elif mode == "panoptic":
        from src.panoptic import paps
        if config.backbone == "utae":
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
                encoder=True,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.backbone == "uconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=True,
                padding_mode="zeros",
                pad_value=0,
            )
        else:
            raise NotImplementedError

        model = paps.PaPs(
            encoder=model,
            num_classes=config.num_classes,
            shape_size=config.shape_size,
            mask_conv=config.mask_conv,
            min_confidence=config.min_confidence,
            min_remain=config.min_remain,
            mask_threshold=config.mask_threshold,
        )
        return model
    elif mode == "forecasting":
        from src.backbones.utae import UTAE
        import torch.nn as nn
        import torch
        import torch.nn.functional as F

        class TemporalAttentionModule(nn.Module):
            """Modulo di attention temporale per sfruttare l'alta autocorrelazione"""
            def __init__(self, in_channels, n_heads=8, d_model=256):
                super(TemporalAttentionModule, self).__init__()
                self.n_heads = n_heads
                self.d_model = d_model
                
                # Proiezioni per Q, K, V
                self.query_proj = nn.Linear(in_channels, d_model)
                self.key_proj = nn.Linear(in_channels, d_model)
                self.value_proj = nn.Linear(in_channels, d_model)
                
                # Multi-head attention
                self.multihead_attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=0.1,
                    batch_first=True
                )
                
                # Output projection
                self.out_proj = nn.Linear(d_model, in_channels)
                self.layer_norm = nn.LayerNorm(in_channels)
                
            def forward(self, x):
                # x: (B, C, H, W)
                B, C, H, W = x.shape
                
                # Reshape per attention: (B, H*W, C)
                x_flat = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
                
                # Proiezioni
                q = self.query_proj(x_flat)  # (B, H*W, d_model)
                k = self.key_proj(x_flat)    # (B, H*W, d_model)
                v = self.value_proj(x_flat)  # (B, H*W, d_model)
                
                # Multi-head attention
                attn_output, _ = self.multihead_attn(q, k, v)  # (B, H*W, d_model)
                
                # Output projection
                output = self.out_proj(attn_output)  # (B, H*W, C)
                
                # Residual connection + Layer norm
                output = self.layer_norm(output + x_flat)
                
                # Reshape back: (B, C, H, W)
                output = output.transpose(1, 2).view(B, C, H, W)
                
                return output

        class ResidualBlock(nn.Module):
            """Blocco residuale per la forecasting head"""
            def __init__(self, in_channels, out_channels):
                super(ResidualBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                # Skip connection
                if in_channels != out_channels:
                    self.skip = nn.Conv2d(in_channels, out_channels, 1)
                else:
                    self.skip = nn.Identity()
                
            def forward(self, x):
                residual = self.skip(x)
                
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                
                return F.relu(out + residual)

        class EnhancedForecastingHead(nn.Module):
            """Testa di forecasting migliorata con temporal attention e residual blocks"""
            def __init__(self, in_channels, forecast_steps, ndvi_stress_threshold=0.4):
                super(EnhancedForecastingHead, self).__init__()
                
                # Temporal Attention per sfruttare l'autocorrelazione +302%
                self.temporal_attention = TemporalAttentionModule(in_channels, n_heads=8, d_model=256)
                
                # Feature enhancement
                self.feature_enhance = nn.Sequential(
                    nn.Conv2d(in_channels, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU()
                )
                
                # NDVI Map Branch con residual blocks
                self.ndvi_residual1 = ResidualBlock(128, 96)
                self.ndvi_residual2 = ResidualBlock(96, 64)
                self.ndvi_residual3 = ResidualBlock(64, 48)
                
                self.ndvi_final = nn.Sequential(
                    nn.Conv2d(48, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, forecast_steps, 1),
                    nn.Tanh()  # Output tra -1 e 1 per NDVI
                )
                
                # NDVI Mean Branch migliorato
                self.ndvi_mean = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, forecast_steps),
                    nn.Tanh()  # Output tra -1 e 1 per NDVI
                )
                
                # Stress Global Branch
                self.stress_global = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, forecast_steps),
                    nn.Sigmoid()  # Output tra 0 e 1 per probabilità stress
                )
                
                self.ndvi_stress_threshold = ndvi_stress_threshold
                
            def forward(self, x):
                # x: (B, C, H, W)
                
                # Apply temporal attention per sfruttare autocorrelazione
                x_attended = self.temporal_attention(x)
                
                # Feature enhancement
                features = self.feature_enhance(x_attended)
                
                # NDVI Map con residual blocks
                ndvi_features = self.ndvi_residual1(features)
                ndvi_features = self.ndvi_residual2(ndvi_features)
                ndvi_features = self.ndvi_residual3(ndvi_features)
                ndvi_map = self.ndvi_final(ndvi_features)  # (B, forecast_steps, H, W)
                
                # NDVI Mean
                ndvi_mean = self.ndvi_mean(features)  # (B, forecast_steps)
                ndvi_mean = ndvi_mean.unsqueeze(-1)  # (B, forecast_steps, 1)
                
                # Stress globale
                stress_global = self.stress_global(features)  # (B, forecast_steps)
                stress_global = stress_global.unsqueeze(-1)  # (B, forecast_steps, 1)
                
                # Stress map basato su NDVI map
                stress_map = torch.sigmoid((self.ndvi_stress_threshold - ndvi_map) * 5.0)
                
                return ndvi_map, ndvi_mean, stress_global, stress_map

        # Crea UTAE backbone con più feature channels
        backbone = UTAE(
            input_dim=10,
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=[32, 64],  # Aumentato da 32 a 64 channels
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
        
        # Crea testa di forecasting migliorata
        head = EnhancedForecastingHead(64, config.forecast_steps, config.ndvi_stress_threshold)
        
        class FullModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head
                
            def forward(self, x, batch_positions=None, forecast=False):
                features = self.backbone(x, batch_positions=batch_positions)
                return self.head(features)
                
        return FullModel(backbone, head)
    else:
        raise NotImplementedError