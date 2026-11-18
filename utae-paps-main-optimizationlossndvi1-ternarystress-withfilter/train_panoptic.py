"""
Main script for panoptic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import inspect
import json
import math
import os
import traceback
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
import torch.nn.functional as F

import src.model_utils_minimal as model_utils
from src.dataset import PASTIS_Dataset
from src.learning.weight_init import weight_init
from src.utils import pad_collate, get_ntrainparams

parser = argparse.ArgumentParser()
# PaPs Parameters
## Architecture Hyperparameters
parser.add_argument("--shape_size", default=16, type=int, help="Shape size for PaPs")
parser.add_argument(
    "--no_mask_conv",
    dest="mask_conv",
    action="store_false",
    help="With this flag no residual CNN is used after combination of global saliency and local shape.",
)
parser.add_argument(
    "--backbone",
    default="utae",
    type=str,
    help="Backbone encoder for PaPs (utae or uconvlstm)",
)

## Losses & metrics
parser.add_argument(
    "--l_center", default=1, type=float, help="Coefficient for centerness loss"
)
parser.add_argument("--l_size", default=1, type=float, help="Coefficient for size loss")
parser.add_argument(
    "--l_shape", default=1, type=float, help="Coefficient for shape loss"
)
parser.add_argument(
    "--l_class", default=1, type=float, help="Coefficient for class loss"
)
parser.add_argument(
    "--beta", default=4, type=float, help="Beta parameter for centerness loss"
)
parser.add_argument(
    "--no_autotune",
    dest="autotune",
    action="store_false",
    help="If this flag is used the confidence threshold for the pseudo-nms will NOT be tuned automatically on the validation set",
)
parser.add_argument(
    "--no_supmax",
    dest="supmax",
    action="store_false",
    help="If this flag is used, ALL local maxima are supervised (and not just the more confident center per ground truth object)",
)
parser.add_argument(
    "--warmup",
    default=5,
    type=int,
    help="Number of epochs to do with only the centerness loss as supervision.",
)
parser.add_argument(
    "--val_metrics_only",
    dest="val_metrics_only",
    action="store_true",
    help="If true, panoptic metrics are computed only on validation and test epochs.",
)
parser.add_argument(
    "--val_every",
    default=5,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=65,
    type=int,
    help="Do validation only after that many epochs.",
)
parser.add_argument(
    "--save_every_n_epochs",
    default=5,
    type=int,
    help="(forecasting) Salva model_epoch_*.pth.tar ogni N epoche; ultima epoca sempre salvata.",
)

## Thresholds
parser.add_argument(
    "--min_remain",
    default=0.5,
    type=float,
    help="Minimum remain fraction for the pseudo-nms.",
)
parser.add_argument(
    "--mask_threshold",
    default=0.4,
    type=float,
    help="Binary threshold for instancce masks.",
)
parser.add_argument(
    "--min_confidence",
    default=0.2,
    type=float,
    help="Minimum confidence for the pseudo-nms (tuned automatically by default)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 20]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)

## Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument(
    "--mono_date",
    default=None,
    type=str,
    help="If one date is specified, model is training on a single date.",
)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--void_label", default=19, type=int)
parser.add_argument("--background_label", default=0, type=int)
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of data loading workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)

parser.add_argument("--mode", default="panoptic", type=str, help="Modalità di training: panoptic, semantic, forecasting")
parser.add_argument("--plot_val_loss", action="store_true", help="Plotta la curva di validation loss da trainlog.json al termine")
parser.add_argument("--forecasting_decoder_type", default="gru", type=str, help="Tipo di decoder per la testa di forecasting: 'gru' o 'transformer'")
parser.add_argument("--forecasting_hidden_dim", default=128, type=int, help="Dimensione hidden del decoder di forecasting")
parser.add_argument("--forecasting_n_layers", default=1, type=int, help="Numero di layer del decoder di forecasting")
parser.add_argument("--forecasting_forecast_steps", default=2, type=int, help="Numero di passi di forecasting (forecast steps)")
parser.add_argument("--valid_patches_path", default=None, type=str, help="Path to file with valid patch IDs for forecasting")
parser.add_argument("--input_steps", default=4, type=int, help="Numero di step temporali di input")
parser.add_argument("--forecast_steps", default=2, type=int, help="Numero di step temporali da prevedere")
parser.add_argument("--forecast_lag_days", default=7.0, type=float, help="Lag temporale in giorni per ciascun forecast step (default: 7)")
parser.add_argument("--ndvi_stress_threshold", default=0.4, type=float, help="Soglia alta NDVI per stress ternario (healthy >= 0.4)")
parser.add_argument("--lr_scheduler", default=None, type=str, help="Learning rate scheduler: 'step' or 'cosine'")
parser.add_argument("--lr_step_size", default=50, type=int, help="Step size for learning rate scheduler")
parser.add_argument("--lr_gamma", default=0.5, type=float, help="Gamma for learning rate scheduler")
parser.add_argument("--weight_ndvi_map", default=1.0, type=float, help="Peso λ1 per NDVI map loss (default: 1.0, come nell'articolo)")
parser.add_argument("--weight_ndvi_mean", default=0.5, type=float, help="Peso λ2 per NDVI mean loss (default: 0.5, come nell'articolo)")
parser.add_argument("--weight_stress", default=0.3, type=float, help="Peso λ3 = λ4 per le due stress loss, globale e pixel-wise (default: 0.3, come nell'articolo)")
parser.add_argument(
    "--ndvi_only_loss",
    action="store_true",
    help="Pretrain NDVI: ottimizza solo loss_ndvi_map + loss_ndvi_mean. Stress disattivato in loss e validazione.",
)
parser.add_argument("--ndvi_loss_type", default="mse", type=str, choices=["mse", "mse_mae", "mae", "huber"], help="Tipo di loss per NDVI: 'mse' (solo MSE), 'mse_mae' (70%% MSE + 30%% MAE), 'mae' (solo MAE), 'huber' (Huber loss)")
parser.add_argument(
    "--resume_path",
    default="",
    type=str,
    help="Checkpoint path per riprendere il training forecasting (state_dict + optimizer).",
)
parser.add_argument(
    "--use_focal_stress",
    action="store_true",
    help="Usa modulazione focal (multiclass) sulla CrossEntropy dello stress (globale e map)",
)
parser.add_argument(
    "--focal_gamma",
    default=2.0,
    type=float,
    help="Gamma focal per lo stress multiclass (default: 2.0). Disattiva focal con --focal_gamma 0",
)
parser.add_argument(
    "--focal_alpha",
    default=1.0,
    type=float,
    help="(Deprecato) Non usato nella loss multiclass attuale; mantenuto solo per compatibilita` CLI.",
)
parser.add_argument(
    "--stress_class_weight_mode",
    default="auto",
    type=str,
    choices=["auto", "none"],
    help="Pesi classe per stress ternario: 'auto' stima frequenze su train set, 'none' uniforma",
)
parser.add_argument(
    "--stress_weight_estimation_max_samples",
    default=0,
    type=int,
    help="Se >0, stima pesi stress solo sui primi N campioni del train set (piu` veloce). 0 = usa tutto il train set.",
)
parser.add_argument(
    "--stress_label_smoothing",
    default=0.0,
    type=float,
    help="Label smoothing per CrossEntropy dello stress (0.0 disabilitato). Tipico: 0.02-0.05",
)
parser.add_argument(
    "--use_cloud_mask_train",
    action="store_true",
    help="Nel training forecasting maschera NDVI map/mean loss sui pixel clear usando una U-Net cloud detector congelata.",
)
parser.add_argument(
    "--cloud_detector_weights",
    default="",
    type=str,
    help="Checkpoint della U-Net cloud detector usata solo per il masked training forecasting.",
)
parser.add_argument(
    "--cloud_threshold",
    default=0.4,
    type=float,
    help="Soglia su cloud probability per marcare un pixel come non valido nel masked training (default: 0.4).",
)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]

parser.set_defaults(
    cache=False, mask_conv=True, supmax=True, autotune=True, val_metrics_only=False
)


def iterate(
    model,
    data_loader,
    criterion,
    config,
    optimizer=None,
    mode="train",
    device=None,
    compute_metrics=True,
    heatmap_only=False,
    autotune=False,
):
    if compute_metrics:
        from src.panoptic.metrics import PanopticMeter

    loss_meter = tnt.meter.AverageValueMeter()
    loss_center_meter = tnt.meter.AverageValueMeter()
    loss_size_meter = tnt.meter.AverageValueMeter()
    loss_shape_meter = tnt.meter.AverageValueMeter()
    loss_class_meter = tnt.meter.AverageValueMeter()

    if compute_metrics:
        pano_meter = PanopticMeter(
            num_classes=config.num_classes, void_label=config.void_label
        )

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        if device is not None:
            batch = recursive_todevice(batch, device)
        (x, dates), y = batch

        if mode != "train":
            with torch.no_grad():
                predictions = model(
                    x,
                    batch_positions=dates,
                    pseudo_nms=compute_metrics,
                    heatmap_only=heatmap_only,
                )
        else:
            zones = y[:, :, :, 2] if config.supmax else None
            optimizer.zero_grad()
            predictions = model(
                x,
                batch_positions=dates,
                pseudo_nms=compute_metrics,
                zones=zones,
                heatmap_only=heatmap_only,
            )
        loss = criterion(predictions, y, heatmap_only=heatmap_only)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if compute_metrics:
            pano_meter.add(predictions, y)

        ce, si, sh, cl = criterion.value
        loss_center_meter.add(ce)
        loss_size_meter.add(si)
        loss_shape_meter.add(sh)
        loss_class_meter.add(cl)
        loss_meter.add(float(loss.item()))

        if (i + 1) % config.display_step == 0:
            if compute_metrics:
                sq, rq, pq = pano_meter.value()
                print(
                    "Step [{}/{}], Loss: {:.4f}, SQ {:.1f},  RQ {:.1f}  , PQ {:.1f}".format(
                        i + 1,
                        len(data_loader),
                        loss_meter.value()[0],
                        sq * 100,
                        rq * 100,
                        pq * 100,
                    )
                )
            else:
                print(
                    "Step [{}/{}], Loss: {:.4f} ".format(
                        i + 1, len(data_loader), loss_meter.value()[0]
                    )
                )

    if autotune:
        thrsh = tune_threshold(criterion.predicted_confidences, criterion.achieved_ious)
        model.min_confidence = torch.tensor(
            [thrsh], device=next(model.parameters()).device
        )
        config.min_confidence = thrsh

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))

    metrics = {
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_center_loss".format(mode): loss_center_meter.value()[0],
        "{}_size_loss".format(mode): loss_size_meter.value()[0],
        "{}_shape_loss".format(mode): loss_shape_meter.value()[0],
        "{}_class_loss".format(mode): loss_class_meter.value()[0],
        "{}_epoch_time".format(mode): total_time,
    }
    if compute_metrics:
        SQ, RQ, PQ = pano_meter.value()
        metrics.update(
            {
                "{}_SQ".format(mode): float(SQ),
                "{}_RQ".format(mode): float(RQ),
                "{}_PQ".format(mode): float(PQ),
            }
        )
    if mode == "test":
        return metrics, pano_meter.get_table()
    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config.res_dir, exist_ok=True)
    # Crea solo la directory per il fold 1
    os.makedirs(os.path.join(config.res_dir, "Fold_1"), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, table, config):
    with open(
        os.path.join(config.res_dir, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)
    np.save(os.path.join(config.res_dir, "Fold_{}".format(fold), "test_tables"), table)


def plot_validation_loss(fold_dir, title_suffix=""):
    """Legge Fold_{k}/trainlog.json e plotta la curva di val_loss nel tempo (epoche)."""
    log_path = os.path.join(fold_dir, "trainlog.json")
    if not os.path.exists(log_path):
        print(f"[WARNING] plot_validation_loss: file non trovato: {log_path}")
        return
    try:
        with open(log_path, "r") as f:
            trainlog = json.load(f)
    except Exception as e:
        print(f"[WARNING] plot_validation_loss: errore nel leggere {log_path}: {e}")
        return
    if not isinstance(trainlog, dict) or len(trainlog) == 0:
        print(f"[WARNING] plot_validation_loss: trainlog vuoto o formato inatteso: {log_path}")
        return
    # trainlog: {epoch(str or int): { 'val_loss': ..., ... }}
    epochs = []
    vals = []
    for ep_str, metrics in trainlog.items():
        try:
            ep = int(ep_str)
        except Exception:
            continue
        if not isinstance(metrics, dict):
            continue
        if "val_loss" in metrics:
            epochs.append(ep)
            vals.append(metrics["val_loss"])
    if len(epochs) == 0:
        print(f"[INFO] plot_validation_loss: nessuna chiave 'val_loss' trovata in {log_path}")
        return
    # Ordina per epoca
    order = np.argsort(np.array(epochs))
    epochs_sorted = np.array(epochs)[order]
    vals_sorted = np.array(vals, dtype=float)[order]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_sorted, vals_sorted, label="val_loss", linewidth=2.5, color="tab:blue")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Validation Loss", fontsize=16)
    title = "Validation loss" + (f" - {title_suffix}" if title_suffix else "")
    plt.title(title, fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def tune_threshold(confidences, target):
    t = target.squeeze()
    t = t > 0.5
    p = confidences.squeeze()
    best_score = 0
    best_threshold = 0
    print("Tuning confidence threshold . . . ")
    for ct in np.arange(0, 1, 0.01):
        TP = ((p > ct) * t).sum()
        FP = ((p > ct) * (~t)).sum()
        FN = ((p < ct) * t).sum()
        score = TP / (TP + 0.5 * (FP + FN))
        if score > best_score:
            best_threshold = ct
            best_score = score

    print(
        "Best F-Score : {:.2f} / Threshold : {:.2f}".format(best_score, best_threshold)
    )
    return best_threshold


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = torch.nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


def _estimate_stress_class_weights_from_dataset(dt_train, max_samples: int = 0):
    """
    Stima pesi classe per stress ternario usando:
    - stress_target (media NDVI per step)
    - classi pixel-wise da ndvi_target (stesse soglie del training)
    """
    num_classes = 3
    global_counts = torch.zeros(num_classes, dtype=torch.float64)
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)

    n = len(dt_train)
    if max_samples and max_samples > 0:
        n = min(n, int(max_samples))

    for i in range(n):
        _, tgt = dt_train[i]
        st = tgt["stress_target"].view(-1).long()
        for c in st.tolist():
            if 0 <= c < num_classes:
                global_counts[c] += 1.0

        ndvi = tgt["ndvi_target"]
        y = torch.zeros_like(ndvi, dtype=torch.long)
        y[(ndvi >= 0.1) & (ndvi < 0.4)] = 1
        y[ndvi < 0.1] = 2
        pixel_counts += torch.bincount(y.view(-1), minlength=num_classes).to(torch.float64)

    def _balanced_weights(counts: torch.Tensor) -> torch.Tensor:
        counts = counts.clamp_min(1.0)
        w = counts.sum() / (num_classes * counts)
        w = w / w.mean().clamp_min(1e-8)
        return w.to(torch.float32)

    w_global = _balanced_weights(global_counts)
    w_pixel = _balanced_weights(pixel_counts)
    return w_global, w_pixel, global_counts, pixel_counts


def _stress_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    label_smoothing: float,
    use_focal: bool,
    focal_gamma: float,
) -> torch.Tensor:
    """
    logits: (N, C, ...) target: (N, ...) con classi long
    """
    if not use_focal or focal_gamma <= 0:
        return F.cross_entropy(
            logits,
            target,
            weight=weight,
            label_smoothing=float(label_smoothing),
        )

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    tgt = target.unsqueeze(1)
    log_pt = log_probs.gather(1, tgt).squeeze(1)
    pt = probs.gather(1, tgt).squeeze(1)
    ce = -log_pt
    focal = (1.0 - pt.clamp(0.0, 1.0)) ** float(focal_gamma)
    loss = focal * ce
    if weight is not None:
        wvec = weight.to(device=loss.device, dtype=loss.dtype)
        # weight[target] mantiene la forma di target (N,) o (N,H,W); evita gather+view(-1)
        # che appiattisce i pesi e rompe il broadcast con loss spaziale.
        w = wvec[target.long()]
        loss = loss * w
    return loss.mean()


def _multiclass_confusion_from_probs(
    probs: torch.Tensor, targets: torch.Tensor, num_classes: int = 3
) -> torch.Tensor:
    """Confusion matrix multiclass da probabilita` (ultima dim = classi) e target interi."""
    with torch.no_grad():
        p = probs.detach()
        t = targets.detach().long()
        pred = torch.argmax(p, dim=-1).reshape(-1)
        tgt = t.reshape(-1)
        valid = (tgt >= 0) & (tgt < num_classes) & torch.isfinite(pred.float())
        if not torch.any(valid):
            return torch.zeros((num_classes, num_classes), dtype=torch.long)
        pred = pred[valid]
        tgt = tgt[valid]
        idx = tgt * num_classes + pred
        cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        return cm.cpu()


def _multiclass_metrics_from_confusion(cm: torch.Tensor) -> dict:
    """Accuracy + macro precision/recall/f1/iou per stress ternario."""
    cm = cm.to(torch.float64)
    total = cm.sum().item()
    correct = torch.diag(cm).sum().item()
    acc = (correct / total) if total > 0 else float("nan")

    per_class = {}
    precisions, recalls, f1s, ious = [], [], [], []
    for cls_idx, cls_name in enumerate(["healthy", "warning", "critical"]):
        tp = cm[cls_idx, cls_idx].item()
        fp = cm[:, cls_idx].sum().item() - tp
        fn = cm[cls_idx, :].sum().item() - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2.0 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

        per_class[cls_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "support": int(tp + fn),
            "tn": int(tn),
        }

        if np.isfinite(precision):
            precisions.append(precision)
        if np.isfinite(recall):
            recalls.append(recall)
        if np.isfinite(f1):
            f1s.append(f1)
        if np.isfinite(iou):
            ious.append(iou)

    return {
        "acc": float(acc),
        "precision_macro": float(np.mean(precisions)) if precisions else float("nan"),
        "recall_macro": float(np.mean(recalls)) if recalls else float("nan"),
        "f1_macro": float(np.mean(f1s)) if f1s else float("nan"),
        "iou_macro": float(np.mean(ious)) if ious else float("nan"),
        "per_class": per_class,
    }


class _CloudDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CloudUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = _CloudDoubleConv(in_channels, c1)
        self.enc2 = _CloudDoubleConv(c1, c2)
        self.enc3 = _CloudDoubleConv(c2, c3)
        self.enc4 = _CloudDoubleConv(c3, c4)
        self.bottleneck = _CloudDoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = _CloudDoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _CloudDoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _CloudDoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _CloudDoubleConv(c1 + c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


def _infer_cloud_detector_spec(state_dict: dict) -> tuple[int, int, int]:
    first_weight = state_dict.get("enc1.block.0.weight")
    head_weight = state_dict.get("head.weight")
    if first_weight is None or head_weight is None:
        raise KeyError(
            "Checkpoint cloud detector non compatibile con la U-Net RGB attesa "
            "(chiavi mancanti: 'enc1.block.0.weight' e/o 'head.weight')."
        )
    in_channels = int(first_weight.shape[1])
    base_channels = int(first_weight.shape[0])
    num_classes = int(head_weight.shape[0])
    return in_channels, num_classes, base_channels


def load_cloud_detector(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint cloud detector non trovato: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint
    in_channels, num_classes, base_channels = _infer_cloud_detector_spec(state_dict)
    cloud_model = _CloudUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)
    missing, unexpected = cloud_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[TRAIN CLOUD MASK] Missing U-Net keys: {missing}")
    if unexpected:
        print(f"[TRAIN CLOUD MASK] Unexpected U-Net keys: {unexpected}")
    for param in cloud_model.parameters():
        param.requires_grad_(False)
    cloud_model.eval()
    print(
        f"[TRAIN CLOUD MASK] loaded frozen cloud detector from {checkpoint_path} "
        f"(in_channels={in_channels}, num_classes={num_classes}, base_channels={base_channels})"
    )
    return cloud_model


def extract_cloud_input_rgb(
    sample_x: torch.Tensor,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Usa l'ultimo timestep osservato come proxy spaziale per il masking cloud-aware.
    Il dataset forecasting non espone frame futuri RGB nel training, quindi la mask è
    stimata sull'ultimo input Sentinel-2 del sample.
    """
    s2_last = sample_x[-1].detach().to(torch.float32)
    if norm_mean is not None and norm_std is not None:
        mean_t = norm_mean.detach().to(dtype=torch.float32, device=s2_last.device).view(-1, 1, 1)
        std_t = norm_std.detach().to(dtype=torch.float32, device=s2_last.device).view(-1, 1, 1)
        s2_last = s2_last * (std_t + 1e-6) + mean_t

    rgb = s2_last[[0, 1, 2]].clone()
    for c in range(3):
        band = rgb[c]
        finite = torch.isfinite(band)
        if not torch.any(finite):
            rgb[c] = torch.zeros_like(band)
            continue
        band_valid = band[finite]
        min_val = band_valid.min()
        max_val = band_valid.max()
        if float(max_val - min_val) > 1e-8:
            rgb[c] = (band - min_val) / (max_val - min_val)
        else:
            rgb[c] = torch.zeros_like(band)
    return rgb.clamp(0.0, 1.0)


def predict_cloud_mask(cloud_model: nn.Module, rgb_image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        logits = cloud_model(rgb_image.unsqueeze(0).to(device))
        if logits.shape[-2:] != rgb_image.shape[-2:]:
            logits = F.interpolate(logits, size=rgb_image.shape[-2:], mode="bilinear", align_corners=False)
        if logits.shape[1] == 1:
            cloud_prob = torch.sigmoid(logits[:, 0])
        else:
            probs = torch.softmax(logits, dim=1)
            if logits.shape[1] == 2:
                cloud_prob = probs[:, 1]
            else:
                cloud_prob = 1.0 - probs[:, 0]
    return cloud_prob.squeeze(0).detach()


def _compute_ndvi_loss_tensor(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "mse_mae":
        return 0.7 * F.mse_loss(pred, target) + 0.3 * F.l1_loss(pred, target)
    if loss_type == "mae":
        return F.l1_loss(pred, target)
    if loss_type == "huber":
        return F.huber_loss(pred, target, delta=0.1)
    return F.mse_loss(pred, target)


def compute_masked_ndvi_losses(
    pred_ndvi: torch.Tensor,
    target_ndvi: torch.Tensor,
    valid_masks: torch.Tensor,
    *,
    ndvi_loss_type: str,
    weight_ndvi_map: float,
    weight_ndvi_mean: float,
) -> tuple[torch.Tensor, torch.Tensor, list[float], int]:
    """
    Applica le loss NDVI solo ai pixel validi. `valid_masks` ha shape (B,H,W) e viene
    condivisa su tutti i forecast steps del sample.
    """
    pred_valid_values = []
    target_valid_values = []
    pred_means = []
    target_means = []
    valid_ratios = []
    kept_samples = 0

    B = pred_ndvi.shape[0]
    for b in range(B):
        mask = valid_masks[b].to(device=pred_ndvi.device, dtype=torch.bool)
        valid_ratio = float(mask.float().mean().item())
        valid_ratios.append(valid_ratio)
        if not torch.any(mask):
            continue
        kept_samples += 1
        for step in range(pred_ndvi.shape[1]):
            pred_valid = pred_ndvi[b, step][mask]
            target_valid = target_ndvi[b, step][mask]
            if pred_valid.numel() == 0:
                continue
            pred_valid_values.append(pred_valid)
            target_valid_values.append(target_valid)
            pred_means.append(pred_valid.mean().view(1))
            target_means.append(target_valid.mean().view(1))

    zero = pred_ndvi.sum() * 0.0
    if pred_valid_values:
        pred_cat = torch.cat(pred_valid_values, dim=0)
        target_cat = torch.cat(target_valid_values, dim=0)
        loss_ndvi_map = weight_ndvi_map * _compute_ndvi_loss_tensor(pred_cat, target_cat, ndvi_loss_type)
    else:
        loss_ndvi_map = zero

    if pred_means:
        pred_mean_cat = torch.cat(pred_means, dim=0)
        target_mean_cat = torch.cat(target_means, dim=0)
        loss_ndvi_mean = weight_ndvi_mean * _compute_ndvi_loss_tensor(
            pred_mean_cat, target_mean_cat, ndvi_loss_type
        )
    else:
        loss_ndvi_mean = zero

    return loss_ndvi_map, loss_ndvi_mean, valid_ratios, kept_samples


def main(config):
    print("=== INIZIO MAIN ===")
    print(f"[INFO] Script in esecuzione: {os.path.abspath(__file__)}")
    try:
        import src.utils as _utils_mod

        _u_path = os.path.abspath(_utils_mod.__file__)
        print(f"[INFO] src.utils (pad_collate): {_u_path}")
        try:
            if "out=out" in inspect.getsource(_utils_mod.pad_collate):
                print(
                    "[WARNING] pad_collate contiene ancora out=out -> UserWarning PyTorch. "
                    "Copia sul server anche src/utils.py aggiornato (solo torch.stack(batch, 0))."
                )
        except OSError:
            pass
    except Exception as exc:
        print(f"[WARNING] Impossibile ispezionare src.utils: {exc}")

    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    print("=== PREPARE OUTPUT FATTO ===")

    device = torch.device(config.device)
    print(f"=== DEVICE: {device} ===")

    # Configurazione per training sul fold specificato
    if config.fold == 1:
        train_folds = [2, 3, 4]
        val_fold = [5]
        test_fold = [1]
        fold = 0  # Indice per il fold 1
    elif config.fold == 2:
        train_folds = [1, 3, 4]
        val_fold = [5]
        test_fold = [2]
        fold = 1  # Indice per il fold 2
    elif config.fold == 3:
        train_folds = [1, 2, 4]
        val_fold = [5]
        test_fold = [3]
        fold = 2  # Indice per il fold 3
    elif config.fold == 4:
        train_folds = [1, 2, 3]
        val_fold = [5]
        test_fold = [4]
        fold = 3  # Indice per il fold 4
    elif config.fold == 5:
        train_folds = [1, 2, 3]
        val_fold = [4]
        test_fold = [5]
        fold = 4  # Indice per il fold 5
    else:
        raise ValueError(f"Fold {config.fold} non supportato")
    
    print(f"Training forecasting con fold={fold}, res_dir={config.res_dir}")
    print("=== CONFIGURAZIONE FOLD FATTA ===")
    
    dt_args = dict(
        folder=config.dataset_folder,
        norm=True,
        reference_date=config.ref_date,
        mono_date=config.mono_date,
        input_steps=config.input_steps,
        forecast_steps=config.forecast_steps,
        forecast_lag_days=config.forecast_lag_days,
        ndvi_stress_threshold=config.ndvi_stress_threshold,
        target="forecasting",
    )
    # Dataset di training con augmentation abilitata
    dt_train = PASTIS_Dataset(**dt_args, cache=config.cache, folds=train_folds, 
                              training=True, apply_augmentation=True)
    print("=== DATASET TRAIN CARICATO (con augmentation) ===")
    
    # Verifica lunghezza serie temporali
    print("=== VERIFICA TRAIN DATASET ===")
    valid_train_patches = dt_train.check_temporal_length()
    
    # Dataset di validazione SENZA augmentation
    dt_val = PASTIS_Dataset(**dt_args, folds=val_fold, 
                           training=False, apply_augmentation=False)
    print("=== DATASET VAL CARICATO (senza augmentation) ===")
    
    print("=== VERIFICA VAL DATASET ===")
    valid_val_patches = dt_val.check_temporal_length()
    
    # Dataset di test SENZA augmentation
    dt_test = PASTIS_Dataset(**dt_args, folds=test_fold, 
                            training=False, apply_augmentation=False)
    print("=== DATASET TEST CARICATO (senza augmentation) ===")
    
    print("=== VERIFICA TEST DATASET ===")
    valid_test_patches = dt_test.check_temporal_length()

    train_loader = data.DataLoader(
        dt_train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=pad_collate,
    )
    # Val/test: batch_size al massimo len(dataset) cosi esiste sempre almeno un batch se len>0;
    # drop_last=False; shuffle=False. num_workers=0 evita edge case su worker che svuotano il loader.
    _val_bs = min(int(config.batch_size), max(1, len(dt_val)))
    _test_bs = min(int(config.batch_size), max(1, len(dt_test)))
    if len(dt_val) == 0:
        raise ValueError("Dataset di validazione vuoto (nessun patch nei fold val): impossibile validare.")
    if len(dt_test) == 0:
        raise ValueError("Dataset di test vuoto (nessun patch nei fold test).")
    print(
        f"[INFO] DataLoader val: len={len(dt_val)}, batch_size={_val_bs}, "
        f"num_batches={(len(dt_val) + _val_bs - 1) // _val_bs}"
    )
    print(
        f"[INFO] DataLoader test: len={len(dt_test)}, batch_size={_test_bs}, "
        f"batches≈{(len(dt_test) + _test_bs - 1) // _test_bs}"
    )
    val_loader = data.DataLoader(
        dt_val,
        batch_size=_val_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
        num_workers=0,
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=_test_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
        num_workers=0,
    )
    if len(val_loader) < 1:
        raise RuntimeError(
            f"val_loader ha len={len(val_loader)} con len(dt_val)={len(dt_val)}: controlla dataset e batch_size."
        )
    if config.mode == "forecasting":
        # Carica dati e modello per forecasting
        model = model_utils.get_model(config, mode="forecasting")
        model = model.to(config.device)
        
        # Learning rate normale per forecasting
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # Usa LR normale
        
        # Learning rate scheduler
        if hasattr(config, 'lr_scheduler') and config.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       step_size=getattr(config, 'lr_step_size', 15), 
                                                       gamma=getattr(config, 'lr_gamma', 0.3))
            
            print(f"*****Using StepLR scheduler with step_size={getattr(config, 'lr_step_size', 15)} and gamma={getattr(config, 'lr_gamma', 0.3)}")
            #quit()
        else:
            scheduler = None
        
        # Gradient clipping meno aggressivo
        max_grad_norm = 1.0  # Permetti gradienti più grandi
        
        # Loss functions
        mse_loss = torch.nn.MSELoss()
        mae_loss = torch.nn.L1Loss()
        huber_loss = torch.nn.HuberLoss(delta=0.1)  # Huber loss per robustezza
        use_focal = bool(getattr(config, "use_focal_stress", False))
        focal_gamma = float(getattr(config, "focal_gamma", 2.0))
        stress_label_smoothing = float(getattr(config, "stress_label_smoothing", 0.0))
        focal_on = use_focal and focal_gamma > 0

        # Tipo di loss NDVI (default: mse)
        ndvi_loss_type = getattr(config, 'ndvi_loss_type', 'mse')
        print(f"[INFO] NDVI loss type: {ndvi_loss_type}")

        weight_ndvi_map = getattr(config, 'weight_ndvi_map', 1.0)
        weight_ndvi_mean = getattr(config, 'weight_ndvi_mean', 0.5)
        ndvi_only_loss = bool(getattr(config, "ndvi_only_loss", False))
        weight_stress = 0.0 if ndvi_only_loss else getattr(config, 'weight_stress', 0.3)
        stress_prob_threshold = float(getattr(config, "ndvi_stress_threshold", 0.4))
        if ndvi_only_loss:
            print(
                "[INFO] NDVI-only training: loss = weight_ndvi_map * L(map) + weight_ndvi_mean * L(mean). "
                "Stress disattivato in loss e validazione."
            )
            print(f"[INFO] Loss (NDVI only) — map: {weight_ndvi_map}, mean: {weight_ndvi_mean}")
        else:
            print(f"[INFO] Loss weights - NDVI map: {weight_ndvi_map}, NDVI mean: {weight_ndvi_mean}, Stress: {weight_stress}")
            if weight_stress > 0:
                print(
                    f"[INFO] Ratio NDVI:Stress = {(weight_ndvi_map + weight_ndvi_mean):.1f}:{weight_stress:.1f} "
                    f"= {((weight_ndvi_map + weight_ndvi_mean) / weight_stress):.1f}:1"
                )
            print(f"[INFO] Stress metrics probability threshold: {stress_prob_threshold}")
            print(
                f"[INFO] Stress loss: cross_entropy su logits | focal={focal_on} (gamma={focal_gamma}) | "
                f"label_smoothing={stress_label_smoothing}"
            )

        # Pesi classe stress (auto) stimati sul train set
        stress_global_class_weight = None
        stress_map_class_weight = None
        if not ndvi_only_loss:
            stress_w_mode = getattr(config, "stress_class_weight_mode", "auto")
            max_w_samples = int(getattr(config, "stress_weight_estimation_max_samples", 0))
            if stress_w_mode == "auto":
                w_g, w_p, c_g, c_p = _estimate_stress_class_weights_from_dataset(dt_train, max_samples=max_w_samples)
                stress_global_class_weight = w_g.to(config.device)
                stress_map_class_weight = w_p.to(config.device)
                print("[INFO] Stress class weights (global):", stress_global_class_weight.detach().cpu().tolist())
                print("[INFO] Stress class counts (global):", c_g.detach().cpu().tolist())
                print("[INFO] Stress class weights (map pixels):", stress_map_class_weight.detach().cpu().tolist())
                print("[INFO] Stress class counts (map pixels):", c_p.detach().cpu().tolist())
            else:
                print("[INFO] Stress class weights: disabled (uniform)")

        cloud_train_model = None
        cloud_train_mean = None
        cloud_train_std = None
        if bool(getattr(config, "use_cloud_mask_train", False)):
            if not getattr(config, "cloud_detector_weights", ""):
                raise ValueError("--use_cloud_mask_train richiede anche --cloud_detector_weights")
            cloud_train_model = load_cloud_detector(config.cloud_detector_weights, device)
            sat_name = dt_train.sats[0] if len(dt_train.sats) > 0 else None
            if sat_name is not None and isinstance(dt_train.norm, dict) and sat_name in dt_train.norm:
                cloud_train_mean, cloud_train_std = dt_train.norm[sat_name]
            print(
                f"[TRAIN CLOUD MASK] enabled | threshold={float(getattr(config, 'cloud_threshold', 0.4)):.3f}"
            )

        fold_dir = os.path.join(config.res_dir, f"Fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        best_val_loss = float("inf")
        best_val_epoch = -1
        best_model_path = ""
        best_so_far_path = os.path.join(fold_dir, "model_best_so_far.pth.tar")
        best_so_far_meta_path = os.path.join(fold_dir, "best_so_far_status.json")
        start_epoch = 0
        # In resume, il confronto "best di questo run" deve ripartire da +inf.
        # Non usiamo best_val_loss storica del checkpoint come soglia.
        resume_val_threshold = None

        if getattr(config, "resume_path", ""):
            ckpt_path = config.resume_path
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint di resume non trovato: {ckpt_path}")
            print(f"[RESUME] Caricamento checkpoint da: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
            if "state_dict" not in checkpoint:
                raise KeyError("Checkpoint senza chiave 'state_dict': impossibile fare resume.")
            ckpt_state = checkpoint["state_dict"]
            model_state = model.state_dict()
            filtered_state = {}
            skipped_mismatch = []
            skipped_missing = []

            for k, v in ckpt_state.items():
                if k not in model_state:
                    skipped_missing.append(k)
                    continue
                if model_state[k].shape != v.shape:
                    skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                    continue
                filtered_state[k] = v

            model.load_state_dict(filtered_state, strict=False)
            print(
                f"[RESUME] Pesi caricati: {len(filtered_state)}/{len(ckpt_state)} "
                "(chiavi compatibili per nome+shape)."
            )
            if skipped_mismatch:
                print("[RESUME] Layer saltati per shape incompatibile:")
                for name, old_shape, new_shape in skipped_mismatch[:20]:
                    print(f"  - {name}: ckpt={old_shape} -> model={new_shape}")
                if len(skipped_mismatch) > 20:
                    print(f"  ... altri {len(skipped_mismatch) - 20} layer incompatibili")
            if skipped_missing:
                print(f"[RESUME] Chiavi non presenti nel modello corrente: {len(skipped_missing)}")
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as exc:
                    print(f"[WARNING] Impossibile caricare optimizer state, continuo solo con i pesi modello: {exc}")
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float("inf")
            best_val_epoch = -1
            resume_val_threshold = None
            print(
                f"[RESUME] Checkpoint epoca {start_epoch} caricato. "
                "Best su validation resettato: confronto solo sulle epoche di questo run."
            )

        # In resume, --epochs indica quante epoche AGGIUNTIVE eseguire.
        total_epochs = start_epoch + config.epochs
        if start_epoch > 0:
            print(f"[RESUME] Riparto da epoca {start_epoch + 1} fino a {total_epochs} (+{config.epochs} epoche)")
        else:
            print(f"[TRAIN] Training da zero fino a {total_epochs} epoche")
        print(
            f"[INFO] Regole validazione: val_every={getattr(config, 'val_every', '?')}, "
            f"val_after={getattr(config, 'val_after', '?')} (serve (epoca+1)>val_after e (epoca+1)%val_every==0)"
        )

        trainlog = {}
        for epoch in range(start_epoch, total_epochs):
            model.train()
            total_loss = 0
            total_ndvi_map_loss = 0
            total_ndvi_mean_loss = 0
            total_stress_global_loss = 0
            total_stress_map_loss = 0
            num_batches = 0
            epoch_cloud_valid_ratio = 0.0
            epoch_cloud_mask_samples = 0
            tr_sg_cm = torch.zeros((3, 3), dtype=torch.long)
            tr_sm_cm = torch.zeros((3, 3), dtype=torch.long)
            
            for batch_idx, batch in enumerate(train_loader):
                (x, dates), targets = batch
                x = x.to(config.device)
                dates = dates.to(config.device)
                ndvi_target = targets['ndvi_target'].to(config.device)
                stress_target = targets['stress_target'].to(config.device)
                ndvi_mean_target = targets['ndvi_mean_target'].to(config.device)
                
                ndvi_map_pred, ndvi_mean_pred, stress_global_pred, stress_map_pred = model(x, batch_positions=dates, forecast=True)

                # Upsample se necessario per NDVI map
                if ndvi_map_pred.shape[-2:] != ndvi_target.shape[-2:]:
                    ndvi_map_pred = ndvi_map_pred.view(-1, *ndvi_map_pred.shape[-2:])
                    ndvi_map_pred = F.interpolate(
                        ndvi_map_pred.unsqueeze(1),
                        size=ndvi_target.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    ndvi_map_pred = ndvi_map_pred.view_as(ndvi_target)
                
                # Upsample stress map ternaria se necessario.
                # Shape attesa: (B, S, H, W, C) con C=3
                if stress_map_pred.shape[2:4] != ndvi_target.shape[-2:]:
                    B, S, H, W, C = stress_map_pred.shape
                    stress_map_pred = stress_map_pred.permute(0, 1, 4, 2, 3).contiguous()  # (B,S,C,H,W)
                    stress_map_pred = stress_map_pred.view(B * S, C, H, W)
                    stress_map_pred = F.interpolate(
                        stress_map_pred,
                        size=ndvi_target.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    H2, W2 = ndvi_target.shape[-2:]
                    stress_map_pred = stress_map_pred.view(B, S, C, H2, W2).permute(0, 1, 3, 4, 2).contiguous()

                # Calcolo delle loss NDVI con tipo selezionato
                if cloud_train_model is not None:
                    with torch.no_grad():
                        valid_masks = []
                        for sample_idx in range(x.shape[0]):
                            rgb_input = extract_cloud_input_rgb(
                                x[sample_idx],
                                norm_mean=cloud_train_mean,
                                norm_std=cloud_train_std,
                            )
                            cloud_prob = predict_cloud_mask(cloud_train_model, rgb_input, device=device)
                            valid_masks.append((cloud_prob < float(getattr(config, "cloud_threshold", 0.4))).to(x.device))
                        valid_masks = torch.stack(valid_masks, dim=0)
                    loss_ndvi_map, loss_ndvi_mean, valid_ratios, kept_samples = compute_masked_ndvi_losses(
                        ndvi_map_pred,
                        ndvi_target,
                        valid_masks,
                        ndvi_loss_type=ndvi_loss_type,
                        weight_ndvi_map=weight_ndvi_map,
                        weight_ndvi_mean=weight_ndvi_mean,
                    )
                    epoch_cloud_valid_ratio += float(np.sum(valid_ratios))
                    epoch_cloud_mask_samples += len(valid_ratios)
                    if batch_idx % 10 == 0 and valid_ratios:
                        print(
                            f"  [TRAIN CLOUD MASK] batch {batch_idx}: "
                            f"mean_valid_ratio={float(np.mean(valid_ratios)):.4f}, "
                            f"kept_samples={kept_samples}/{len(valid_ratios)}"
                        )
                else:
                    # Loss NDVI map principale
                    if ndvi_loss_type == "mse":
                        loss_ndvi_map = weight_ndvi_map * mse_loss(ndvi_map_pred, ndvi_target)
                    elif ndvi_loss_type == "mse_mae":
                        # Combinazione 70% MSE + 30% MAE per robustezza
                        loss_ndvi_map = weight_ndvi_map * (0.7 * mse_loss(ndvi_map_pred, ndvi_target) + 0.3 * mae_loss(ndvi_map_pred, ndvi_target))
                    elif ndvi_loss_type == "mae":
                        # Solo MAE (L1)
                        loss_ndvi_map = weight_ndvi_map * mae_loss(ndvi_map_pred, ndvi_target)
                    elif ndvi_loss_type == "huber":
                        loss_ndvi_map = weight_ndvi_map * huber_loss(ndvi_map_pred, ndvi_target)
                    else:
                        loss_ndvi_map = weight_ndvi_map * mse_loss(ndvi_map_pred, ndvi_target)

                    if ndvi_loss_type == "mse":
                        loss_ndvi_mean = weight_ndvi_mean * mse_loss(ndvi_mean_pred, ndvi_mean_target)
                    elif ndvi_loss_type == "mse_mae":
                        loss_ndvi_mean = weight_ndvi_mean * (
                            0.7 * mse_loss(ndvi_mean_pred, ndvi_mean_target) + 0.3 * mae_loss(ndvi_mean_pred, ndvi_mean_target)
                        )
                    elif ndvi_loss_type == "mae":
                        loss_ndvi_mean = weight_ndvi_mean * mae_loss(ndvi_mean_pred, ndvi_mean_target)
                    elif ndvi_loss_type == "huber":
                        loss_ndvi_mean = weight_ndvi_mean * huber_loss(ndvi_mean_pred, ndvi_mean_target)
                    else:
                        loss_ndvi_mean = weight_ndvi_mean * mse_loss(ndvi_mean_pred, ndvi_mean_target)

                loss_stress_global = torch.tensor(0.0, device=config.device)
                loss_stress_map = torch.tensor(0.0, device=config.device)
                if not ndvi_only_loss:
                    stress_target_global = stress_target.squeeze(-1).long()
                    stress_map_target = torch.zeros_like(ndvi_target, dtype=torch.long)
                    stress_map_target[(ndvi_target >= 0.1) & (ndvi_target < 0.4)] = 1
                    stress_map_target[ndvi_target < 0.1] = 2

                    # Stress ternario: logits -> CE (pesi classe + opzionale focal)
                    stress_global_logits = stress_global_pred.contiguous()
                    stress_map_logits = stress_map_pred.contiguous()

                    sg_logits = stress_global_logits.view(-1, 3)
                    sg_tgt = stress_target_global.contiguous().view(-1)
                    loss_stress_global = weight_stress * _stress_ce_loss(
                        sg_logits,
                        sg_tgt,
                        weight=stress_global_class_weight,
                        label_smoothing=stress_label_smoothing,
                        use_focal=focal_on,
                        focal_gamma=focal_gamma,
                    )

                    Bm, Sm, Hm, Wm, Cm = stress_map_logits.shape
                    sm_logits = stress_map_logits.permute(0, 1, 4, 2, 3).contiguous().view(-1, Cm, Hm, Wm)
                    sm_tgt = stress_map_target.contiguous().view(-1, Hm, Wm)
                    loss_stress_map = weight_stress * _stress_ce_loss(
                        sm_logits,
                        sm_tgt,
                        weight=stress_map_class_weight,
                        label_smoothing=stress_label_smoothing,
                        use_focal=focal_on,
                        focal_gamma=focal_gamma,
                    )

                    with torch.no_grad():
                        stress_global_prob = torch.softmax(stress_global_logits, dim=-1)
                        stress_map_prob = torch.softmax(stress_map_logits, dim=-1)
                    tr_sg_cm += _multiclass_confusion_from_probs(stress_global_prob, stress_target_global, num_classes=3)
                    tr_sm_cm += _multiclass_confusion_from_probs(stress_map_prob, stress_map_target, num_classes=3)

                loss = loss_ndvi_map + loss_ndvi_mean + loss_stress_global + loss_stress_map

                optimizer.zero_grad()
                loss.backward()
                
                # Aggiungi gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                # Accumula statistiche
                total_loss += loss.item()
                total_ndvi_map_loss += loss_ndvi_map.item()
                total_ndvi_mean_loss += loss_ndvi_mean.item()
                total_stress_global_loss += loss_stress_global.item()
                total_stress_map_loss += loss_stress_map.item()
                num_batches += 1
                
                # Debug ogni 10 batch
                if batch_idx % 10 == 0:
                    if ndvi_only_loss:
                        print(
                            f"  Batch {batch_idx}: Loss={loss.item():.4f}, "
                            f"NDVI_map={loss_ndvi_map.item():.4f}, "
                            f"NDVI_mean={loss_ndvi_mean.item():.4f}"
                        )
                    else:
                        print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, "
                              f"NDVI_map={loss_ndvi_map.item():.4f}, "
                              f"NDVI_mean={loss_ndvi_mean.item():.4f}, "
                              f"Stress_global={loss_stress_global.item():.4f}, "
                              f"Stress_map={loss_stress_map.item():.4f}")
                    
                    # Verifica che i gradienti non siano zero
                    total_grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    print(f"  Gradient norm: {total_grad_norm:.6f}")
            
            # Step del learning rate scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Calcola medie
            avg_loss = total_loss / num_batches
            avg_ndvi_map_loss = total_ndvi_map_loss / num_batches
            avg_ndvi_mean_loss = total_ndvi_mean_loss / num_batches
            avg_stress_global_loss = total_stress_global_loss / num_batches
            avg_stress_map_loss = total_stress_map_loss / num_batches
            tr_sg_metrics = _multiclass_metrics_from_confusion(tr_sg_cm)
            tr_sm_metrics = _multiclass_metrics_from_confusion(tr_sm_cm)

            if ndvi_only_loss:
                print(
                    f"Epoch {epoch+1}/{total_epochs} - "
                    f"Total Loss (NDVI only): {avg_loss:.4f} | "
                    f"NDVI map loss: {avg_ndvi_map_loss:.4f} | "
                    f"NDVI mean loss: {avg_ndvi_mean_loss:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{total_epochs} - "
                    f"Total Loss: {avg_loss:.4f} | "
                    f"NDVI map loss: {avg_ndvi_map_loss:.4f} | "
                    f"NDVI mean loss: {avg_ndvi_mean_loss:.4f} | "
                    f"Stress global loss: {avg_stress_global_loss:.4f} | "
                    f"Stress map loss: {avg_stress_map_loss:.4f}"
                )
                print(
                    f"  Stress global (ternary) - acc: {tr_sg_metrics['acc']:.4f}, "
                    f"prec_macro: {tr_sg_metrics['precision_macro']:.4f}, rec_macro: {tr_sg_metrics['recall_macro']:.4f}, "
                    f"f1_macro: {tr_sg_metrics['f1_macro']:.4f}, iou_macro: {tr_sg_metrics['iou_macro']:.4f}"
                )
                print(
                    f"  Stress map (ternary) - acc: {tr_sm_metrics['acc']:.4f}, "
                    f"prec_macro: {tr_sm_metrics['precision_macro']:.4f}, rec_macro: {tr_sm_metrics['recall_macro']:.4f}, "
                    f"f1_macro: {tr_sm_metrics['f1_macro']:.4f}, iou_macro: {tr_sm_metrics['iou_macro']:.4f}"
                )
            if cloud_train_model is not None and epoch_cloud_mask_samples > 0:
                mean_valid_ratio_epoch = epoch_cloud_valid_ratio / epoch_cloud_mask_samples
                print(
                    f"  [TRAIN CLOUD MASK] mean_valid_ratio_epoch: {mean_valid_ratio_epoch:.4f}"
                )
            
            trainlog[epoch+1] = {
                "total_loss": avg_loss,
                "ndvi_map_loss": avg_ndvi_map_loss,
                "ndvi_mean_loss": avg_ndvi_mean_loss,
                "stress_global_loss": avg_stress_global_loss,
                "stress_map_loss": avg_stress_map_loss,
                "ndvi_only_loss": ndvi_only_loss,
            }
            if not ndvi_only_loss:
                trainlog[epoch + 1].update({
                    "stress_global_acc": tr_sg_metrics["acc"],
                    "stress_global_precision_macro": tr_sg_metrics["precision_macro"],
                    "stress_global_recall_macro": tr_sg_metrics["recall_macro"],
                    "stress_global_f1_macro": tr_sg_metrics["f1_macro"],
                    "stress_global_iou_macro": tr_sg_metrics["iou_macro"],
                    "stress_map_acc": tr_sm_metrics["acc"],
                    "stress_map_precision_macro": tr_sm_metrics["precision_macro"],
                    "stress_map_recall_macro": tr_sm_metrics["recall_macro"],
                    "stress_map_f1_macro": tr_sm_metrics["f1_macro"],
                    "stress_map_iou_macro": tr_sm_metrics["iou_macro"],
                })
            if cloud_train_model is not None and epoch_cloud_mask_samples > 0:
                trainlog[epoch + 1]["train_cloud_mask_mean_valid_ratio"] = (
                    epoch_cloud_valid_ratio / epoch_cloud_mask_samples
                )

            # ===============================
            # Validation (forecasting) opzioniale
            # ===============================
            do_val = True
            if hasattr(config, "val_every"):
                do_val = do_val and ((epoch + 1) % getattr(config, "val_every", 1) == 0)
            if hasattr(config, "val_after"):
                do_val = do_val and ((epoch + 1) > getattr(config, "val_after", 0))
            if do_val:
                model.eval()
                val_err = None
                with torch.no_grad():
                    val_total = 0.0
                    val_ndvi_map = 0.0
                    val_ndvi_mean = 0.0
                    val_stress_g = 0.0
                    val_stress_m = 0.0
                    val_batches = 0
                    val_sg_cm = torch.zeros((3, 3), dtype=torch.long)
                    val_sm_cm = torch.zeros((3, 3), dtype=torch.long)
                    try:
                        for (xv, dv), tv in val_loader:
                            xv = xv.to(config.device)
                            dv = dv.to(config.device)
                            ndvi_target_v = tv["ndvi_target"].to(config.device)
                            stress_target_v = tv["stress_target"].to(config.device)
                            ndvi_mean_target_v = tv["ndvi_mean_target"].to(config.device)

                            ndvi_map_pred_v, ndvi_mean_pred_v, stress_global_pred_v, stress_map_pred_v = model(
                                xv, batch_positions=dv, forecast=True
                            )

                            if ndvi_map_pred_v.shape[-2:] != ndvi_target_v.shape[-2:]:
                                ndvi_map_pred_v = ndvi_map_pred_v.view(-1, *ndvi_map_pred_v.shape[-2:])
                                ndvi_map_pred_v = F.interpolate(
                                    ndvi_map_pred_v.unsqueeze(1),
                                    size=ndvi_target_v.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(1)
                                ndvi_map_pred_v = ndvi_map_pred_v.view_as(ndvi_target_v)

                            if stress_map_pred_v.shape[2:4] != ndvi_target_v.shape[-2:]:
                                Bv, Sv, Hv, Wv, Cv = stress_map_pred_v.shape
                                stress_map_pred_v = stress_map_pred_v.permute(0, 1, 4, 2, 3).contiguous()  # (B,S,C,H,W)
                                stress_map_pred_v = stress_map_pred_v.view(Bv * Sv, Cv, Hv, Wv)
                                stress_map_pred_v = F.interpolate(
                                    stress_map_pred_v,
                                    size=ndvi_target_v.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                H2, W2 = ndvi_target_v.shape[-2:]
                                stress_map_pred_v = stress_map_pred_v.view(Bv, Sv, Cv, H2, W2).permute(0, 1, 3, 4, 2).contiguous()

                            if ndvi_loss_type == "mse":
                                v_loss_ndvi_map = weight_ndvi_map * mse_loss(ndvi_map_pred_v, ndvi_target_v)
                            elif ndvi_loss_type == "mse_mae":
                                v_loss_ndvi_map = weight_ndvi_map * (
                                    0.7 * mse_loss(ndvi_map_pred_v, ndvi_target_v) + 0.3 * mae_loss(ndvi_map_pred_v, ndvi_target_v)
                                )
                            elif ndvi_loss_type == "mae":
                                v_loss_ndvi_map = weight_ndvi_map * mae_loss(ndvi_map_pred_v, ndvi_target_v)
                            elif ndvi_loss_type == "huber":
                                v_loss_ndvi_map = weight_ndvi_map * huber_loss(ndvi_map_pred_v, ndvi_target_v)
                            else:
                                v_loss_ndvi_map = weight_ndvi_map * mse_loss(ndvi_map_pred_v, ndvi_target_v)

                            if ndvi_loss_type == "mse":
                                v_loss_ndvi_mean = weight_ndvi_mean * mse_loss(ndvi_mean_pred_v, ndvi_mean_target_v)
                            elif ndvi_loss_type == "mse_mae":
                                v_loss_ndvi_mean = weight_ndvi_mean * (
                                    0.7 * mse_loss(ndvi_mean_pred_v, ndvi_mean_target_v)
                                    + 0.3 * mae_loss(ndvi_mean_pred_v, ndvi_mean_target_v)
                                )
                            elif ndvi_loss_type == "mae":
                                v_loss_ndvi_mean = weight_ndvi_mean * mae_loss(ndvi_mean_pred_v, ndvi_mean_target_v)
                            elif ndvi_loss_type == "huber":
                                v_loss_ndvi_mean = weight_ndvi_mean * huber_loss(ndvi_mean_pred_v, ndvi_mean_target_v)
                            else:
                                v_loss_ndvi_mean = weight_ndvi_mean * mse_loss(ndvi_mean_pred_v, ndvi_mean_target_v)

                            v_loss_stress_global = torch.tensor(0.0, device=config.device)
                            v_loss_stress_map = torch.tensor(0.0, device=config.device)
                            if not ndvi_only_loss:
                                stress_target_global_v = stress_target_v.squeeze(-1).long()
                                stress_map_target_v = torch.zeros_like(ndvi_target_v, dtype=torch.long)
                                stress_map_target_v[(ndvi_target_v >= 0.1) & (ndvi_target_v < 0.4)] = 1
                                stress_map_target_v[ndvi_target_v < 0.1] = 2

                                stress_global_logits_v = stress_global_pred_v.contiguous()
                                stress_map_logits_v = stress_map_pred_v.contiguous()

                                sg_logits_v = stress_global_logits_v.view(-1, 3)
                                sg_tgt_v = stress_target_global_v.contiguous().view(-1)
                                v_loss_stress_global = weight_stress * _stress_ce_loss(
                                    sg_logits_v,
                                    sg_tgt_v,
                                    weight=stress_global_class_weight,
                                    label_smoothing=stress_label_smoothing,
                                    use_focal=focal_on,
                                    focal_gamma=focal_gamma,
                                )

                                Bv, Sv, Hv, Wv, Cv = stress_map_logits_v.shape
                                sm_logits_v = stress_map_logits_v.permute(0, 1, 4, 2, 3).contiguous().view(-1, Cv, Hv, Wv)
                                sm_tgt_v = stress_map_target_v.contiguous().view(-1, Hv, Wv)
                                v_loss_stress_map = weight_stress * _stress_ce_loss(
                                    sm_logits_v,
                                    sm_tgt_v,
                                    weight=stress_map_class_weight,
                                    label_smoothing=stress_label_smoothing,
                                    use_focal=focal_on,
                                    focal_gamma=focal_gamma,
                                )

                                with torch.no_grad():
                                    stress_global_prob_v = torch.softmax(stress_global_logits_v, dim=-1)
                                    stress_map_prob_v = torch.softmax(stress_map_logits_v, dim=-1)
                                val_sg_cm += _multiclass_confusion_from_probs(stress_global_prob_v, stress_target_global_v, num_classes=3)
                                val_sm_cm += _multiclass_confusion_from_probs(stress_map_prob_v, stress_map_target_v, num_classes=3)

                            v_loss = v_loss_ndvi_map + v_loss_ndvi_mean + v_loss_stress_global + v_loss_stress_map
                            val_total += float(v_loss.item())
                            val_ndvi_map += float(v_loss_ndvi_map.item())
                            val_ndvi_mean += float(v_loss_ndvi_mean.item())
                            val_stress_g += float(v_loss_stress_global.item())
                            val_stress_m += float(v_loss_stress_map.item())
                            val_batches += 1
                    except Exception as exc:
                        val_err = exc
                        print(f"[ERROR] Validazione epoca {epoch + 1} fallita (prima o durante i batch): {exc}")
                        traceback.print_exc()
                if val_err is not None:
                    pass
                elif val_batches > 0:
                    current_val_loss = val_total / val_batches
                    val_log_update = {
                        "val_loss": current_val_loss,
                        "val_ndvi_map_loss": val_ndvi_map / val_batches,
                        "val_ndvi_mean_loss": val_ndvi_mean / val_batches,
                        "val_stress_global_loss": val_stress_g / val_batches,
                        "val_stress_map_loss": val_stress_m / val_batches,
                    }
                    if not ndvi_only_loss:
                        val_sg_metrics = _multiclass_metrics_from_confusion(val_sg_cm)
                        val_sm_metrics = _multiclass_metrics_from_confusion(val_sm_cm)
                        val_log_update.update({
                            "val_stress_global_acc": val_sg_metrics["acc"],
                            "val_stress_global_precision_macro": val_sg_metrics["precision_macro"],
                            "val_stress_global_recall_macro": val_sg_metrics["recall_macro"],
                            "val_stress_global_f1_macro": val_sg_metrics["f1_macro"],
                            "val_stress_global_iou_macro": val_sg_metrics["iou_macro"],
                            "val_stress_map_acc": val_sm_metrics["acc"],
                            "val_stress_map_precision_macro": val_sm_metrics["precision_macro"],
                            "val_stress_map_recall_macro": val_sm_metrics["recall_macro"],
                            "val_stress_map_f1_macro": val_sm_metrics["f1_macro"],
                            "val_stress_map_iou_macro": val_sm_metrics["iou_macro"],
                        })
                    trainlog[epoch + 1].update(val_log_update)
                    if not ndvi_only_loss:
                        print(
                            f"[VAL][Stress global ternary] acc={val_sg_metrics['acc']:.4f}, "
                            f"prec_macro={val_sg_metrics['precision_macro']:.4f}, rec_macro={val_sg_metrics['recall_macro']:.4f}, "
                            f"f1_macro={val_sg_metrics['f1_macro']:.4f}, iou_macro={val_sg_metrics['iou_macro']:.4f}"
                        )
                        print(
                            f"[VAL][Stress map ternary] acc={val_sm_metrics['acc']:.4f}, "
                            f"prec_macro={val_sm_metrics['precision_macro']:.4f}, rec_macro={val_sm_metrics['recall_macro']:.4f}, "
                            f"f1_macro={val_sm_metrics['f1_macro']:.4f}, iou_macro={val_sm_metrics['iou_macro']:.4f}"
                        )
                    _thr = best_val_loss
                    print(
                        f"[VAL] Epoca {epoch + 1}: val_loss={current_val_loss:.6f} "
                        f"(migliore da battere in questo run: {_thr:.6f})"
                    )

                    # Salva miglior modello in base alla validation loss
                    if current_val_loss < best_val_loss:
                        # Rimuovi il precedente best con epoca nel nome per non accumulare file
                        if best_model_path and os.path.exists(best_model_path):
                            try:
                                os.remove(best_model_path)
                            except OSError:
                                pass
                        best_val_loss = current_val_loss
                        best_val_epoch = epoch + 1
                        best_model_path = os.path.join(
                            fold_dir, f"model_best_val_loss_epoch_{best_val_epoch}.pth.tar"
                        )
                        torch.save(
                            {
                                "epoch": best_val_epoch,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "best_val_loss": best_val_loss,
                            },
                            best_model_path,
                        )
                        # Copia stabile (ultimo best) senza epoca nel nome
                        torch.save(
                            {
                                "epoch": best_val_epoch,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "best_val_loss": best_val_loss,
                            },
                            os.path.join(fold_dir, "model_best_val_loss.pth.tar"),
                        )
                        print(
                            f"[BEST] Nuovo best val_loss={best_val_loss:.6f} "
                            f"all'epoca {best_val_epoch} -> {best_model_path}"
                        )
                else:
                    print(
                        f"[WARNING] Epoca {epoch + 1}: validazione con 0 batch completati "
                        f"(len(dt_val)={len(dt_val)}, val_batch_size={_val_bs}, len(val_loader)={len(val_loader)}). "
                        "Nessun aggiornamento best_val_loss."
                    )

            # Salva ogni N epoche (configurabile)
            _save_every = max(1, int(getattr(config, "save_every_n_epochs", 5)))
            if (epoch + 1) % _save_every == 0 or (epoch + 1) == total_epochs:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(fold_dir, f"model_epoch_{epoch+1}.pth.tar"),
                )
                with open(os.path.join(fold_dir, f"trainlog_epoch_{epoch+1}.json"), "w") as outfile:
                    json.dump(trainlog, outfile, indent=4)
                print(f"✅ Modello salvato all'epoca {epoch+1} in {fold_dir}")

            # Salva SEMPRE ad ogni epoca il "migliore finora" SOLO se definito da val_loss totale minima.
            # Niente fallback al modello corrente: il best deve essere realmente basato sulla validazione.
            best_payload = None
            if best_val_epoch > 0 and os.path.isfile(os.path.join(fold_dir, "model_best_val_loss.pth.tar")):
                try:
                    best_payload = torch.load(
                        os.path.join(fold_dir, "model_best_val_loss.pth.tar"),
                        map_location="cpu",
                        weights_only=False,
                    )
                except Exception as exc:
                    print(f"[WARNING] Impossibile leggere model_best_val_loss.pth.tar, skip best-so-far: {exc}")

            if best_payload is not None:
                best_payload["saved_at_epoch"] = epoch + 1
                best_payload["best_epoch_so_far"] = best_val_epoch
                torch.save(best_payload, best_so_far_path)
                best_so_far_numbered_path = os.path.join(
                    fold_dir, f"model_best_so_far_epoch_{best_payload['best_epoch_so_far']}.pth.tar"
                )
                torch.save(best_payload, best_so_far_numbered_path)
                with open(best_so_far_meta_path, "w") as outfile:
                    json.dump(
                        {
                            "saved_at_epoch": epoch + 1,
                            "best_epoch_so_far": best_payload["best_epoch_so_far"],
                            "best_val_loss_so_far": best_payload.get("best_val_loss", None),
                            "source": "model_best_val_loss.pth.tar",
                        },
                        outfile,
                        indent=4,
                    )
                print(
                    f"[BEST-SO-FAR] Aggiornato all'epoca {epoch + 1}: "
                    f"latest={best_so_far_path} | numbered={best_so_far_numbered_path}"
                )
            else:
                print(
                    f"[BEST-SO-FAR] Epoca {epoch + 1}: nessun best validazione disponibile, "
                    "salvataggio best-so-far non aggiornato."
                )

        # Salva modello finale
        fold_dir = os.path.join(config.res_dir, f"Fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Salva con timestamp per evitare sovrascrittura
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(fold_dir, f"model_final_{timestamp}.pth.tar")
        
        torch.save(
            {
                "epoch": total_epochs,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            final_model_path,
        )
        
        # Salva anche una copia senza timestamp per compatibilità
        torch.save(
            {
                "epoch": total_epochs,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(fold_dir, "model.pth.tar"),
        )
        
        with open(os.path.join(fold_dir, f"trainlog_{timestamp}.json"), "w") as outfile:
            json.dump(trainlog, outfile, indent=4)
        with open(os.path.join(fold_dir, "trainlog.json"), "w") as outfile:
            json.dump(trainlog, outfile, indent=4)
        print(f"✅ Modello finale salvato in {fold_dir}")
        print(f"✅ Copia con timestamp: {final_model_path}")
        if best_model_path and os.path.isfile(best_model_path):
            print(
                f"✅ Miglior modello su validation loss in questo run: epoca {best_val_epoch} "
                f"(val_loss={best_val_loss:.6f})"
            )
            print(f"✅ Path best model: {best_model_path}")
        else:
            print(
                "[WARNING] Nessun best model su validation loss salvato in questo run "
                "(validazione non eseguita, fallita, o zero batch validi: vedi [ERROR]/[WARNING] sopra)."
            )
        exit()

    print(
        "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
    )

    model = model_utils.get_model(config, mode="panoptic")

    config.N_params = get_ntrainparams(model)
    with open(os.path.join(config.res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
    print("Trainable layers:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(config)
    quit()
    if hasattr(config, 'lr_scheduler') and config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,#intervenire qui correggendo i valori 
                                                   step_size=getattr(config, 'lr_step_size', 50), 
                                                   gamma=getattr(config, 'lr_gamma', 0.5))
        print(f"*****Using StepLR scheduler with step_size={getattr(config, 'lr_step_size', 15)} and gamma={getattr(config, 'lr_gamma', 0.3)}")
        quit()
    else:
        scheduler = None

    model.apply(weight_init)
    trainlog = {}
    start_epoch = 0

    from src.panoptic.paps_loss import PaPsLoss

    criterion = PaPsLoss(
        l_center=config.l_center,
        l_size=config.l_size,
        l_shape=config.l_shape,
        l_class=config.l_class,
        beta=config.beta,
        void_label=config.void_label,
    )

    best_pq = -1.0
    for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
        print("EPOCH {}/{}".format(epoch, config.epochs))
        heatmap_only = epoch - 1 < config.warmup if config.warmup > 0 else False
        model.train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="train",
            device=device,
            compute_metrics=epoch > config.warmup and not config.val_metrics_only,
            heatmap_only=heatmap_only,
        )

        trainlog[epoch] = {**train_metrics}
        if scheduler is not None:
            scheduler.step()
        if (
            epoch > config.warmup
            and epoch % config.val_every == 0
            and epoch > config.val_after
        ):
            print("Validation . . . ")
            model.eval()
            val_metrics = iterate(
                model,
                data_loader=val_loader,
                criterion=criterion,
                config=config,
                optimizer=optimizer,
                mode="val",
                device=device,
                compute_metrics=epoch > config.warmup,
                heatmap_only=heatmap_only,
                autotune=config.autotune,
            )
            trainlog[epoch].update(val_metrics)
            val_pq = val_metrics["val_PQ"]
            if val_pq > best_pq:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
                    ),
                )
                best_pq = val_pq
        checkpoint(fold + 1, trainlog, config)

    print("Testing best epoch . . .")
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.res_dir, "Fold_{}".format(fold + 1), "model.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()
    test_metrics, tables = iterate(
        model,
        data_loader=test_loader,
        criterion=criterion,
        config=config,
        optimizer=None,
        mode="test",
        device=device,
    )
    save_results(fold + 1, test_metrics, tables, config)

    # Plot opzionale della curva di validation loss dal trainlog.json del fold corrente
    if getattr(config, "plot_val_loss", False):
        fold_dir = os.path.join(config.res_dir, f"Fold_{fold + 1}")
        plot_validation_loss(fold_dir, title_suffix=f"Fold {fold + 1}")


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    pprint.pprint(config)
    main(config)