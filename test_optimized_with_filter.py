"""
Main script for panoptic experiments
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pprint
import sys
from pathlib import Path

# Fix encoding per Windows (evita errori con caratteri Unicode)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

import src.model_utils_optimized as model_utils
from src.dataset import PASTIS_Dataset
from src.panoptic.paps_loss import PaPsLoss
from src.utils import pad_collate
from sklearn.metrics import f1_score, accuracy_score
import joblib
import pickle

# Import ResNet per cloud detection
# Cerca il modulo resnet in vari percorsi possibili
RESNET_AVAILABLE = False
current_path = Path(__file__).resolve()

# Lista di percorsi possibili per resnet.py
possible_paths = [
    current_path.parent / "domain-adaptation-cloud-detection-main",
    current_path.parent.parent / "domain-adaptation-cloud-detection-main",
    current_path.parent.parent.parent / "domain-adaptation-cloud-detection-main",
    current_path.parent / "domain-adaptation-cloud-detection-main" / "domain-adaptation-cloud-detection-main",
    current_path.parent.parent / "domain-adaptation-cloud-detection-main" / "domain-adaptation-cloud-detection-main",
    current_path.parent.parent.parent / "domain-adaptation-cloud-detection-main" / "domain-adaptation-cloud-detection-main",
]

# Cerca resnet.py nei percorsi possibili
resnet_path = None
for path in possible_paths:
    resnet_file = path / "resnet.py"
    if resnet_file.exists():
        resnet_path = path
        sys.path.insert(0, str(path))
        print(f"[OK] Trovato resnet.py in: {path}")
        break

# Prova a importare
if resnet_path is not None:
    try:
        from resnet import ResNet18, ResNet34, ResNet50
        RESNET_AVAILABLE = True
        print(f"[OK] Modulo resnet importato con successo (ResNet18, ResNet34, ResNet50 disponibili)")
    except ImportError as e:
        RESNET_AVAILABLE = False
        print(f"[WARNING] Errore nell'importazione di resnet: {e}")
else:
    RESNET_AVAILABLE = False
    print("[WARNING] Modulo resnet non trovato. Filtro nuvole disabilitato.")
    print(f"   Percorsi cercati:")
    for path in possible_paths:
        print(f"     - {path}")


# ===============================
# FUNZIONI METRICHE "SAFE"
# ===============================
def safe_r2_score(y_true, y_pred):
    """Calcola R² in modo sicuro"""
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Rimuovi valori NaN e Inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.0
            
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        return max(-1.0, min(1.0, r2))  # Clamp tra -1 e 1
    except Exception as e:
        print(f"Errore nel calcolo R²: {e}")
        return 0.0


def safe_explained_variance(y_true, y_pred):
    """Calcola Explained Variance in modo sicuro"""
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Rimuovi valori NaN e Inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.0
            
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if denominator == 0:
            return 0.0
            
        explained_var = 1 - (numerator / denominator)
        return max(-1.0, min(1.0, explained_var))
    except Exception as e:
        print(f"Errore nel calcolo Explained Variance: {e}")
        return 0.0


def safe_correlation(y_true, y_pred):
    """Calcola correlazione in modo sicuro"""
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Rimuovi valori NaN e Inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 2:
            return 0.0
            
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return 0.0 if np.isnan(corr) else corr
    except Exception as e:
        print(f"Errore nel calcolo correlazione: {e}")
        return 0.0


def safe_mape(y_true, y_pred, epsilon=0.01, max_mape=100.0):
    """
    Calcola MAPE (Mean Absolute Percentage Error) in modo sicuro.
    
    Formula: MAPE = (1/n) * Σ |(yi - ŷi) / yi| * 100%
    
    Per valori normalizzati (NDVI tra -1 e 1), usa epsilon più grande e clamp.
    
    IMPORTANTE: Il MAPE può superare il 100% quando i valori target sono piccoli.
    Per limitare il MAPE a 100%, usiamo max_mape=100.0 e una strategia più conservativa:
    - Filtriamo valori target troppo piccoli (|y_true| < epsilon)
    - Clampiamo ogni errore percentuale individuale a 100%
    - Questo garantisce che il MAPE finale sia sempre <= 100%
    
    Args:
        y_true: Valori reali
        y_pred: Valori predetti
        epsilon: Soglia minima per il denominatore (default 0.01 per valori normalizzati)
        max_mape: Valore massimo per il MAPE (default 100.0 per limitare a 100%)
    
    Returns:
        MAPE in percentuale (float), clampato a max_mape (default 100%)
    """
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Rimuovi valori NaN e Inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.0
        
        # Per valori normalizzati, usa epsilon più grande (0.01 invece di 1e-8)
        # Filtra valori troppo piccoli per evitare divisioni per zero
        # Usa valore assoluto per gestire valori negativi (NDVI può essere negativo)
        abs_y_true = np.abs(y_true)
        valid_mask = abs_y_true > epsilon
        
        if valid_mask.sum() == 0:
            return 0.0
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        # Calcola MAPE: |(yi - ŷi) / yi| * 100%
        # Clamp ogni errore percentuale individuale a max_mape (default 100%) per evitare valori esplosivi
        percentage_errors = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100.0
        
        # Clamp errori percentuali individuali a max_mape (default 100%) per garantire MAPE <= 100%
        percentage_errors = np.clip(percentage_errors, 0.0, max_mape)
        
        # Rimuovi eventuali NaN o Inf rimasti
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        
        if len(percentage_errors) == 0:
            return 0.0
        
        mape = np.mean(percentage_errors)
        # Clamp finale per sicurezza (garantisce MAPE <= 100%)
        mape = min(mape, max_mape) if np.isfinite(mape) else 0.0
        return mape
    except Exception as e:
        print(f"Errore nel calcolo MAPE: {e}")
        return 0.0


def safe_smape(y_true, y_pred):
    """
    Calcola sMAPE (Symmetric Mean Absolute Percentage Error) in modo sicuro.
    
    Formula: sMAPE = (1/n) * Σ |yi - ŷi| / ((|yi| + |ŷi|) / 2) * 100%
    
    Il sMAPE è più stabile del MAPE quando i valori target sono piccoli o vicini a zero.
    Il sMAPE è sempre limitato tra 0% e 200%, ma tipicamente rimane sotto il 100%.
    
    Args:
        y_true: Valori reali
        y_pred: Valori predetti
    
    Returns:
        sMAPE in percentuale (float), sempre <= 200%
    """
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Rimuovi valori NaN e Inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.0
        
        # Calcola sMAPE: |yi - ŷi| / ((|yi| + |ŷi|) / 2) * 100%
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        
        # Evita divisione per zero (quando entrambi sono zero, errore è zero)
        valid_mask = denominator > 1e-8
        if valid_mask.sum() == 0:
            return 0.0
        
        numerator_valid = numerator[valid_mask]
        denominator_valid = denominator[valid_mask]
        
        percentage_errors = (numerator_valid / denominator_valid) * 100.0
        
        # Rimuovi eventuali NaN o Inf rimasti
        percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
        
        if len(percentage_errors) == 0:
            return 0.0
        
        smape = np.mean(percentage_errors)
        # Clamp a 200% (limite teorico del sMAPE)
        smape = min(smape, 200.0) if np.isfinite(smape) else 0.0
        return smape
    except Exception as e:
        print(f"Errore nel calcolo sMAPE: {e}")
        return 0.0


def calculate_error_percentages(y_true, y_pred, thresholds=[0.1, 0.2]):
    """Calcola percentuali di predizioni entro soglie di errore"""
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Rimuovi valori NaN e Inf
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {f'within_{int(t*100)}pct': 0.0 for t in thresholds}
        
        results = {}
        for threshold in thresholds:
            # Calcola errore percentuale con protezione da divisione per zero
            denominator = np.abs(y_true) + 1e-8  # Evita divisione per zero
            error_pct = np.abs((y_pred - y_true) / denominator)
            within_threshold = np.mean(error_pct <= threshold)
            results[f'within_{int(threshold*100)}pct'] = within_threshold
            
        return results
    except Exception as e:
        print(f"Errore nel calcolo error percentages: {e}")
        return {f'within_{int(t*100)}pct': 0.0 for t in thresholds}


def calculate_step_metrics(ndvi_pred, ndvi_target, stress_pred, stress_target, forecast_steps):
    """Calcola metriche per ogni step temporale"""
    step_metrics = {
        'ndvi_rmse': [],
        'ndvi_r2': [],
        'ndvi_corr': [],
        'stress_f1': [],
        'stress_acc': []
    }
    
    for step in range(forecast_steps):
        # NDVI metrics per questo step
        ndvi_step_pred = ndvi_pred[:, step, :, :].flatten()
        ndvi_step_target = ndvi_target[:, step, :, :].flatten()
        
        rmse = np.sqrt(np.mean((ndvi_step_pred - ndvi_step_target) ** 2))
        r2 = safe_r2_score(ndvi_step_target, ndvi_step_pred)
        corr = safe_correlation(ndvi_step_target, ndvi_step_pred)
        
        step_metrics['ndvi_rmse'].append(rmse)
        step_metrics['ndvi_r2'].append(r2)
        step_metrics['ndvi_corr'].append(corr)
        
        # Stress metrics per questo step
        stress_step_pred = stress_pred[:, step, :].flatten()
        stress_step_target = stress_target[:, step, :].flatten()
        
        # Converti in binary per metriche
        stress_pred_binary = (stress_step_pred > 0.5).astype(int)
        stress_target_binary = stress_step_target.astype(int)
        
        f1 = f1_score(stress_target_binary, stress_pred_binary, average='binary')
        acc = accuracy_score(stress_target_binary, stress_pred_binary)
        
        step_metrics['stress_f1'].append(f1)
        step_metrics['stress_acc'].append(acc)
    
    return step_metrics


# ===============================
# FUNZIONI PER CLOUD FILTER (ResNet50)
# ===============================
def load_cloud_detector(checkpoint_path, device, backbone='resnet50'):
    """
    Carica il modello ResNet cloud detector dal checkpoint.
    
    Args:
        checkpoint_path: Path al checkpoint del modello cloud detector
        device: Device su cui caricare il modello
        backbone: Nome del backbone ('resnet50', 'resnet18', 'resnet34', etc.)
    
    Returns:
        Modello ResNet in modalità eval
    """
    if not RESNET_AVAILABLE:
        raise ImportError("Modulo resnet non disponibile. Installa le dipendenze necessarie.")
    
    print(f"Caricando cloud detector da: {checkpoint_path}")
    
    # Crea modello ResNet per cloud detection (2 classi: clear/cloudy, 3 bande RGB)
    backbone_lower = backbone.lower()
    if backbone_lower == 'resnet50':
        cloud_model = ResNet50(num_classes=2, num_bands=3)
    elif backbone_lower == 'resnet18':
        cloud_model = ResNet18(num_classes=2, num_bands=3)
    elif backbone_lower == 'resnet34':
        cloud_model = ResNet34(num_classes=2, num_bands=3)
    else:
        raise ValueError(f"Backbone {backbone} non supportato. Usa 'resnet18', 'resnet34', o 'resnet50'.")
    
    # Carica checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Estrai state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Carica pesi
    cloud_model.load_state_dict(state_dict, strict=False)
    cloud_model = cloud_model.to(device)
    cloud_model.eval()
    
    print("[OK] Cloud detector caricato con successo")
    return cloud_model


def compute_rf_features_exact(s2_image):
    """
    Calcola le STESSE 27 feature esatte usate dal Random Forest nel notebook.
    Questa è la funzione identica a compute_features_from_xt nel notebook.
    
    Args:
        s2_image: Tensor (C, H, W) con 10 bande Sentinel-2
    
    Returns:
        numpy array di 27 feature (stesso ordine del Random Forest)
    """
    # Converti in numpy se necessario
    if isinstance(s2_image, torch.Tensor):
        x_t = s2_image.detach().cpu().numpy()
    else:
        x_t = s2_image
    
    C, H, W = x_t.shape
    
    feats = []
    
    # 1) mean/std per ogni banda (10 bande = 20 feature)
    for c in range(C):
        band = x_t[c]
        feats.append(band.mean())
        feats.append(band.std())
    
    # 2) NDVI: (NIR - RED) / (NIR + RED)
    red = x_t[3]  # B04
    nir = x_t[7]  # B08
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    feats.append(ndvi.mean())
    feats.append(ndvi.std())
    
    # frazioni NDVI in range particolari
    feats.append((ndvi < 0).mean())     # frazione NDVI negativi
    feats.append((ndvi > 0.7).mean())   # frazione NDVI molto alti
    
    # 3) indicatori sulla banda Blue (B02)
    blue = x_t[1]
    feats.append(blue.mean())
    feats.append(blue.std())
    feats.append((blue > 0.25).mean())  # soglia grezza per "molto chiaro"
    
    return np.array(feats, dtype=np.float32)


def compute_rgb_quality_features(s2_image, rgb_image):
    """
    Calcola feature statistiche correlate alla qualità del segnale RGB/Sentinel-2.
    Ora usa le STESSE feature del Random Forest + alcune aggiuntive per RGB.
    
    Args:
        s2_image: Tensor (C, H, W) con 10 bande Sentinel-2
        rgb_image: Tensor (3, H, W) con RGB convertito
    
    Returns:
        dict con feature statistiche (include le 27 feature RF + aggiuntive)
    """
    # Converti in numpy se necessario
    if isinstance(s2_image, torch.Tensor):
        s2_np = s2_image.detach().cpu().numpy()
    else:
        s2_np = s2_image
    
    if isinstance(rgb_image, torch.Tensor):
        rgb_np = rgb_image.detach().cpu().numpy()
    else:
        rgb_np = rgb_image
    
    features = {}
    
    # Calcola le 27 feature esatte del Random Forest
    rf_feats = compute_rf_features_exact(s2_image)
    
    # Aggiungi le feature RF con nomi espliciti
    C = s2_np.shape[0]
    idx = 0
    for c in range(C):
        features[f'band{c}_mean'] = float(rf_feats[idx])
        features[f'band{c}_std'] = float(rf_feats[idx+1])
        idx += 2
    
    features['ndvi_mean'] = float(rf_feats[idx])
    features['ndvi_std'] = float(rf_feats[idx+1])
    features['ndvi_negative_ratio'] = float(rf_feats[idx+2])
    features['ndvi_high_ratio'] = float(rf_feats[idx+3])
    idx += 4
    
    features['blue_mean'] = float(rf_feats[idx])
    features['blue_std'] = float(rf_feats[idx+1])
    features['blue_high_ratio'] = float(rf_feats[idx+2])
    
    # Aggiungi feature RGB aggiuntive (opzionali, per migliorare)
    rgb_std_per_pixel = rgb_np.std(axis=0)
    features['rgb_saturation_mean'] = float(rgb_std_per_pixel.mean())
    features['rgb_low_variance'] = float((rgb_std_per_pixel < 0.1).mean())
    
    # Calcola contrasto RGB (differenza tra max e min per pixel)
    rgb_max = rgb_np.max(axis=0)
    rgb_min = rgb_np.min(axis=0)
    rgb_contrast_per_pixel = rgb_max - rgb_min
    features['rgb_contrast'] = float(rgb_contrast_per_pixel.mean())
    
    # Calcola luminosità RGB (media dei 3 canali)
    rgb_brightness_per_pixel = rgb_np.mean(axis=0)
    features['rgb_brightness'] = float(rgb_brightness_per_pixel.mean())
    
    return features


def compute_statistical_quality_score(features):
    """
    Calcola uno score di qualità basato sulle feature statistiche.
    Score più alto = migliore qualità del segnale per forecasting.
    VERSIONE PIÙ RESTRITTIVA per essere più selettiva come il Random Forest.
    
    Args:
        features: dict con feature statistiche
    
    Returns:
        score: float tra 0 e 1 (1 = alta qualità, 0 = bassa qualità)
    """
    score = 1.0
    
    # 1) Penalizza NDVI negativo (stress/cloud) - MOLTO PIÙ RESTRITTIVO
    if features['ndvi_negative_ratio'] > 0.05:  # Era 0.1, ora 0.05 (molto più selettivo)
        penalty = min(0.9, features['ndvi_negative_ratio'] * 1.5)  # Penalità ancora più alta
        score *= (1.0 - penalty)
    
    # 2) Penalizza alta varianza NDVI (instabilità) - MOLTO PIÙ RESTRITTIVO
    if features['ndvi_std'] > 0.25:  # Era 0.3, ora 0.25 (più selettivo)
        penalty = min(0.7, (features['ndvi_std'] - 0.25) * 1.0)  # Penalità più alta
        score *= (1.0 - penalty)
    
    # 3) Penalizza blue molto alto (nuvole) - MOLTO PIÙ RESTRITTIVO
    if features['blue_high_ratio'] > 0.15:  # Era 0.2, ora 0.15 (più selettivo)
        penalty = min(0.8, features['blue_high_ratio'] * 1.2)  # Penalità più alta
        score *= (1.0 - penalty)
    
    # 4) Penalizza bassa varianza RGB (immagine piatta, nuvola) - MOLTO PIÙ RESTRITTIVO
    if features['rgb_low_variance'] > 0.25:  # Era 0.3, ora 0.25 (più selettivo)
        penalty = min(0.7, features['rgb_low_variance'] * 1.0)  # Penalità più alta
        score *= (1.0 - penalty)
    
    # 5) Penalizza basso contrasto RGB - SOGLIA PIÙ ALTA
    if features['rgb_contrast'] < 0.15:  # Era 0.1, ora 0.15
        penalty = 0.5 - (features['rgb_contrast'] / 0.15) * 0.3  # Penalità proporzionale
        score *= (1.0 - penalty)
    
    # 6) Penalizza NDVI medio molto basso (stress/cloud)
    if features['ndvi_mean'] < -0.1:
        penalty = min(0.5, abs(features['ndvi_mean']) * 2.0)
        score *= (1.0 - penalty)
    
    # 7) Penalizza alta luminosità RGB (tipico nuvole)
    if features['rgb_brightness'] > 0.6:
        penalty = min(0.4, (features['rgb_brightness'] - 0.6) * 1.0)
        score *= (1.0 - penalty)
    
    # 8) Penalizza bassa saturation RGB (immagine piatta)
    if features['rgb_saturation_mean'] < 0.1:
        penalty = 0.3
        score *= (1.0 - penalty)
    
    # 9) Bonus per NDVI alto (vegetazione sana) - RIDOTTO
    if features['ndvi_mean'] > 0.4:  # Era 0.3, ora 0.4
        score *= 1.05  # Era 1.1, ora 1.05 (bonus più conservativo)
    
    # 10) Penalizza varianza NDVI molto alta (instabilità estrema)
    if features['ndvi_std'] > 0.6:
        penalty = min(0.5, (features['ndvi_std'] - 0.6) * 1.5)
        score *= (1.0 - penalty)
    
    # Clamp tra 0 e 1
    score = max(0.0, min(1.0, score))
    
    return score


def convert_pastis_to_rgb(s2_image, normalize=True):
    """
    Converte un'immagine Sentinel-2 PASTIS (10 bande) in RGB.
    
    Bande Sentinel-2 PASTIS:
    - Banda 0: Blue
    - Banda 1: Green
    - Banda 2: Red
    - Banda 3-6: Red Edge 1-4
    - Banda 7: NIR
    - Banda 8: SWIR1
    - Banda 9: SWIR2
    
    Args:
        s2_image: Tensor (C, H, W) o (T, C, H, W) con 10 bande Sentinel-2
        normalize: Se True, normalizza i valori in [0, 1]
    
    Returns:
        Tensor RGB (3, H, W) o (T, 3, H, W) se input ha dimensione temporale
    """
    if s2_image.dim() == 4:
        # (T, C, H, W) - serie temporale
        T, C, H, W = s2_image.shape
        rgb = s2_image[:, [0, 1, 2], :, :]  # Blue, Green, Red
        if normalize:
            # Normalizza ogni banda separatamente
            for t in range(T):
                for c in range(3):
                    band = rgb[t, c, :, :]
                    min_val = band.min()
                    max_val = band.max()
                    if max_val > min_val:
                        rgb[t, c, :, :] = (band - min_val) / (max_val - min_val)
                    else:
                        rgb[t, c, :, :] = torch.zeros_like(band)
        return rgb
    elif s2_image.dim() == 3:
        # (C, H, W) - singola immagine
        C, H, W = s2_image.shape
        rgb = s2_image[[0, 1, 2], :, :]  # Blue, Green, Red
        if normalize:
            # Normalizza ogni banda separatamente
            for c in range(3):
                band = rgb[c, :, :]
                min_val = band.min()
                max_val = band.max()
                if max_val > min_val:
                    rgb[c, :, :] = (band - min_val) / (max_val - min_val)
                else:
                    rgb[c, :, :] = torch.zeros_like(band)
        return rgb
    else:
        raise ValueError(f"Formato immagine non supportato: {s2_image.shape}")


def detect_clouds_spatial(cloud_model, rgb_image, device, patch_size=128, stride=64, threshold=0.5):
    """
    Rileva nuvole a livello spaziale usando sliding window.
    
    Args:
        cloud_model: Modello ResNet50 cloud detector
        rgb_image: Tensor RGB (3, H, W) o (T, 3, H, W)
        device: Device per i calcoli
        patch_size: Dimensione delle patch per classificazione (default 128, resize a 512)
        stride: Stride per sliding window
        threshold: Soglia di probabilità per classificare come nuvoloso
    
    Returns:
        cloud_mask: Tensor binario (H, W) o (T, H, W) - 1 = nuvoloso, 0 = chiaro
        cloud_prob: Tensor probabilità (H, W) o (T, H, W)
    """
    cloud_model.eval()
    
    with torch.no_grad():
        if rgb_image.dim() == 4:
            # (T, 3, H, W) - serie temporale
            T, C, H, W = rgb_image.shape
            cloud_masks = []
            cloud_probs = []
            
            for t in range(T):
                mask, prob = detect_clouds_spatial(cloud_model, rgb_image[t], device, patch_size, stride, threshold)
                cloud_masks.append(mask)
                cloud_probs.append(prob)
            
            return torch.stack(cloud_masks, dim=0), torch.stack(cloud_probs, dim=0)
        
        elif rgb_image.dim() == 3:
            # (3, H, W) - singola immagine
            C, H, W = rgb_image.shape
            
            # Se l'immagine è già 128x128, usa direttamente
            if H == 128 and W == 128:
                # Resize a 512x512 per il modello (addestrato su 512x512)
                rgb_resized = F.interpolate(
                    rgb_image.unsqueeze(0), 
                    size=(512, 512), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # Classifica l'intera immagine
                rgb_batch = rgb_resized.unsqueeze(0).to(device)  # (1, 3, 512, 512)
                logits = cloud_model(rgb_batch)  # (1, 2)
                probs = F.softmax(logits, dim=1)  # (1, 2)
                cloud_prob_patch = probs[0, 1].item()  # Probabilità nuvoloso
                
                # Crea maschera uniforme per tutta l'immagine
                cloud_mask = torch.full((H, W), 1.0 if cloud_prob_patch > threshold else 0.0)
                cloud_prob = torch.full((H, W), cloud_prob_patch)
                
                return cloud_mask, cloud_prob
            else:
                # Sliding window per immagini più grandi
                cloud_mask = torch.zeros(H, W)
                cloud_prob = torch.zeros(H, W)
                count = torch.zeros(H, W)
                
                # Sliding window
                for y in range(0, H - patch_size + 1, stride):
                    for x in range(0, W - patch_size + 1, stride):
                        patch = rgb_image[:, y:y+patch_size, x:x+patch_size]  # (3, patch_size, patch_size)
                        
                        # Resize a 512x512
                        patch_resized = F.interpolate(
                            patch.unsqueeze(0), 
                            size=(512, 512), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)
                        
                        # Classifica
                        patch_batch = patch_resized.unsqueeze(0).to(device)
                        logits = cloud_model(patch_batch)
                        probs = F.softmax(logits, dim=1)
                        cloud_prob_patch = probs[0, 1].item()
                        
                        # Aggiorna maschera
                        cloud_mask[y:y+patch_size, x:x+patch_size] += (1.0 if cloud_prob_patch > threshold else 0.0)
                        cloud_prob[y:y+patch_size, x:x+patch_size] += cloud_prob_patch
                        count[y:y+patch_size, x:x+patch_size] += 1.0
                
                # Normalizza
                cloud_prob = cloud_prob / (count + 1e-8)
                cloud_mask = (cloud_prob > threshold).float()
                
                return cloud_mask, cloud_prob
        else:
            raise ValueError(f"Formato immagine non supportato: {rgb_image.shape}")


def compute_forecast_clear_mask(x_tensor, cloud_model, device, input_steps, forecast_steps, threshold=0.5, 
                                 use_statistical_features=True, statistical_weight=0.5, combination_mode="boosted",
                                 random_forest_clf=None):
    """
    x_tensor: (T, C, H, W), dove T dovrebbe essere = input_steps.
    
    Strategia:
      - guardiamo gli ULTIMI 2 input step (o meno, se input_steps < 2)
      - calcoliamo P(cloudy) per ciascuno usando ResNet50
      - calcoliamo feature statistiche del segnale RGB/Sentinel-2
      - combiniamo probabilità ResNet50 + score statistico
      - se QUALSIASI degli ultimi 2 è nuvoloso (score combinato > soglia) → consideriamo
        tutta la sequenza come "cloudy" e scartiamo TUTTI i forecast steps
      - altrimenti la sequenza è "clear" → teniamo TUTTI i forecast steps.
    
    Args:
        use_statistical_features: Se True, combina ResNet50 con feature statistiche
        statistical_weight: Peso delle feature statistiche (0.0-1.0). 
                           ResNet50 weight = 1.0 - statistical_weight
        combination_mode: Modalità di combinazione: 'linear' (media pesata), 'max' (massimo, molto selettivo), 'boosted' (compromesso)
      
    Output:
      clear_mask: array booleano di shape (forecast_steps,)
                  True  = step tenuto
                  False = step scartato
    """
    T = x_tensor.shape[0]

    # se per qualche motivo abbiamo meno step di quelli di input richiesti,
    # non applichiamo filtraggio (teniamo tutto)
    if T < input_steps:
        print(f"[WARNING] compute_forecast_clear_mask: T={T} < input_steps={input_steps}, uso tutti clear.")
        return np.ones(forecast_steps, dtype=bool)

    # scegliamo gli ultimi 2 input step (oppure 1 se input_steps == 1)
    start_t = max(0, input_steps - 2)
    used_ts = list(range(start_t, input_steps))

    resnet_probs = []
    statistical_scores = []
    combined_scores = []
    flags = []
    
    for t in used_ts:
        # Converti step in RGB
        step_s2 = x_tensor[t]  # (C, H, W)
        step_rgb = convert_pastis_to_rgb(step_s2, normalize=True)  # (3, H, W)
        
        # 1) Rileva nuvole usando ResNet50 (REQUIRED)
        if cloud_model is None:
            raise ValueError("cloud_model deve essere fornito per il filtro")
        
        cloud_mask_step, cloud_prob_step = detect_clouds_spatial(
            cloud_model, step_rgb, device, threshold=threshold
        )
        
        # Usa probabilità media come P(cloudy) da ResNet50
        if isinstance(cloud_prob_step, torch.Tensor):
            p_cloudy_resnet = cloud_prob_step.mean().item()
        else:
            p_cloudy_resnet = float(cloud_prob_step)
        
        resnet_probs.append(p_cloudy_resnet)
        
        # 2) Calcola feature statistiche e score di qualità
        if use_statistical_features:
            features = compute_rgb_quality_features(step_s2, step_rgb)
            quality_score = compute_statistical_quality_score(features)
            statistical_scores.append(quality_score)
            
            # Combina ResNet50 prob (cloudy) con quality score (clear)
            # ResNet50: alto = nuvoloso, basso = chiaro
            # Quality score: alto = buona qualità, basso = cattiva qualità
            # Score combinato: alto = nuvoloso/cattiva qualità, basso = chiaro/buona qualità
            
            # Converti quality_score in probabilità di problema (invertito)
            stat_problem_prob = 1.0 - quality_score
            
            # COMBINAZIONE: scegli modalità in base a combination_mode
            resnet_weight = 1.0 - statistical_weight
            
            if combination_mode == "linear":
                # Media pesata lineare
                combined_score = (resnet_weight * p_cloudy_resnet + 
                                 statistical_weight * stat_problem_prob)
            elif combination_mode == "max":
                # MAX: più conservativo, scarta se anche solo uno indica problema
                combined_score = max(p_cloudy_resnet, stat_problem_prob)
            elif combination_mode == "boosted":
                # BOOSTED: compromesso - se entrambi indicano problema, aumenta il peso
                combined_linear = (resnet_weight * p_cloudy_resnet + 
                                  statistical_weight * stat_problem_prob)
                if p_cloudy_resnet > 0.3 and stat_problem_prob > 0.3:
                    # Se entrambi > 0.3, aumenta il peso del più alto
                    combined_score = 0.6 * max(p_cloudy_resnet, stat_problem_prob) + 0.4 * combined_linear
                else:
                    combined_score = combined_linear
            else:
                # Default: linear
                combined_score = (resnet_weight * p_cloudy_resnet + 
                                 statistical_weight * stat_problem_prob)
            
            combined_scores.append(combined_score)
            
            # Usa score combinato per decisione
            final_score = combined_score
        else:
            # Usa solo ResNet50
            final_score = p_cloudy_resnet
            statistical_scores.append(None)
            combined_scores.append(None)
        
        flags.append(final_score > threshold)

    # regola: se ALMENO UNO degli ultimi step è nuvoloso → sequenza cloudy
    is_cloudy_seq = any(flags)

    if is_cloudy_seq:
        clear_mask = np.zeros(forecast_steps, dtype=bool)
    else:
        clear_mask = np.ones(forecast_steps, dtype=bool)

    # Debug ridotto (solo per primi batch o quando necessario)
    # Rimuovi questi print per migliorare le performance in produzione
    # resnet_str = ", ".join([f"{p:.3f}" for p in resnet_probs])
    # print(f"[DEBUG] Input steps usati = {used_ts}")
    # print(f"[DEBUG] ResNet50 P(cloudy) = [{resnet_str}]")
    # if use_statistical_features:
    #     stat_str = ", ".join([f"{s:.3f}" if s is not None else "N/A" for s in statistical_scores])
    #     comb_str = ", ".join([f"{c:.3f}" if c is not None else "N/A" for c in combined_scores])
    #     print(f"[DEBUG] Statistical quality = [{stat_str}]")
    #     print(f"[DEBUG] Combined score = [{comb_str}] (mode: {combination_mode}, weight: ResNet={1.0-statistical_weight:.2f}, Stat={statistical_weight:.2f})")
    # flags_str = ", ".join(["CLOUDY" if f else "CLEAR" for f in flags])
    # print(f"[DEBUG] Decisione = [{flags_str}] -> Sequenza {'CLOUDY' if is_cloudy_seq else 'CLEAR'} -> clear_mask = {clear_mask}")

    return clear_mask



def save_specific_predictions(model, test_loader, config, output_dir):
    """Salva predizioni specifiche per ogni patch"""
    
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Il formato del batch per forecasting è: ((x, dates), targets_dict)
            # dove targets_dict contiene: ndvi_target, ndvi_mean_target, stress_target, dates_target
            try:
                (x, dates), targets = batch
                x = x.to(config.device)
                dates = dates.to(config.device)
                
                # Estrai i target dal dizionario
                ndvi_target = targets['ndvi_target'].to(config.device)
                ndvi_mean_target = targets['ndvi_mean_target'].to(config.device)
                stress_target = targets['stress_target'].to(config.device)
                dates_target = targets['dates_target'].to(config.device)
                
            except Exception as e:
                print(f"ERROR: Errore nel parsing del batch {batch_idx}: {e}")
                print(f"Tipo batch: {type(batch)}")
                if isinstance(batch, tuple):
                    print(f"Lunghezza batch: {len(batch)}")
                    if len(batch) > 0:
                        print(f"Tipo batch[0]: {type(batch[0])}")
                        if isinstance(batch[0], tuple):
                            print(f"Lunghezza batch[0]: {len(batch[0])}")
                continue
            
            # Forward pass
            ndvi_map_pred, ndvi_mean_pred, stress_global_pred, stress_map_pred = model(x, batch_positions=dates, forecast=True)
            
            # Per ogni patch nel batch
            for i in range(x.shape[0]):
                patch_id = f"patch_{batch_idx}_{i}"
                
                # Calcola metriche per ogni step temporale
                step_predictions = {}
                for step in range(config.forecast_steps):
                    step_key = f"t+{step+1}"
                    
                    # NDVI predictions
                    ndvi_map_pred_step = ndvi_map_pred[i, step]  # (H, W)
                    ndvi_mean_pred_step = ndvi_mean_pred[i, step]  # (1,)
                    stress_global_pred_step = stress_global_pred[i, step]  # (1,)
                    stress_map_pred_step = stress_map_pred[i, step]  # (H, W)
                    
                    # NDVI targets
                    ndvi_target_step = ndvi_target[i, step]  # (H, W)
                    ndvi_mean_target_step = ndvi_mean_target[i, step]  # (1,)
                    stress_target_step = stress_target[i, step]  # (1,)
                    
                    step_predictions[step_key] = {
                        'ndvi_map_prediction': ndvi_map_pred_step.cpu().numpy().tolist(),
                        'ndvi_mean_prediction': float(ndvi_mean_pred_step.cpu().numpy().item()),
                        'stress_global_prediction': float(stress_global_pred_step.cpu().numpy().item()),
                        'stress_map_prediction': float(stress_map_pred_step.mean().cpu().numpy()),
                        'ndvi_target': float(ndvi_target_step.mean().cpu().numpy()),
                        'ndvi_mean_target': float(ndvi_mean_target_step.cpu().numpy().item()),
                        'stress_target': float(stress_target_step.cpu().numpy().item()),
                        'ndvi_map_error': float(abs(ndvi_map_pred_step.mean() - ndvi_target_step.mean()).cpu().numpy()),
                        'ndvi_mean_error': float(abs(ndvi_mean_pred_step - ndvi_mean_target_step).cpu().numpy().item()),
                        'stress_global_error': float(abs(stress_global_pred_step - stress_target_step).cpu().numpy().item()),
                        'stress_map_error': float(abs(stress_map_pred_step.mean() - (ndvi_target_step.mean() < config.ndvi_stress_threshold).float()).cpu().numpy())
                    }
                
                predictions[patch_id] = {
                    'input_dates': dates[i, :config.input_steps].cpu().numpy().tolist(),
                    'forecast_dates': dates_target[i, :config.forecast_steps].cpu().numpy().tolist(),
                    'predictions': step_predictions
                }
    
    # Salva in JSON
    output_file = os.path.join(output_dir, f"specific_predictions_fold{config.fold}_t+{config.forecast_steps}.json")
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"[OK] Predizioni specifiche salvate in: {output_file}")
    return predictions


# ===============================
# ARGPARSE
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./inference_paps",
    type=str,
    help="Path to directory where results are written."
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
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
    "--mode", 
    default="panoptic", 
    type=str, 
    help="Modalità di test: panoptic, semantic, forecasting"
)
parser.add_argument(
    "--input_steps", 
    default=4, 
    type=int, 
    help="Numero di step temporali di input per forecasting"
)
parser.add_argument(
    "--forecast_steps", 
    default=2, 
    type=int, 
    help="Numero di step temporali da prevedere"
)
parser.add_argument(
    "--ndvi_stress_threshold", 
    default=0.4, 
    type=float, 
    help="Soglia NDVI per stress binario"
)
parser.add_argument(
    "--cloud_detector_weights",
    default="",
    type=str,
    help="Path al checkpoint del modello ResNet cloud detector. Se vuoto, niente cloud filtering."
)
parser.add_argument(
    "--cloud_detector_backbone",
    default="resnet50",
    type=str,
    choices=["resnet18", "resnet34", "resnet50"],
    help="Backbone del cloud detector (default: resnet50)"
)
parser.add_argument(
    "--cloud_threshold",
    default=0.5,
    type=float,
    help="Soglia di probabilità per classificare come nuvoloso (default: 0.5)"
)
parser.add_argument(
    "--use_statistical_features",
    action="store_true",
    help="Combina ResNet50 con feature statistiche del segnale RGB/Sentinel-2 per migliorare il filtro"
)
parser.add_argument(
    "--statistical_weight",
    default=0.5,
    type=float,
    help="Peso delle feature statistiche nella combinazione con ResNet50 (default: 0.5, range: 0.0-1.0). ResNet50 weight = 1.0 - statistical_weight. Valori più alti (0.5-0.7) rendono il filtro più selettivo come il Random Forest"
)
parser.add_argument(
    "--combination_mode",
    default="boosted",
    type=str,
    choices=["linear", "max", "boosted"],
    help="Modalità di combinazione ResNet50 + statistiche: 'linear' (media pesata), 'max' (massimo, molto selettivo), 'boosted' (compromesso, default)"
)
parser.add_argument(
    "--random_forest_path",
    default="",
    type=str,
    help="Path al Random Forest addestrato (.pkl o .joblib). Se specificato, usa RF per statistiche invece di score manuale. Combina ResNet50 + RF."
)


def main(config):
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)

    device = torch.device(config.device)

    if config.mode == "forecasting":
        # Test forecasting mode
        print("=== TEST OPTIMIZED MODEL ===")
        
        # Configurazione per test solo sul fold 1
        test_fold = [1]
        fold = 0  # Indice per il fold 1
        
        # Carica dataset di test
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            input_steps=config.input_steps,
            forecast_steps=config.forecast_steps,
            ndvi_stress_threshold=config.ndvi_stress_threshold,
            target="forecasting",
        )
        
        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)
        print(f"Dataset di test: {len(dt_test)} campioni")
        
        test_loader = data.DataLoader(
            dt_test,
            batch_size=1,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=0,
        )
        
        # Carica modello forecasting
        model = model_utils.get_model(config, mode="forecasting")
        model = model.to(device)
        print("Modello forecasting caricato")
        
        # Carica pesi addestrati
        model_path = os.path.join(config.weight_folder, f"Fold_{config.fold}", "model.pth.tar")
        print(f"Caricando pesi da: {model_path}")
        
        # Verifica che il file esista
        if not os.path.exists(model_path):
            print(f"\n[ERROR] File modello non trovato: {model_path}")
            print(f"  weight_folder: {config.weight_folder}")
            print(f"  fold: {config.fold}")
            print(f"  Path completo cercato: {model_path}")
            print(f"\n  Suggerimenti:")
            print(f"  1. Verifica che il path weight_folder sia corretto")
            print(f"  2. Verifica che esista la cartella Fold_{config.fold} dentro weight_folder")
            print(f"  3. Verifica che il file model.pth.tar esista in quella cartella")
            print(f"  4. Controlla eventuali spazi nel path (es: 'panottica /results' -> 'panottica/results')")
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('Missing keys:', missing)
        print('Unexpected keys:', unexpected)
        model.eval()

        # Carica opzionalmente il cloud detector (ResNet50)
        cloud_model = None
        use_cloud_filter = getattr(config, "cloud_detector_weights", "") and RESNET_AVAILABLE
        use_statistical_features = getattr(config, "use_statistical_features", False)
        statistical_weight = getattr(config, "statistical_weight", 0.5)
        
        if use_cloud_filter:
            try:
                cloud_model = load_cloud_detector(
                    config.cloud_detector_weights,
                    device,
                    backbone=getattr(config, "cloud_detector_backbone", "resnet50")
                )
                print(f"[OK] Cloud detector caricato da: {config.cloud_detector_weights}")
                
                # Informazioni sul filtro
                combination_mode = getattr(config, "combination_mode", "boosted")
                if use_statistical_features:
                    print(f"[OK] Filtro cloud: ResNet50 + Feature Statistiche RGB/Sentinel-2")
                    print(f"   - Modalita combinazione: {combination_mode} (linear/media pesata, max/massimo, boosted/compromesso)")
                    print(f"   - Peso ResNet50: {1.0 - statistical_weight:.2f}")
                    print(f"   - Peso Feature Statistiche: {statistical_weight:.2f}")
                    print(f"   - Feature incluse: RGB (mean/std/contrast), NDVI, indicatori qualita")
                else:
                    print(f"[OK] Filtro cloud: Solo ResNet50 (feature statistiche disabilitate)")
            except Exception as e:
                print(f"[WARNING] Impossibile caricare il cloud detector: {e}")
                cloud_model = None
                use_cloud_filter = False
        else:
            if not RESNET_AVAILABLE:
                print("[WARNING] Modulo resnet non disponibile. Filtro nuvole disabilitato.")
            else:
                print("[INFO] Cloud detector non specificato. Filtro nuvole disabilitato.")
        
        # Metriche di test per ogni step temporale
        forecast_steps = config.forecast_steps
        
        # Metriche per ogni step temporale
        step_metrics = {
            'ndvi_loss': [0] * forecast_steps,
            'ndvi_mean_loss': [0] * forecast_steps,
            'stress_loss': [0] * forecast_steps,
            'rmse_ndvi': [0] * forecast_steps,
            'mae_ndvi': [0] * forecast_steps,
            'corr_ndvi': [0] * forecast_steps,
            'r2_ndvi': [0] * forecast_steps,
            'explained_var_ndvi': [0] * forecast_steps,
            'within_10pct_ndvi': [0] * forecast_steps,
            'within_20pct_ndvi': [0] * forecast_steps,
            'stress_accuracy': [0] * forecast_steps,
            'stress_precision': [0] * forecast_steps,
            'stress_recall': [0] * forecast_steps,
            'stress_f1': [0] * forecast_steps,
        }
        
        # Metriche totali (media su tutti gli step)
        total_ndvi_loss = 0
        total_stress_loss = 0
        total_stress_map_loss = 0
        total_ndvi_mean_loss = 0
        num_samples = 0
        
        # Accumuliamo TUTTI i dati per calcolo metriche globali (migliore per R², correlazione, etc.)
        all_ndvi_pred = []
        all_ndvi_target = []
        all_stress_pred = []
        all_stress_target = []
        
        # Metriche aggiuntive totali (per backward compatibility, ma useremo quelle globali)
        total_rmse_ndvi = 0
        total_mae_ndvi = 0
        total_mape_ndvi = 0
        total_corr_ndvi = 0
        total_r2_ndvi = 0
        total_explained_var_ndvi = 0
        total_within_10pct_ndvi = 0
        total_within_20pct_ndvi = 0
        total_stress_accuracy = 0
        total_stress_precision = 0
        total_stress_recall = 0
        total_stress_f1 = 0

        # 🔢 Conteggio sequenze clear/cloudy secondo il cloud classifier
        total_cloudy_sequences = 0
        total_clear_sequences = 0
        
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()
        
        print("\n=== INIZIO TEST FORECASTING ===")
        
        # Analisi preliminare dei dati
        print("=== ANALISI DATASET ===")
        print("DEBUG: Inizio analisi dataset...")
        stress_counts = 0
        total_stress_samples = 0
        
        # Analisi diverse definizioni di stress
        ndvi_stress_counts = 0
        ndvi_low_counts = 0
        ndvi_high_counts = 0
        ndvi_var_counts = 0
        
        # Analisi distribuzione NDVI
        all_ndvi_values = []
        all_ndvi_means = []
        all_ndvi_stds = []
        
        # Analisi dettagliata distribuzione stress
        stress_distribution = {0: 0, 1: 0}
        ndvi_mean_distribution = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                (x, dates), targets = batch
                stress_target = targets['stress_target']
                ndvi_target = targets['ndvi_target']
                
                # Conta i casi di stress con diverse definizioni
                stress_binary = (stress_target > 0.5).float()
                stress_counts += stress_binary.sum().item()
                total_stress_samples += stress_target.numel()
                
                # Analisi dettagliata distribuzione stress
                stress_binary_np = stress_binary.cpu().numpy()
                for val in stress_binary_np.flatten():
                    stress_distribution[int(val)] += 1
                
                # Analisi NDVI per diverse soglie
                ndvi_mean = ndvi_target.mean()
                ndvi_std = ndvi_target.std()
                
                # Raccogli tutti i valori NDVI per analisi
                all_ndvi_values.extend(ndvi_target.view(-1).cpu().numpy())
                all_ndvi_means.append(ndvi_mean.item())
                all_ndvi_stds.append(ndvi_std.item())
                ndvi_mean_distribution.append(ndvi_mean.item())
                
                # Stress basato su soglie NDVI diverse
                ndvi_stress = (ndvi_mean < 0.0).float()  # NDVI negativo (stress)
                ndvi_low = (ndvi_mean < -0.5).float()    # NDVI molto basso
                ndvi_high = (ndvi_mean > 0.5).float()    # NDVI molto alto
                ndvi_var = (ndvi_std > 0.3).float()      # Alta variabilità NDVI
                
                ndvi_stress_counts += ndvi_stress.sum().item()
                ndvi_low_counts += ndvi_low.sum().item()
                ndvi_high_counts += ndvi_high.sum().item()
                ndvi_var_counts += ndvi_var.sum().item()
                
                if batch_idx < 5:  # Analizza solo i primi 5 batch
                    print(f"Batch {batch_idx+1}:")
                    print(f"  - Stress target range: [{stress_target.min():.4f}, {stress_target.max():.4f}]")
                    print(f"  - Stress binary sum: {stress_binary.sum().item()}")
                    print(f"  - NDVI target range: [{ndvi_target.min():.4f}, {ndvi_target.max():.4f}]")
                    print(f"  - NDVI mean: {ndvi_mean:.4f}, std: {ndvi_std:.4f}")
                    print(f"  - NDVI stress (mean < 0): {ndvi_stress.sum().item()}")
                    print(f"  - NDVI low (mean < -0.5): {ndvi_low.sum().item()}")
                    print(f"  - NDVI high (mean > 0.5): {ndvi_high.sum().item()}")
                    print(f"  - NDVI var (std > 0.3): {ndvi_var.sum().item()}")
                    
                    # Analisi bande spettrali per stress
                    if 'x' in batch[0]:
                        x_tmp = batch[0][0]  # Input data
                        print(f"  - Bande spettrali range:")
                        for i, band_name in enumerate(['Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 'Red Edge 4', 'SWIR1', 'SWIR2']):
                            band_data = x_tmp[:, i, :, :]
                            print(f"    {band_name}: [{band_data.min():.4f}, {band_data.max():.4f}]")
        
        # Analisi distribuzione NDVI
        print("DEBUG: Creo array numpy...")
        all_ndvi_values = np.array(all_ndvi_values)
        all_ndvi_means = np.array(all_ndvi_means)
        all_ndvi_stds = np.array(all_ndvi_stds)
        
        print("DEBUG: Stampa analisi distribuzione...")
        print(f"\n=== ANALISI DISTRIBUZIONE NDVI ===")
        print(f"NDVI valori - Min: {all_ndvi_values.min():.4f}, Max: {all_ndvi_values.max():.4f}")
        print(f"NDVI valori - Mean: {all_ndvi_values.mean():.4f}, Std: {all_ndvi_values.std():.4f}")
        print(f"NDVI valori - Percentili: 10%={np.percentile(all_ndvi_values, 10):.4f}, 25%={np.percentile(all_ndvi_values, 25):.4f}, 50%={np.percentile(all_ndvi_values, 50):.4f}, 75%={np.percentile(all_ndvi_values, 75):.4f}, 90%={np.percentile(all_ndvi_values, 90):.4f}")
        
        print(f"\nNDVI medie - Min: {all_ndvi_means.min():.4f}, Max: {all_ndvi_means.max():.4f}")
        print(f"NDVI medie - Mean: {all_ndvi_means.mean():.4f}, Std: {all_ndvi_means.std():.4f}")
        print(f"NDVI medie - Percentili: 10%={np.percentile(all_ndvi_means, 10):.4f}, 25%={np.percentile(all_ndvi_means, 25):.4f}, 50%={np.percentile(all_ndvi_means, 50):.4f}, 75%={np.percentile(all_ndvi_means, 75):.4f}, 90%={np.percentile(all_ndvi_means, 90):.4f}")
        
        # Test diverse soglie per stress
        thresholds = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        print(f"\n=== TEST DIVERSE SOGLIE STRESS ===")
        for threshold in thresholds:
            stress_count = (all_ndvi_means < threshold).sum()
            stress_percent = stress_count / len(all_ndvi_means) * 100
            print(f"Soglia NDVI < {threshold:.1f}: {stress_count} casi ({stress_percent:.1f}%)")
        
        print(f"\n=== STATISTICHE STRESS ===")
        print(f"Totale casi di stress (originale): {stress_counts}")
        print(f"Totale campioni: {total_stress_samples}")
        print(f"Percentuale stress originale: {stress_counts/total_stress_samples*100:.2f}%")
        
        print(f"\n=== DISTRIBUZIONE DETTAGLIATA STRESS ===")
        print(f"Classe 0 (No Stress): {stress_distribution[0]} ({stress_distribution[0]/total_stress_samples*100:.2f}%)")
        print(f"Classe 1 (Stress): {stress_distribution[1]} ({stress_distribution[1]/total_stress_samples*100:.2f}%)")
        print(f"Ratio Stress/No-Stress: {stress_distribution[1]/max(stress_distribution[0], 1):.3f}")
        
        print(f"\n=== ANALISI NDVI MEAN DISTRIBUZIONE ===")
        ndvi_mean_distribution = np.array(ndvi_mean_distribution)
        print(f"NDVI Mean - Min: {ndvi_mean_distribution.min():.4f}, Max: {ndvi_mean_distribution.max():.4f}")
        print(f"NDVI Mean - Mean: {ndvi_mean_distribution.mean():.4f}, Std: {ndvi_mean_distribution.std():.4f}")
        print(f"NDVI Mean - Percentili: 5%={np.percentile(ndvi_mean_distribution, 5):.4f}, 10%={np.percentile(ndvi_mean_distribution, 10):.4f}, 25%={np.percentile(ndvi_mean_distribution, 25):.4f}, 50%={np.percentile(ndvi_mean_distribution, 50):.4f}, 75%={np.percentile(ndvi_mean_distribution, 75):.4f}, 90%={np.percentile(ndvi_mean_distribution, 90):.4f}, 95%={np.percentile(ndvi_mean_distribution, 95):.4f}")
        
        print(f"\n=== NUOVE DEFINIZIONI STRESS ===")
        print(f"NDVI stress (mean < 0): {ndvi_stress_counts}")
        print(f"NDVI low (mean < -0.5): {ndvi_low_counts}")
        print(f"NDVI high (mean > 0.5): {ndvi_high_counts}")
        print(f"NDVI var (std > 0.3): {ndvi_var_counts}")
        print(f"Percentuale NDVI stress: {ndvi_stress_counts/total_stress_samples*100:.2f}%")
        print(f"Percentuale NDVI low: {ndvi_low_counts/total_stress_samples*100:.2f}%")
        print(f"Percentuale NDVI high: {ndvi_high_counts/total_stress_samples*100:.2f}%")
        print(f"Percentuale NDVI var: {ndvi_var_counts/total_stress_samples*100:.2f}%")
        
        # Reset del dataloader per il test completo
        test_loader = data.DataLoader(
            dt_test,
            batch_size=1,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=0,
        )
        
        print("\n=== INIZIO TEST COMPLETO ===")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                (x, dates), targets = batch
                x = x.to(device)
                dates = dates.to(device)
                ndvi_target = targets['ndvi_target'].to(device)
                stress_target = targets['stress_target'].to(device)
                ndvi_mean_target = targets['ndvi_mean_target'].to(device)
                
                # Forward pass
                ndvi_map_pred, ndvi_mean_pred, stress_global_pred, stress_map_pred = model(x, batch_positions=dates, forecast=True)
                
                # ============================
                # NDVI + STRESS con CLOUD FILTER (ResNet REQUIRED)
                # ============================
                skip_sample = False
                if cloud_model is not None:
                    # x ha shape (1, T, C, H, W)
                    x_cpu = x[0].cpu()  # (T,C,H,W)
                    clear_mask = compute_forecast_clear_mask(
                        x_cpu,
                        cloud_model,
                        device,
                        config.input_steps,
                        config.forecast_steps,
                        threshold=getattr(config, "cloud_threshold", 0.5),
                        use_statistical_features=getattr(config, "use_statistical_features", False),
                        statistical_weight=getattr(config, "statistical_weight", 0.5),
                        combination_mode=getattr(config, "combination_mode", "boosted")
                    )

                    if clear_mask.sum() == 0:
                        # tutta la sequenza è considerata cloudy → NON usiamo questa patch
                        total_cloudy_sequences += 1
                        skip_sample = True
                    else:
                        # sequenza considerata utilizzabile (clear) → applichiamo la mask
                        total_clear_sequences += 1

                        ndvi_pred_sel = ndvi_map_pred[:, clear_mask, :, :].contiguous()
                        ndvi_target_sel = ndvi_target[:, clear_mask, :, :].contiguous()
                        ndvi_mean_pred_sel = ndvi_mean_pred[:, clear_mask].contiguous()
                        ndvi_mean_target_sel = ndvi_mean_target[:, clear_mask].contiguous()
                        ndvi_pred_flat = ndvi_pred_sel.view(-1).cpu().numpy()
                        ndvi_target_flat = ndvi_target_sel.view(-1).cpu().numpy()

                        stress_pred_sel = stress_global_pred.squeeze(-1)[:, clear_mask].contiguous()
                        stress_target_sel = stress_target.squeeze(-1)[:, clear_mask].contiguous()
                        stress_map_pred_sel = stress_map_pred[:, clear_mask, :, :].contiguous()
                        stress_pred_vec = stress_pred_sel.view(-1)
                        stress_target_vec = stress_target_sel.view(-1)
                        
                        # Calcola loss SOLO sugli step filtrati (per coerenza con le metriche)
                        loss_ndvi = mse_loss(ndvi_pred_sel, ndvi_target_sel)
                        loss_ndvi_mean = mse_loss(ndvi_mean_pred_sel, ndvi_mean_target_sel)
                        loss_stress_global = bce_loss(stress_pred_sel, stress_target_sel)
                        stress_map_target_sel = (ndvi_target_sel < config.ndvi_stress_threshold).float()
                        loss_stress_map = bce_loss(stress_map_pred_sel, stress_map_target_sel)
                else:
                    # Nessun cloud detector: usiamo tutti gli step come prima
                    ndvi_pred_flat = ndvi_map_pred.view(-1).cpu().numpy()
                    ndvi_target_flat = ndvi_target.view(-1).cpu().numpy()
                    stress_pred_vec = stress_global_pred.squeeze(-1).view(-1)
                    stress_target_vec = stress_target.squeeze(-1).view(-1)
                    
                    # Calcola loss su tutti gli step
                    loss_ndvi = mse_loss(ndvi_map_pred, ndvi_target)
                    loss_ndvi_mean = mse_loss(ndvi_mean_pred, ndvi_mean_target)
                    loss_stress_global = bce_loss(stress_global_pred.squeeze(-1), stress_target.squeeze(-1))
                    stress_map_target = (ndvi_target < config.ndvi_stress_threshold).float()
                    loss_stress_map = bce_loss(stress_map_pred, stress_map_target)
                    
                # Se la sequenza è completamente cloudy → salta la patch
                if skip_sample:
                    continue

                
                # Accumula dati per calcolo metriche globali (MIGLIORE per R², correlazione, etc.)
                all_ndvi_pred.append(ndvi_pred_flat)
                all_ndvi_target.append(ndvi_target_flat)
                all_stress_pred.append(stress_pred_vec.cpu().numpy())
                all_stress_target.append(stress_target_vec.cpu().numpy())
                
                # Calcola metriche per batch (per backward compatibility e debug)
                rmse_ndvi = np.sqrt(np.mean((ndvi_pred_flat - ndvi_target_flat) ** 2))
                mae_ndvi = np.mean(np.abs(ndvi_pred_flat - ndvi_target_flat))
                mape_ndvi = safe_mape(ndvi_target_flat, ndvi_pred_flat)
                corr_ndvi = safe_correlation(ndvi_target_flat, ndvi_pred_flat)
                r2_ndvi = safe_r2_score(ndvi_target_flat, ndvi_pred_flat)
                explained_var_ndvi = safe_explained_variance(ndvi_target_flat, ndvi_pred_flat)
                error_pct_ndvi = calculate_error_percentages(ndvi_target_flat, ndvi_pred_flat)
                
                # Metriche per stress globale (binario)
                stress_pred_binary = (stress_pred_vec > 0.5).float()
                stress_target_binary = stress_target_vec
                stress_accuracy = (stress_pred_binary == stress_target_binary).float().mean()
                tp = (stress_pred_binary * stress_target_binary).sum()
                fp = (stress_pred_binary * (1 - stress_target_binary)).sum()
                fn = ((1 - stress_pred_binary) * stress_target_binary).sum()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                # Accumula loss e metriche per batch (useremo quelle globali alla fine)
                total_ndvi_loss += loss_ndvi.item()
                total_ndvi_mean_loss += loss_ndvi_mean.item()
                total_stress_loss += loss_stress_global.item()
                total_stress_map_loss += loss_stress_map.item()
                total_rmse_ndvi += rmse_ndvi
                total_mae_ndvi += mae_ndvi
                total_mape_ndvi += mape_ndvi
                total_corr_ndvi += corr_ndvi
                total_r2_ndvi += r2_ndvi
                total_explained_var_ndvi += explained_var_ndvi
                total_within_10pct_ndvi += error_pct_ndvi['within_10pct']
                total_within_20pct_ndvi += error_pct_ndvi['within_20pct']
                total_stress_accuracy += stress_accuracy.item()
                total_stress_precision += precision.item()
                total_stress_recall += recall.item()
                total_stress_f1 += f1.item()
                num_samples += 1
                
                # Debug solo ogni 50 batch
                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx} - NDVI Target mean: {ndvi_target_flat.mean():.4f}, Pred mean: {ndvi_pred_flat.mean():.4f}, R² batch: {r2_ndvi:.4f}")
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx+1}/{len(test_loader)} - "
                          f"NDVI Loss: {loss_ndvi.item():.4f}, "
                          f"NDVI Mean Loss: {loss_ndvi_mean.item():.4f}, "
                          f"Stress Global Loss: {loss_stress_global.item():.4f}, "
                          f"Stress Map Loss: {loss_stress_map.item():.4f}")
        
        # ===============================
        #   CLOUD FILTERING SUMMARY
        # ===============================
        print("\n=== CLOUDY SUMMARY ===")
        print("Num cloudy forecast sequences:", total_cloudy_sequences)
        print("Num clear  forecast sequences:", total_clear_sequences)
        print("================================\n")
        
        # Calcola metriche finali GLOBALI (su tutti i dati concatenati) - MIGLIORE per R², correlazione, etc.
        if len(all_ndvi_pred) > 0:
            # Concatena tutti i dati
            all_ndvi_pred_global = np.concatenate(all_ndvi_pred, axis=0)
            all_ndvi_target_global = np.concatenate(all_ndvi_target, axis=0)
            all_stress_pred_global = np.concatenate(all_stress_pred, axis=0)
            all_stress_target_global = np.concatenate(all_stress_target, axis=0)
            
            print(f"\n[INFO] Calcolo metriche GLOBALI (corrette) su {len(all_ndvi_pred_global)} valori totali aggregati")
            print(f"  (Prima: calcolo per batch e media -> Sbagliato!)")
            print(f"  (Ora: aggregazione totale e calcolo globale -> Corretto!)")
            
            # Calcola metriche globali (più accurate)
            avg_rmse_ndvi = np.sqrt(np.mean((all_ndvi_pred_global - all_ndvi_target_global) ** 2))
            avg_mae_ndvi = np.mean(np.abs(all_ndvi_pred_global - all_ndvi_target_global))
            avg_mape_ndvi = safe_mape(all_ndvi_target_global, all_ndvi_pred_global)
            avg_corr_ndvi = safe_correlation(all_ndvi_target_global, all_ndvi_pred_global)
            avg_r2_ndvi = safe_r2_score(all_ndvi_target_global, all_ndvi_pred_global)
            avg_explained_var_ndvi = safe_explained_variance(all_ndvi_target_global, all_ndvi_pred_global)
            error_pct_ndvi_global = calculate_error_percentages(all_ndvi_target_global, all_ndvi_pred_global)
            avg_within_10pct_ndvi = error_pct_ndvi_global['within_10pct']
            avg_within_20pct_ndvi = error_pct_ndvi_global['within_20pct']
            
            print(f"  R² globale (corretto): {avg_r2_ndvi:.4f}")
            print(f"  Correlazione globale (corretta): {avg_corr_ndvi:.4f}")
            print(f"  Explained Variance globale (corretta): {avg_explained_var_ndvi:.4f}")
            print(f"  MAPE globale (corretto): {avg_mape_ndvi:.4f}%\n")
            
            # Metriche stress globali
            stress_pred_binary_global = (all_stress_pred_global > 0.5).astype(float)
            stress_target_binary_global = all_stress_target_global.astype(float)
            avg_stress_accuracy = np.mean(stress_pred_binary_global == stress_target_binary_global)
            tp = np.sum(stress_pred_binary_global * stress_target_binary_global)
            fp = np.sum(stress_pred_binary_global * (1 - stress_target_binary_global))
            fn = np.sum((1 - stress_pred_binary_global) * stress_target_binary_global)
            avg_stress_precision = tp / (tp + fp + 1e-8)
            avg_stress_recall = tp / (tp + fn + 1e-8)
            avg_stress_f1 = 2 * avg_stress_precision * avg_stress_recall / (avg_stress_precision + avg_stress_recall + 1e-8)
        else:
            # Fallback (non dovrebbe succedere)
            avg_rmse_ndvi = 0.0
            avg_mae_ndvi = 0.0
            avg_mape_ndvi = 0.0
            avg_corr_ndvi = 0.0
            avg_r2_ndvi = 0.0
            avg_explained_var_ndvi = 0.0
            avg_within_10pct_ndvi = 0.0
            avg_within_20pct_ndvi = 0.0
            avg_stress_accuracy = 0.0
            avg_stress_precision = 0.0
            avg_stress_recall = 0.0
            avg_stress_f1 = 0.0
        
        # Loss medie (calcolate per batch, ok così)
        avg_ndvi_loss = total_ndvi_loss / num_samples if num_samples > 0 else 0.0
        avg_ndvi_mean_loss = total_ndvi_mean_loss / num_samples if num_samples > 0 else 0.0
        avg_stress_loss = total_stress_loss / num_samples if num_samples > 0 else 0.0
        avg_stress_map_loss = total_stress_map_loss / num_samples if num_samples > 0 else 0.0
        
        print("\n=== RISULTATI TEST FORECASTING ===")
        print(f"Numero di campioni testati: {num_samples}")
        print(f"\n--- Metriche NDVI (Regressione) ---")
        print(f"NDVI Map Loss media: {avg_ndvi_loss:.4f}")
        print(f"NDVI Mean Loss media: {avg_ndvi_mean_loss:.4f}")
        print(f"NDVI RMSE media: {avg_rmse_ndvi:.4f}")
        print(f"NDVI MAE media: {avg_mae_ndvi:.4f}")
        print(f"NDVI MAPE media: {avg_mape_ndvi:.4f}%")
        print(f"NDVI Correlazione globale (corretta): {avg_corr_ndvi:.4f}")
        print(f"NDVI R² globale (corretto): {avg_r2_ndvi:.4f}")
        print(f"NDVI Explained Variance globale (corretta): {avg_explained_var_ndvi:.4f}")
        print(f"NDVI Predizioni entro 10% errore: {avg_within_10pct_ndvi:.2%}")
        print(f"NDVI Predizioni entro 20% errore: {avg_within_20pct_ndvi:.2%}")
        print(f"\n--- Metriche Stress (Classificazione) ---")
        print(f"Stress Global Loss media: {avg_stress_loss:.4f}")
        print(f"Stress Map Loss media: {avg_stress_map_loss:.4f}")
        print(f"Stress Accuracy media: {avg_stress_accuracy:.4f}")
        print(f"Stress Precision media: {avg_stress_precision:.4f}")
        print(f"Stress Recall media: {avg_stress_recall:.4f}")
        print(f"Stress F1-Score media: {avg_stress_f1:.4f}")
        print(f"\n--- Metriche Totali ---")
        print(f"Loss totale media: {avg_ndvi_loss + avg_ndvi_mean_loss + avg_stress_loss + avg_stress_map_loss:.4f}")
        
        # Salva risultati
        def convert_numpy_to_python(obj):
            """Converte oggetti numpy in tipi Python standard per JSON"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return obj

        results = {
            "num_samples": num_samples,
            "ndvi_map_loss": avg_ndvi_loss,
            "ndvi_mean_loss": avg_ndvi_mean_loss,
            "stress_global_loss": avg_stress_loss,
            "stress_map_loss": avg_stress_map_loss,
            "total_loss": avg_ndvi_loss + avg_ndvi_mean_loss + avg_stress_loss + avg_stress_map_loss,
            "ndvi_rmse": avg_rmse_ndvi,
            "ndvi_mae": avg_mae_ndvi,
            "ndvi_mape": avg_mape_ndvi,
            "ndvi_correlation": avg_corr_ndvi,
            "ndvi_r2": avg_r2_ndvi,
            "ndvi_explained_variance": avg_explained_var_ndvi,
            "ndvi_within_10pct": avg_within_10pct_ndvi,
            "ndvi_within_20pct": avg_within_20pct_ndvi,
            "stress_accuracy": avg_stress_accuracy,
            "stress_precision": avg_stress_precision,
            "stress_recall": avg_stress_recall,
            "stress_f1": avg_stress_f1,
        }

        # Converti tutti i valori
        results = convert_numpy_to_python(results)

        # Salva in file JSON
        results_path = os.path.join(config.res_dir, "test_forecasting_results.json")
        os.makedirs(config.res_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"[OK] Risultati salvati in: {results_path}")
        
        # Dopo il test normale, salva predizioni specifiche
        specific_predictions = save_specific_predictions(model, test_loader, config, config.res_dir)
        return
        
    # ===============================
    # Codice originale per panoptic/semantic
    # ===============================
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )

    model = model_utils.get_model(config, mode="panoptic")
    model = model.to(device)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1
        dt_args = dict(
            folder=config.dataset_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="instance",
        )

        dt_test = PASTIS_Dataset(**dt_args, folds=test_fold)

        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=pad_collate,
            num_workers=config.num_workers,
        )

        # Load weights
        sd = torch.load(
            os.path.join(config.weight_folder, "Fold_{}".format(fold+1), "model.pth.tar"),
            map_location=device,
        )
        model.load_state_dict(sd["state_dict"])

        criterion = PaPsLoss(
            l_center=config.l_center,
            l_size=config.l_size,
            l_shape=config.l_shape,
            l_class=config.l_class,
            beta=config.beta,
            void_label=config.void_label,
        )
        print("Testing . . .")
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


if __name__ == "__main__":
    test_config = parser.parse_args()

    # Pulisci i percorsi da eventuali caratteri strani o virgolette
    if hasattr(test_config, 'weight_folder') and test_config.weight_folder:
        # Rimuovi virgolette doppie e spazi iniziali/finali, ma mantieni spazi interni al path
        test_config.weight_folder = test_config.weight_folder.strip('"\' ')
    
    if hasattr(test_config, 'dataset_folder') and test_config.dataset_folder:
        # Rimuovi virgolette doppie e spazi iniziali/finali, ma mantieni spazi interni al path
        test_config.dataset_folder = test_config.dataset_folder.strip('"\' ')
    
    if hasattr(test_config, 'res_dir') and test_config.res_dir:
        # Rimuovi virgolette doppie e spazi iniziali/finali, ma mantieni spazi interni al path
        test_config.res_dir = test_config.res_dir.strip('"\' ')

    # Prova a caricare conf.json, altrimenti usa valori di default
    try:
        conf_path = os.path.join(test_config.weight_folder, "conf.json")
        with open(conf_path) as file:
            model_config = json.loads(file.read())
    except FileNotFoundError:
        print("[WARNING] conf.json non trovato, uso valori di default")
        model_config = {
            "encoder_widths": [64, 64, 64, 128],
            "decoder_widths": [32, 32, 64, 128],
            "out_conv": [32, 20],
            "str_conv_k": 4,
            "str_conv_s": 2,
            "str_conv_p": 1,
            "agg_mode": "att_group",
            "encoder_norm": "group",
            "n_head": 16,
            "d_model": 256,
            "d_k": 4,
            "pad_value": 0,
            "padding_mode": "reflect",
            "ref_date": "2018-09-01",
            "mono_date": None,
            "rdm_seed": 1
        }

    config = {**model_config, **vars(test_config)}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold
    
    # Aggiungi parametri mancanti per forecasting
    if not hasattr(config, 'mode'):
        config.mode = 'panoptic'
    if not hasattr(config, 'input_steps'):
        config.input_steps = 4
    if not hasattr(config, 'forecast_steps'):
        config.forecast_steps = 2
    if not hasattr(config, 'ndvi_stress_threshold'):
        config.ndvi_stress_threshold = 0.4
    if not hasattr(config, 'encoder_widths'):
        config.encoder_widths = [64, 64, 64, 128]
    if not hasattr(config, 'decoder_widths'):
        config.decoder_widths = [32, 32, 64, 128]
    if not hasattr(config, 'out_conv'):
        config.out_conv = [32, 20]
    if not hasattr(config, 'str_conv_k'):
        config.str_conv_k = 4
    if not hasattr(config, 'str_conv_s'):
        config.str_conv_s = 2
    if not hasattr(config, 'str_conv_p'):
        config.str_conv_p = 1
    if not hasattr(config, 'agg_mode'):
        config.agg_mode = 'att_group'
    if not hasattr(config, 'encoder_norm'):
        config.encoder_norm = 'group'
    if not hasattr(config, 'n_head'):
        config.n_head = 16
    if not hasattr(config, 'd_model'):
        config.d_model = 256
    if not hasattr(config, 'd_k'):
        config.d_k = 4
    if not hasattr(config, 'pad_value'):
        config.pad_value = 0
    if not hasattr(config, 'padding_mode'):
        config.padding_mode = 'reflect'

    pprint.pprint(config)
    main(config)
