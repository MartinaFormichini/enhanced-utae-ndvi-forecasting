# Enhanced U-TAE for multi-step NDVI and vegetation-condition forecasting

Code accompanying the article:

> M. Formichini, C. A. Avizzano. *Enhanced U-Temporal Attention Encoder for Multi-Step NDVI and Vegetation-Condition Forecasting from Sentinel-2 Time Series.* Sensors, 2026 (under review).

The model adapts the U-TAE architecture (Sainte Fare Garnot & Landrieu, ICCV 2021), originally developed for semantic and panoptic segmentation of Sentinel-2 image time series, to spatially explicit multi-step NDVI forecasting on the PASTIS dataset. It adds:

- a **Minimal Spatial Attention (MSA)** module, applied after the U-TAE spatial decoder on downsampled (32 × 32) feature maps;
- a **multi-task forecasting head** with four branches: pixel-wise NDVI maps, patch-level mean NDVI, a patch-level three-class vegetation-condition class and a pixel-wise three-class vegetation-condition map.

The vegetation-condition classes are derived from fixed NDVI thresholds (0.1 and 0.4). They describe NDVI-based canopy condition and are **not** validated measurements of physiological stress.

---

## Repository structure

```
utae-paps-main-optimizationlossndvi1-ternarystress-withfilter/
├── train_panoptic.py            # training script (use --mode forecasting)
├── test_optimized_with_filter.py
└── src/
    ├── dataset.py               # PASTIS loader, NDVI computation, forecasting windows, targets
    ├── model_utils_minimal.py   # Enhanced U-TAE used by train_panoptic.py
    ├── model_utils_optimized.py # same architecture, imported by test_optimized_with_filter.py
    ├── model_utils_constrained.py, model_utils.py, model_utils_convlstm.py  # other variants
    ├── utils.py
    ├── backbones/               # U-TAE, L-TAE and other backbones (from the original U-TAE code)
    ├── learning/                # metrics and weight initialisation
    └── panoptic/                # PaPs code inherited from the original repository (not used for forecasting)
```

### Correspondence between the code and the article

| Article | Code |
|---|---|
| U-TAE backbone (encoder widths 64-64-64-128, decoder widths 32-32-64-128, L-TAE with 16 heads) | `src/backbones/utae.py`, `src/backbones/ltae.py`; defaults in `train_panoptic.py` |
| Minimal Spatial Attention (Section 2.4) | class `MinimalSpatialAttention` in `src/model_utils_minimal.py` (4 heads, d_model = 64, downsampling 128 → 32). In the model it is stored in the attribute `head.temporal_attention`; the attribute name is kept for compatibility with the saved checkpoints. |
| Forecasting head with four branches (Section 2.5) | class `OptimizedForecastingHead` in `src/model_utils_minimal.py` |
| NDVI map branch | output `(B, K, H, W)`, K = number of forecast steps |
| Patch-level mean NDVI branch | output `(B, K, 1)` |
| Patch-level vegetation-condition branch | logits `(B, K, 3)` |
| Pixel-wise vegetation-condition branch | logits `(B, K, H, W, 3)` |
| NDVI computation and forecasting targets (Section 2.1) | `src/dataset.py` |
| Loss weights λ₁ = 1.0, λ₂ = 0.5, λ₃ = λ₄ = 0.3 (Section 2.6) | `--weight_ndvi_map`, `--weight_ndvi_mean`, `--weight_stress` (one value for both classification terms); these are the defaults |
| NDVI regression loss: MAE | `--ndvi_loss_type mae` (must be passed explicitly; the script default is `mse`) |

### NDVI and labels

- **Bands.** PASTIS provides 10 Sentinel-2 L2A bands in the order B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
- **NDVI.** NDVI is computed from the un-normalised reflectance as (B8A − B4) / (B8A + B4 + 10⁻⁴), i.e. channels 7 and 2, and clipped to [−1, 1]. The z-score normalisation with the per-fold PASTIS statistics is applied afterwards, and only to the 10 input channels. NDVI targets stay in physical units.
- **Targets.** The target at t₀ + 7k days is linearly interpolated per pixel between the two acquisitions that bracket that date.
- **Classes.** Vegetation-condition class indices: 0 = NDVI ≥ 0.4, 1 = 0.1 ≤ NDVI < 0.4, 2 = NDVI < 0.1. The patch-level class is obtained by applying the same thresholds to the patch-mean NDVI.

---

## Requirements

Python 3.10 or later with a CUDA-capable GPU. Main packages:

```
torch
torchnet
numpy
pandas
geopandas
scikit-learn
joblib
matplotlib
```

## Data

Download PASTIS from the official repository (https://github.com/VSainteuf/pastis-benchmark) or Zenodo (https://zenodo.org/records/5012942). Pass the dataset path with `--dataset_folder`. The official five-fold split of PASTIS is used.

---

## Training

One model is trained for each forecasting horizon k = 1, …, 8 (lead time 7k days) and each fold f = 1, …, 5. The example below uses the configuration reported in Table 4 of the article:

```bash
python train_panoptic.py \
  --mode forecasting \
  --dataset_folder /path/to/PASTIS \
  --res_dir ./results/FS<k> \
  --fold <f> \
  --input_steps 4 \
  --forecast_steps <k> \
  --forecast_lag_days 7 \
  --ndvi_loss_type mae \
  --weight_ndvi_map 1.0 --weight_ndvi_mean 0.5 --weight_stress 0.3 \
  --batch_size 8 \
  --epochs 150 \
  --lr 1e-4 --lr_scheduler step --lr_step_size 35 --lr_gamma 0.5 \
  --rdm_seed 1
```

Gradient clipping (max norm 1.0) is applied inside the training loop.

Two-stage training (Section 2.6):
1. **NDVI-only stage:** add `--ndvi_only_loss`.
2. **Fine-tuning with all four losses:** start from the selected checkpoint of stage 1 with `--resume_path <checkpoint>`.

## Evaluation

<!-- TO COMPLETE: name and usage of the evaluation script that produced the results of the article -->
The results reported in the article were obtained with a single evaluation script that:
- draws one forecast origin per test patch with a fixed seed;
- applies the fold-specific U-Net cloud detector to the acquisition closest to each target date (cloud probability threshold 0.40, minimum valid-pixel ratio 0.60);
- evaluates all models and the persistence baseline on identical windows and pixels.

```bash
python <evaluation_script>.py --dataset_folder /path/to/PASTIS --weight_folder ./results ...
```

---

## Pretrained weights

Not included. <!-- TO COMPLETE if weights are shared, e.g. on Zenodo -->

## Citation

If you use this code, please cite the article above and the original U-TAE paper:

```
@inproceedings{garnot2021panoptic,
  title     = {Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks},
  author    = {Sainte Fare Garnot, Vivien and Landrieu, Loic},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4872--4881},
  year      = {2021}
}
```

## Acknowledgements

This code builds on the U-TAE / PaPs implementation by V. Sainte Fare Garnot and L. Landrieu (https://github.com/VSainteuf/utae-paps). This work was funded by the European Union – NextGenerationEU under the project "Centro Nazionale di Ricerca per le Tecnologie dell'Agricoltura" – Agritech.

## License

<!-- TO COMPLETE: e.g. MIT. Check that it is compatible with the license of the original utae-paps repository. -->
