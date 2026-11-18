import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import argparse

# Aggiungi il path per importare i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset import PASTIS_Dataset
from src.utils import pad_collate

def count_stress_cases(dataset_folder, fold=1, input_steps=4, forecast_steps=2, ndvi_stress_threshold=0.4):
    """
    Conta i casi di stress nel dataset di test
    """
    print("=== CONTA CASI DI STRESS ===")
    
    # Carica dataset
    dt_test = PASTIS_Dataset(
        folder=dataset_folder,
        norm=True,
        reference_date="2018-09-01",
        input_steps=input_steps,
        forecast_steps=forecast_steps,
        ndvi_stress_threshold=ndvi_stress_threshold,
        target="forecasting",
        folds=[fold]
    )
    
    print(f"Dataset caricato: {len(dt_test)} campioni")
    
    test_loader = data.DataLoader(
        dt_test,
        batch_size=1,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=0,
    )
    
    # Contatori
    total_samples = 0
    stress_samples = 0
    all_ndvi_means = []
    
    print("Analizzando campioni...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            (x, dates), targets = batch
            stress_target = targets['stress_target']
            ndvi_target = targets['ndvi_target']
            
            # Conta stress
            stress_binary = (stress_target > 0.5).float()
            stress_samples += stress_binary.sum().item()
            total_samples += stress_target.numel()
            
            # Analisi NDVI
            ndvi_mean = ndvi_target.mean()
            all_ndvi_means.append(ndvi_mean.item())
            
            if batch_idx < 10:  # Primi 10 batch per debug
                print(f"Batch {batch_idx+1}:")
                print(f"  - NDVI mean: {ndvi_mean:.4f}")
                print(f"  - Stress target: {stress_target}")
                print(f"  - Stress binary: {stress_binary}")
    
    # Statistiche
    all_ndvi_means = np.array(all_ndvi_means)
    
    print(f"\n=== RISULTATI ===")
    print(f"Totale campioni: {total_samples}")
    print(f"Casi di stress: {stress_samples}")
    print(f"Percentuale stress: {stress_samples/total_samples*100:.2f}%")
    
    print(f"\n=== DISTRIBUZIONE NDVI MEDIE ===")
    print(f"Min: {all_ndvi_means.min():.4f}")
    print(f"Max: {all_ndvi_means.max():.4f}")
    print(f"Mean: {all_ndvi_means.mean():.4f}")
    print(f"Std: {all_ndvi_means.std():.4f}")
    
    # Test diverse soglie
    print(f"\n=== TEST DIVERSE SOGLIE ===")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for threshold in thresholds:
        stress_count = (all_ndvi_means < threshold).sum()
        stress_percent = stress_count / len(all_ndvi_means) * 100
        print(f"Soglia NDVI < {threshold:.1f}: {stress_count} casi ({stress_percent:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True, help="Path al dataset PASTIS")
    parser.add_argument("--fold", default=1, type=int, help="Fold da analizzare")
    parser.add_argument("--input_steps", default=4, type=int, help="Numero di step di input")
    parser.add_argument("--forecast_steps", default=2, type=int, help="Numero di step di forecast")
    parser.add_argument("--ndvi_stress_threshold", default=0.4, type=float, help="Soglia NDVI per stress")
    
    args = parser.parse_args()
    
    count_stress_cases(
        dataset_folder=args.dataset_folder,
        fold=args.fold,
        input_steps=args.input_steps,
        forecast_steps=args.forecast_steps,
        ndvi_stress_threshold=args.ndvi_stress_threshold
    )
