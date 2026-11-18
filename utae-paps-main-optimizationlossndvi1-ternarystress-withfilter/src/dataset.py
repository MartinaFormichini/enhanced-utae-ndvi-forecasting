import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        input_steps=4,
        forecast_steps=2,
        forecast_lag_days=7.0,
        ndvi_stress_threshold=0.4,
        training=False,
        apply_augmentation=False,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
            input_steps (int): Number of input steps for forecasting
            forecast_steps (int): Number of forecast steps
            ndvi_stress_threshold (float): Threshold for NDVI stress
            training (bool): If True, enables data augmentation during training
            apply_augmentation (bool): If True, applies augmentation transformations
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.forecast_lag_days = float(forecast_lag_days)
        self.ndvi_stress_threshold = ndvi_stress_threshold
        self.training = training
        self.apply_augmentation = apply_augmentation and training

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

        # Check rapido per forecasting
        if self.target == "forecasting":
            min_length = min([len(self.get_dates(pid, self.sats[0])) for pid in self.id_patches])
            if min_length < self.input_steps + self.forecast_steps:
                print(f"⚠️  ATTENZIONE: Minima lunghezza serie ({min_length}) < richiesta ({self.input_steps + self.forecast_steps})")
            else:
                print(f"✅ Serie temporali OK: min={min_length}, richiesto={self.input_steps + self.forecast_steps}")
            print(f"[INFO] Forecast lag calendario: {self.forecast_lag_days:g} giorni")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def augment_spatial(self, x, h_flip=None, v_flip=None, rot_k=None):
        """Augmentazione spaziale migliorata: SOLO flip orizzontale conservativo
        Per dati satellitari georeferenziati, evitiamo rotazioni e flip verticale.
        Manteniamo solo trasformazioni che preservano la coerenza geografica.
        """
        # x: (T, C, H, W) o (T, H, W)
        if not self.apply_augmentation:
            return x
        
        # Solo flip orizzontale (25% probabilità, molto conservativo)
        # Evitiamo flip verticale e rotazioni per dati geografici
        if h_flip is None:
            h_flip = torch.rand(1) > 0.75  # 25% probabilità (molto conservativo)
        if h_flip:
            x = torch.flip(x, dims=[-1])
        
        # Flip verticale e rotazioni DISABILITATE per dati georeferenziati
        # Non hanno senso fisico e possono distorcere la geografia
        
        return x
    
    def augment_spectral(self, x):
        """Augmentazione spettrale migliorata: variazione atmosferica realistica
        Invece di noise casuale, simuliamo variazioni atmosferiche realistiche
        che possono verificarsi in dati satellitari reali.
        """
        # x: (T, C, H, W)
        if not self.apply_augmentation:
            return x
        
        # Probabilità di applicare augmentation spettrale: 15% (molto conservativo)
        if torch.rand(1) > 0.85:
            return x  # La maggior parte delle volte non applicare nulla
        
        # 1. Variazione di brillantezza per canale (simula condizioni atmosferiche)
        # Solo 10% di variazione massima, molto conservativo
        if torch.rand(1) > 0.5:  # 50% probabilità di applicare
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.1  # ±5%
            x = x * brightness_factor
        
        # 2. Variazione leggera per banda (simula condizioni di illuminazione)
        # Applica variazioni diverse per bande ottiche vs termiche
        # Bande visibili (0-6) possono avere variazioni leggermente diverse
        if torch.rand(1) > 0.6:  # 40% probabilità
            for c in range(min(x.shape[1], 7)):  # Solo per prime 7 bande (ottiche)
                if torch.rand(1) > 0.7:  # 30% probabilità per ogni banda
                    band_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.05  # ±2.5%
                    x[:, c] = x[:, c] * band_factor
        
        # CLAMP finale per evitare valori fuori range
        x = torch.clamp(x, -3.0, 3.0)  # Range conservativo per dati normalizzati
        
        return x
    
    def augment_temporal(self, x):
        """Augmentazione temporale migliorata: temporal dropout conservativo
        Invece di shift che rompe l'allineamento, usiamo dropout temporale
        che maschera alcuni timestep per migliorare la robustezza.
        """
        # x: (T, ...)
        if not self.apply_augmentation:
            return x
        
        # Temporal dropout: maschera casualmente alcuni timestep (15% probabilità)
        if torch.rand(1) > 0.85:
            return x  # La maggior parte delle volte non applicare nulla
        
        # Maschera casualmente fino al 10% dei timestep
        T = x.shape[0]
        if T > 1:
            n_drop = max(1, int(T * 0.1))  # Massimo 10% dei timestep
            drop_indices = torch.randperm(T)[:n_drop]
            # Interpola i valori mascherati dai vicini invece di azzerare
            # Questo mantiene più informazione rispetto al semplice dropout
            for idx in drop_indices:
                if idx > 0 and idx < T - 1:
                    # Interpola dal timestep precedente e successivo
                    x[idx] = 0.5 * x[idx-1] + 0.5 * x[idx+1]
                elif idx == 0 and T > 1:
                    x[idx] = x[idx+1]  # Usa il successivo
                elif idx == T - 1 and T > 1:
                    x[idx] = x[idx-1]  # Usa il precedente
        
        return x
    
    def augment_ndvi_data(self, ndvi_data):
        """Augmentazione per NDVI target: DISABILITATA"""
        # IMPORTANTE: NON augmentiamo il target!
        # Aggiungere noise al target confonde l'apprendimento del modello
        # Il modello deve imparare a predire il target reale, non una versione rumorosa
        return ndvi_data  # Nessuna augmentation sul target
    
    def improved_ndvi_normalization(self, ndvi_data):
        """Normalizzazione NDVI migliorata (non cambia se non necessario)"""
        # Mantieni la normalizzazione già fatta, solo clamp finale
        return torch.clamp(ndvi_data, -1, 1)

    def _interp_ndvi_at_offset(self, ndvi_series, dates, query_offset):
        """
        Interpola linearmente NDVI alla data query_offset usando le due acquisizioni adiacenti.
        ndvi_series: (T, H, W), dates: (T,)
        """
        idx_right = int(np.searchsorted(dates, query_offset, side="left"))
        if idx_right <= 0:
            return ndvi_series[0]
        if idx_right >= len(dates):
            return ndvi_series[-1]
        idx_left = idx_right - 1
        d0 = float(dates[idx_left])
        d1 = float(dates[idx_right])
        if abs(d1 - d0) < 1e-8:
            return ndvi_series[idx_left]
        alpha = float((query_offset - d0) / (d1 - d0))
        return (1.0 - alpha) * ndvi_series[idx_left] + alpha * ndvi_series[idx_right]

    def _frame_at_nearest_date(self, x, dates, query_offset):
        """Frame Sentinel-2 (C,H,W) all'acquisizione più vicina a query_offset (giorni)."""
        idx = int(np.argmin(np.abs(dates - float(query_offset))))
        return x[idx]
    
    def __getitem__(self, item):
        id_patch = self.id_patches[item]
        # Carica la serie temporale Sentinel-2
        data = {
            satellite: np.load(
                os.path.join(
                    self.folder,
                    "DATA_{}".format(satellite),
                    "{}_{}.npy".format(satellite, id_patch),
                )
            ).astype(np.float32)
            for satellite in self.sats
        }
        data = {s: torch.from_numpy(a) for s, a in data.items()}
        x = data[self.sats[0]]  # (T, C, H, W)
        T = x.shape[0]
        # Calcola NDVI per ogni t
        # Calcola NDVI con normalizzazione automatica
        nir_band = x[:, 7]  # NIR band (banda 8)
        red_band = x[:, 2]  # Red band (banda 3)
        
        # Normalizza bande automaticamente se valori anomali
        if nir_band.max() > 10000:
            nir_band = nir_band / 10  # Riduci di 10x
        if red_band.max() > 10000:
            red_band = red_band / 10  # Riduci di 10x
        
        # Calcola NDVI con bande normalizzate
        ndvi_series = (nir_band - red_band) / (nir_band + red_band + 1e-6)  # (T, H, W)
        
        # Forza NDVI in range realistico se anomalo
        if torch.any(torch.abs(ndvi_series) > 100):
            ndvi_series = torch.clamp(ndvi_series, -1, 1)
        dates = self.get_dates(id_patch, self.sats[0]).astype(np.float32)
        if T < self.input_steps:
            raise ValueError(f"Patch {id_patch} troppo corta per input_steps={self.input_steps}")

        max_target_offset = float(dates[-1])
        valid_t0 = []
        for start in range(0, T - self.input_steps + 1):
            last_in_idx = start + self.input_steps - 1
            last_in_off = float(dates[last_in_idx])
            end_query = last_in_off + self.forecast_steps * self.forecast_lag_days
            if end_query <= max_target_offset + 1e-6:
                valid_t0.append(start)
        if not valid_t0:
            raise ValueError(
                f"Patch {id_patch} senza finestre valide per forecasting con lag={self.forecast_lag_days:g} giorni"
            )

        t0 = int(np.random.choice(valid_t0))
        x_in = x[t0:t0 + self.input_steps]  # (input_steps, C, H, W)
        last_input_idx = t0 + self.input_steps - 1
        last_input_offset = float(dates[last_input_idx])
        query_offsets = [
            last_input_offset + (step + 1) * self.forecast_lag_days
            for step in range(self.forecast_steps)
        ]
        ndvi_target = torch.stack(
            [self._interp_ndvi_at_offset(ndvi_series, dates, q) for q in query_offsets],
            dim=0,
        )  # (forecast_steps, H, W)

        # Frame S2 grezzi (non normalizzati) alla data più vicina a ciascun target NDVI:
        # usati dal cloud detector per una maschera per timestep di forecast.
        s2_target_steps = torch.stack(
            [self._frame_at_nearest_date(x, dates, q) for q in query_offsets],
            dim=0,
        )  # (forecast_steps, C, H, W)

        # Applica normalizzazione canale-wise agli input del modello.
        # NOTE: il target NDVI resta in scala fisica e non va normalizzato qui.
        if self.norm is not None:
            sat = self.sats[0]
            mean, std = self.norm[sat]
            x_in = (x_in - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-6)
        
        # APPLICA AUGMENTATION MIGLIORATA AGLI INPUT se in training mode
        # Augmentation conservativa e realistica per dati satellitari georeferenziati
        if self.apply_augmentation:
            # Genera parametri di trasformazione spaziale una volta sola
            # per applicare le stesse trasformazioni a input e target
            h_flip = torch.rand(1) > 0.75  # 25% probabilità (molto conservativo)
            
            # 1. Augmentazione spaziale: SOLO flip orizzontale (stessa per input e target)
            # Mantiene coerenza geometrica tra input e target NDVI
            x_in = self.augment_spatial(x_in, h_flip=h_flip)
            ndvi_target = self.augment_spatial(ndvi_target, h_flip=h_flip)
            s2_target_steps = self.augment_spatial(s2_target_steps, h_flip=h_flip)
            
            # 2. Augmentazione spettrale: variazioni atmosferiche realistiche
            # Solo per input, NON per target (il target NDVI è già calcolato)
            x_in = self.augment_spectral(x_in)
            
            # 3. Augmentazione temporale: temporal dropout conservativo
            # Solo per input, mantiene allineamento temporale con target
            x_in = self.augment_temporal(x_in)
            
            # NOTA: NON augmentiamo il target NDVI direttamente
            # Il target viene augmentato solo spazialmente per mantenere coerenza
        
        # IMPORTANT:
        # Il target NDVI e gia in scala fisica [-1, 1]. Evitiamo qualsiasi
        # normalizzazione per-patch, che altera la distribuzione tra campioni
        # e rende incoerente la soglia stress (ndvi_stress_threshold).
        ndvi_target = torch.clamp(ndvi_target, -1.0, 1.0)
        
        # Usa reshape invece di view per gestire tensori non contigui dopo augmentation
        ndvi_mean_target = ndvi_target.reshape(self.forecast_steps, -1).mean(dim=1, keepdim=True)  # (forecast_steps, 1)

        # Stress ternario da NDVI medio:
        # 0 = healthy  (NDVI >= 0.4)
        # 1 = warning  (0.1 <= NDVI < 0.4)
        # 2 = critical (NDVI < 0.1)
        stress_target = torch.zeros_like(ndvi_mean_target, dtype=torch.long)
        stress_target[(ndvi_mean_target >= 0.1) & (ndvi_mean_target < 0.4)] = 1
        stress_target[ndvi_mean_target < 0.1] = 2
        
        # Date input e target (target su lag di calendario, non "indice successivo")
        dates_in = torch.from_numpy(dates[t0:t0 + self.input_steps])
        dates_target = torch.tensor(query_offsets, dtype=dates_in.dtype)
        
        return (x_in, dates_in), {
            "ndvi_target": ndvi_target,
            "ndvi_mean_target": ndvi_mean_target,
            "stress_target": stress_target,
            "dates_target": dates_target,
            "s2_target_steps": s2_target_steps,
        }

    def check_temporal_length(self):
        """Verifica la lunghezza delle serie temporali per forecasting"""
        min_lengths = []
        valid_patches = []
        
        print("=== VERIFICA LUNGHEZZA SERIE TEMPORALI ===")
        for patch_id in self.id_patches:
            dates = self.get_dates(patch_id, self.sats[0])
            min_lengths.append(len(dates))
            
            if len(dates) >= self.input_steps + self.forecast_steps:
                valid_patches.append(patch_id)
        
        min_length = min(min_lengths)
        avg_length = np.mean(min_lengths)
        required_length = self.input_steps + self.forecast_steps
        
        print(f"Lunghezza minima serie temporale: {min_length}")
        print(f"Lunghezza media serie temporale: {avg_length:.1f}")
        print(f"Lunghezza richiesta per forecasting: {required_length}")
        print(f"Patch totali: {len(self.id_patches)}")
        print(f"Patch validi per forecasting: {len(valid_patches)}")
        print(f"Patch troppo corte: {len(self.id_patches) - len(valid_patches)}")
        
        if min_length < required_length:
            print(f"⚠️  ATTENZIONE: Alcuni patch sono troppo corti per forecasting!")
            print(f"   Patch con serie < {required_length}: {len(self.id_patches) - len(valid_patches)}")
        
        return valid_patches


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = PASTIS_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0][sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))
