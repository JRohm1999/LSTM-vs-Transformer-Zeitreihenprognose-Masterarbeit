#Traningsskript für den PatchTST

# Import der nötigen Biliotheken
# Identisch zu den anderen Modellen wird Pytorch verwendet
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setzt die Präzision für float32-Matrixberechnung auf medium um Performance zu steigern
# Hintergrund ist einfach das Modelltraning zu beschleunigen
torch.set_float32_matmul_precision("medium")

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

import lightning.pytorch as pl
# Callbacks steuern das Verhalten während des Trainings hier: Early Stopping, LR-Überwachung, Checkpoints
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Import von Funktionen aus dem Logging Skript
from run_logger import get_system_info, save_excel, save_json, save_plots


# -----------------------------------------------------------------------------
# Pfade
# -----------------------------------------------------------------------------
PREP_DIR = Path("data") / "preprocessed"
CSV_PATH = PREP_DIR / "m5_long.csv"
RUNS_DIR = Path("runs") / "patchtst"


# -----------------------------------------------------------------------------
# Trainingdetails
# -----------------------------------------------------------------------------
# Traning der Modelle auf der Grafikkarte (GPU) wenn möglich. Wenn keine GPU vorhanden wird CPU genutzt
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seeds: Anzahl an Traningsläufen mit der identischen Hyperparameterkonfiguration
BASE_SEED = 1 # Seed selbst
NUM_SEEDS = 3 # Anzahl an Seeds für normalen Lauf
MAX_SERIES = 1000

# Encoder = 56 Tage Historie, PRED_LEN = 28 Prognosezeitraum
ENCODER_LEN = 56
PRED_LEN = 28
HORIZON = PRED_LEN

# Anzahl der maximalen Traingsepochen pro Run. Wird ggf. durch Early-Stopping vorher beendet
MAX_EPOCHS = 50

# -----------------------------------------------------------------------------
# Hyperparameter (finale Traningsparameter nach Optunaläufen)
# Fixe Hyperparameter, gesteuerte Parameter durch Optuna in suggest_hyperparameters Funktion
# -----------------------------------------------------------------------------
BATCH_SIZE = 1024
LR = 0.0006559644071632867
D_MODEL = 128   # Synonym für Hidden Size
ATTN_HEAD_SIZE = 2
HIDDEN_CONT_SIZE = int(D_MODEL/2) # Hidden Continuous Size wird als halbe D_MODEL-Größe gesetzt. Entscheidung dafür ist die Modellgröße bewusst im Rahmen zu halten.  
DROPOUT = 0.10827211824776498

# Patch Len und Stride sind spezifische PatchTST Parameter, die für das Patching der Historie genutzt werden
PATCH_LEN = 16
PATCH_STRIDE = 8
NUM_TRANSFORMER_LAYERS = 3
SERIES_EMB_DIM = 16 # Anzahl der Embeding Features

#Learningrate spezifische Parameter
LR_PATIENCE = 3     # nach x Läufen ohne Verbesserung wird LR reduziert
LR_Factor = 0.5     # Faktor zur Verringerung der LR bei erreichen der LR_Patience. z.B. LR 0.3 wird zu LR 0.15 usw. Bis minimal LR_Min erreicht ist.
PATIENCE = 10       # allgemeine Patience für Abbruch des Tranings nach x Läufen ohne Verbesserung
MIN_DELTA = 0.001   # Mindestverbesserung des Losses pro Epoche für eine Verbesserung
LR_MIN = 1e-6       # Untergrenze der Learningrate


# -----------------------------------------------------------------------------
# Optuna-Konfiguration
# -----------------------------------------------------------------------------
USE_OPTUNA = False # Bei False wird normal traniert, bei True wird Optunasuchlauf durchgefürt
OPTUNA_TRIALS = None   # Anzahl an Trials, hier über Zeit gesteuert daher None
OPTUNA_TIMEOUT_SEC = 43200  # Anzahl der Sekunden für den Optunalauf, hier 12 Stunden
OPTUNA_SEEDS_PER_TRIAL = 1  # Seeds per Trial, hier 1 um Computeaufwand im Rahmen der Arbeit zu halten
OPTUNA_DIRECTION = "minimize" # Aufgabe von Optuna: Loss reduzieren


# -----------------------------------------------------------------------------
# Feature-Konfiguration
# -----------------------------------------------------------------------------
LAG_LIST = [1, 7, 14, 28] # LagListe für Features
ROLLING_WINDOWS = [7, 28] # Rollingliste für Features

KNOWN_REAL_FEATURES = [
    "price_s",
    "price_missing",
    "snap",
    "wday_s",
    "month_s",
    "year_s",
    "has_event_1",
    "has_event_2",
] # KNOWN_REAL_FEATURES = Dem Modell bekannte Features aus der Vergangenheit oder bekannte Informationen aus der Zukunft. 

STATIC_CATEGORICALS = ["item_id", "store_id", "state_id"]

# Funktion um Seed zu erzeugen 
def set_seed(seed: int) -> None:
    # Setzt Seed und stellt somit Reproduzierbarkeit der Trainingsläufe sicher
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=False)


# Funktion um die Preprocessed Daten aus der CSV (Subset) zu laden.
def load_preprocessed() -> pd.DataFrame:
    # Prüft ob die Datei existiert und wirft einen Fehler wenn nicht
    if not CSV_PATH.exists():
        raise FileNotFoundError("Keine Datei gefunden (m5_long.csv).")
    # Liest die CSV-Datei ein und wandelt die date Spalte direkt in ein Datumsformat um
    return pd.read_csv(CSV_PATH, parse_dates=["date"])


# Hinzufügen der zeitlichen Features Lag und Rolling
def add_time_series_features(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

    # Erstellen der Lag Features
    for lag in LAG_LIST:
        df[f"y_log_lag_{lag}"] = df.groupby("series_id")["y_log"].shift(lag)

    grouped_shifted = df.groupby("series_id")["y_log"].shift(1)

    # Erstellen der Rolling Features 
    for window in ROLLING_WINDOWS:
        df[f"y_log_roll_mean_{window}"] = (
            grouped_shifted.rolling(window=window, min_periods=1)
            .mean() # Durchschnitt
            .reset_index(level=0, drop=True)
        )
        df[f"y_log_roll_std_{window}"] = (
            grouped_shifted.rolling(window=window, min_periods=1)
            .std() #Standardabweichung
            .reset_index(level=0, drop=True)
        )

    # Alle Features in einer Liste sammeln
    engineered_cols = [f"y_log_lag_{lag}" for lag in LAG_LIST]
    engineered_cols += [f"y_log_roll_mean_{window}" for window in ROLLING_WINDOWS]
    engineered_cols += [f"y_log_roll_std_{window}" for window in ROLLING_WINDOWS]
    df[engineered_cols] = df[engineered_cols].fillna(0.0)

    return df


# Feature Spalten erstellen, Ausgabe der Known Features und der Unknown Features
def build_feature_columns():
    unknown_reals = ["y_log"]
    unknown_reals += [f"y_log_lag_{lag}" for lag in LAG_LIST]
    unknown_reals += [f"y_log_roll_mean_{window}" for window in ROLLING_WINDOWS]
    unknown_reals += [f"y_log_roll_std_{window}" for window in ROLLING_WINDOWS]
    # Gibt beide Listen zurück
    return KNOWN_REAL_FEATURES.copy(), unknown_reals


# MASE Nenner berechnen: MASE besteht aus MAE Prognosemodell /MAE (Naivprognose) 
def compute_mase_denominators(train_df: pd.DataFrame, seasonality: int = 7) -> dict:
    denominators = {}
    for series_id, group in train_df.groupby("series_id"):
        group = group.sort_values("time_idx")
        if "y" in group.columns:
            y_values = group["y"].to_numpy(dtype=np.float32)
        else:
            y_values = np.expm1(group["y_log"].to_numpy(dtype=np.float32))

         # Für sehr kurze Serien wird der Denominator auf 1 gesetzt, um Division durch Null zu vermeiden. Sollte in diesem Datensatz eigentlich nicht vorkommen.
         # Dient nur als Sicherheitsnetz
        if len(y_values) <= seasonality:
            denominators[str(series_id)] = 1.0
            continue
        
        # Abweichung von heute zu vor sieben Tagen
        diffs = np.abs(y_values[seasonality:] - y_values[:-seasonality])
       
        # Mittelwert der absoluten Differenzen ergibt den MAE der Naivprognose
        # Falls Mittelwert null sein sollte wird er auf 1 gesetzt, sonst später Division durch null
        denom = float(np.mean(diffs)) if np.mean(diffs) > 0 else 1.0
        denominators[str(series_id)] = denom
    return denominators

# Holt Zuordnung von numerischem Index zu ursprünglicher Zeitreihen-ID aus dem Dataset
def get_series_id_mapping(training_ds: TimeSeriesDataSet):
    mapping = None
    # Greift auf den internen kategorischen Encoder des TimeSeriesDataSet zu
    if hasattr(training_ds, "categorical_encoders"):
        encoders = getattr(training_ds, "categorical_encoders", {})
        if isinstance(encoders, dict) and "series_id" in encoders:
            encoder = encoders["series_id"]
            classes = None
            # Unterstützt verschiedene Attributnamen des sklearn-Encoders (classes_ oder classes)
            if hasattr(encoder, "classes_"):
                classes = list(getattr(encoder, "classes_"))
            elif hasattr(encoder, "classes"):
                classes = list(getattr(encoder, "classes"))
            if classes is not None:
                # Erstellt ein Dictionary - numerischer Index zu oiginale Zeitreihen-ID als String
                mapping = {int(i): str(v) for i, v in enumerate(classes)}
    return mapping

# Extrahiert die Zeitreihen-IDs aus dem rohen Batch-Dictionary des DataLoaders
def extract_series_ids_from_raw_x(raw_x, series_mapping) -> np.ndarray:
    # Sucht nach dem Gruppenkey im Batch-Dictionary (pytorch_forecasting nutzt 'groups' oder 'group_ids')
    groups_key = None
    if isinstance(raw_x, dict):
        # beides probieren 'groups' oder 'group_ids'
        if "groups" in raw_x:
            groups_key = "groups"
        elif "group_ids" in raw_x:
            groups_key = "group_ids"

    # Falls nichts gefunden wird, wird Fehler ausgegeben
    if groups_key is None:
        raise KeyError("Keine Gruppeninformation (groups/group_ids) in raw.x gefunden.")

    group_values = raw_x[groups_key]
    # Konvertiert Tensor zu NumPy falls nötig
    if isinstance(group_values, torch.Tensor):
        group_values = group_values.detach().cpu().numpy()

    group_values = np.asarray(group_values)
    # Falls das Array mehrdimensional ist (Batch × Gruppen) nur die erste Gruppe (series_id) verwenden
    if group_values.ndim == 2:
        group_values = group_values[:, 0]

    # Mappt die numerischen Indizes auf die ursprünglichen Zeitreihen-ID-Strings für Metrikberechnungen
    series_ids = []
    for value in group_values.tolist():
        try:
            value_int = int(value)
            if series_mapping is not None and value_int in series_mapping:
                series_ids.append(series_mapping[value_int])
            else:
                series_ids.append(str(value_int))
        except Exception:
            series_ids.append(str(value))

    return np.asarray(series_ids, dtype=object)


# Funktion um aus dem Probabilistic-Forecast einen Punkt-Forecast zu erstellen
def extract_point_forecast(prediction_array: np.ndarray) -> np.ndarray:
    predictions = np.asarray(prediction_array)
    if predictions.ndim == 3 and predictions.shape[2] > 1:
        return predictions[:, :, predictions.shape[2] // 2]
    if predictions.ndim == 3:
        return predictions[:, :, 0]
    return predictions


# Berechne den Loss für MSE im Log Space (Traningsmetrik)
def calc_mse_loss_logspace(pred_y_log: np.ndarray, true_y_log: np.ndarray):
    return float(np.mean((pred_y_log - true_y_log) ** 2))

# Berechne die Bewertungsmetriken MASE, MSE, und WAPE gesamt als auch auf wöchentlicher Basis 
# wöchentliche Berechnung ist für die Evaluation nach dem Traning nicht unbedingt nötig, ist aber vollständigkeitshalber hier erhalten geblieben
def eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, series_ids, mase_denoms):
    
    # Die vier Wochen mit den Start und Endtagen
    week_slices = [(0, 7), (7, 14), (14, 21), (21, 28)]

    pred_y = np.asarray(pred_y, dtype=np.float32)
    true_y = np.asarray(true_y, dtype=np.float32)

    # Negative Werte abschneiden: Absatzzahlen können nicht negativ sein
    pred_y = np.clip(pred_y, a_min=0.0, a_max=None)
    true_y = np.clip(true_y, a_min=0.0, a_max=None)

    # Absoluter Fehler für alle Zeitpunkte und Zeitreihen gleichzeitig berechnen
    abs_err = np.abs(pred_y - true_y)

    # MSE
    mse = float(np.mean((pred_y - true_y) ** 2))

    # Nenner für MASE Berechnung, also Funktion mase_denoms aufrufen.
    den = np.array([float(mase_denoms.get(str(series_id), 1.0)) for series_id in series_ids], dtype=np.float32)
    den = np.where(den > 0, den, 1.0)

    # MAE des Modellforecasts
    mae_overall = np.mean(abs_err, axis=1)

    # MASE Berechnung: MAE (Modell) / MAE (Nativprognose)
    mase = float(np.mean(mae_overall / den))

    # MASE Wochenberechnung
    mase_weeks = []
    for start, end in week_slices:
        mae_week = np.mean(abs_err[:, start:end], axis=1)
        mase_weeks.append(float(np.mean(mae_week / den)))

    
    # WAPE Berechnung
    wape_num = float(np.sum(abs_err))
    wape_den = float(np.sum(true_y))
    wape = (wape_num / wape_den) if wape_den > 0 else float("nan")

    # Wape pro Woche
    wape_weeks = []
    for start, end in week_slices:
        num = float(np.sum(abs_err[:, start:end]))
        den_w = float(np.sum(true_y[:, start:end]))
        wape_weeks.append((num / den_w) if den_w > 0 else float("nan"))

    # Rückgabe aller Lossmetriken
    return {
        "mase": mase,
        "mase_w1": mase_weeks[0],
        "mase_w2": mase_weeks[1],
        "mase_w3": mase_weeks[2],
        "mase_w4": mase_weeks[3],
        "mse": mse,
        "wape": float(wape),
        "wape_w1": float(wape_weeks[0]),
        "wape_w2": float(wape_weeks[1]),
        "wape_w3": float(wape_weeks[2]),
        "wape_w4": float(wape_weeks[3]),
    }


# Quantile Loss Funktion: berechnet den Loss für alle Quantile gleichzeitig
# Transformermodelle geben Prognosen in Quantilen aus
def quantile_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    losses = []
    for quantile_index, quantile in enumerate(quantiles):
        # Fehler = Zielwert - Prognose für das jeweilige Quantil
        errors = target - prediction[:, :, quantile_index]
        losses.append(torch.maximum((quantile - 1.0) * errors, quantile * errors).unsqueeze(-1))
    # Alle Quantil-Losses stapeln und mittleren Loss berechnen
    stacked_losses = torch.cat(losses, dim=-1)
    return stacked_losses.mean()

# Hilfsfunktion um einen kompletten Batch auf das "Traningsgerät", hier GPU, zu verschieben
def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


# Python Klasse für den PatchTST Encoderblock
class PatchTSTEncoderBlock(nn.Module):
    def __init__(self, d_model: int, attention_head_size: int, dropout: float, ff_d_model: int):
        super().__init__()

        # Multi-Head Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_head_size,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # LayerNorm nach Attention-Layer
        self.norm_1 = nn.LayerNorm(d_model)
        # Feed-Forward Netzwerk: Projektion auf ff_d_model → GELU → zurück auf d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_d_model, d_model),
        )
        # LayerNorm nach FFN (zweite Residual-Verbindung)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, hidden_states: torch.Tensor):
        # Self-Attention mit Residual-Verbindung
        attention_output, attention_weights = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            need_weights=True, # Gibt Attention-Gewichte für Analyse zurück
            average_attn_weights=False,
        )
        hidden_states = self.norm_1(hidden_states + self.dropout(attention_output))
        feed_forward_output = self.ffn(hidden_states)
        hidden_states = self.norm_2(hidden_states + self.dropout(feed_forward_output))
        return hidden_states, attention_weights


# Python Klasse für das PatchTST Modell (Basis ist PyTorch Lightning Framework)
# PatchTST teilt die Eingangssequenz in überlappende Patches auf und verarbeitet diese in den Transformerblöcken
class PatchTSTModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        num_series: int,  # Gesamtanzahl der Zeitreihen (für Embedding-Größe)
        learning_rate: float,
        d_model: int,     # Hidden Size
        attention_head_size: int, # Anzahl der Attention-Heads
        hidden_continuous_size: int, # FFN-Zwischengröße
        dropout: float,
        patch_len: int,
        patch_stride: int,
        num_transformer_layers: int,
        series_emb_dim: int,
        mase_denoms: dict,  # Vorberechnete MAE der Naivprognose je Zeitreihe für MASE-Berechnung
        series_mapping,     # Zuordnung numerischer Index → originale Zeitreihen-ID
        feature_names: list[str], # Namen der Input-Features (nur für Logging und Analyse)
        quantiles: list[float] | None = None, # Zu prognostizierende Quantile, hier keine da Punktforecasrt
    ):
        super().__init__()

        # Festlegen der verschieden Parameter des Modells
        self.save_hyperparameters(ignore=["mase_denoms", "series_mapping", "feature_names"]) # save_hyperparameters speichert alle Parameter für Checkpoint-Wiederherstellung
        self.input_dim = int(input_dim)
        self.horizon = int(horizon)
        self.num_series = int(num_series)
        self.learning_rate = float(learning_rate)
        self.d_model = int(d_model)
        self.attention_head_size = int(attention_head_size)
        self.hidden_continuous_size = int(hidden_continuous_size)
        self.dropout_rate = float(dropout)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.num_transformer_layers = int(num_transformer_layers)
        self.series_emb_dim = int(series_emb_dim)
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)

        self.mase_denoms = mase_denoms or {}
        self.series_mapping = series_mapping
        self.feature_names = list(feature_names)

        # Zeitreihen-Embedding - lernt eine spezifische Repräsentation für jede Zeitreihe
        self.series_embedding = nn.Embedding(self.num_series + 1, self.series_emb_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.patch_projection = nn.Linear(self.patch_len * (self.input_dim + self.series_emb_dim), self.d_model)
        self.positional_embedding = nn.Parameter(torch.randn(1, 256, self.d_model) * 0.02)
        self.encoder_blocks = nn.ModuleList(
            [
                PatchTSTEncoderBlock(
                    d_model=self.d_model,
                    attention_head_size=self.attention_head_size,
                    dropout=self.dropout_rate,
                    ff_d_model=max(self.hidden_continuous_size, self.d_model * 2),
                )
                for _ in range(self.num_transformer_layers)
            ]
        )
        # LayerNorm nach dem letzten Encoder-Block
        self.encoder_norm = nn.LayerNorm(self.d_model)
        # Output-Head: transformiert den gepoolten Hidden State in Quantilprognosen
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.horizon * self.num_quantiles),
        )

        # Zwischenspeicher für Validierungs-Batches . Diese werden am Epochenende zusammengefasst
        self._val_pred_batches = []
        self._val_true_batches = []
        self._val_series_batches = []

        self._last_attention_weights = None 

    # Erstellen der Patches - Besonderheit des PatchTST Modells.
    def make_patches(self, encoder_cont: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
        # Encoder und Embidding Features verarbeiten
        encoder_cont = self.input_norm(encoder_cont)
        series_emb = self.series_embedding(series_ids).unsqueeze(1).expand(-1, encoder_cont.size(1), -1)
        # Econder und Embedding Features aneinanderhängen
        encoder_inputs = torch.cat([encoder_cont, series_emb], dim=-1)

        if encoder_inputs.size(1) < self.patch_len:
            pad_len = self.patch_len - encoder_inputs.size(1)
            encoder_inputs = F.pad(encoder_inputs, (0, 0, pad_len, 0))
        
        # unfold() erstellt überlappende Fenster (Patches) entlang der Zeitdimension
        patches = encoder_inputs.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.contiguous().permute(0, 1, 3, 2).reshape(encoder_inputs.size(0), -1, self.patch_len * encoder_inputs.size(2))
        return patches

    # Funktion des Forwardpasses (Erstellung der Prognose)
    def forward(self, x: dict) -> dict:
        # Encoder-Features aus dem Batch-Dictionary extrahieren
        encoder_cont = x["encoder_cont"].float()

        # Zeitreihen-IDs aus den Gruppen extrahieren
        group_values = x["groups"]
        # Falls die gelieferte Form zweidimensional ist (Batch und Gruppe), dann nehme nu die erste Spalte mit Gruppeninfo 
        if group_values.ndim == 2:
            group_values = group_values[:, 0]
        series_ids = group_values.long()

        # Patches aus der Eingangssequenz erzeugen. Das ist der Kern-Mechanismus des PatchTST
        patches = self.make_patches(encoder_cont, series_ids)
        hidden_states = self.patch_projection(patches)

        # Embedding dem Hidden State hinzufügen
        hidden_states = hidden_states + self.positional_embedding[:, : hidden_states.size(1), :]

        # Sequentiell durch alle gestapelten Encoder-Blöcke laufen und hidden_states sammeln
        attention_weights = None
        for encoder_block in self.encoder_blocks:
            hidden_states, attention_weights = encoder_block(hidden_states)

        # Finale Normalisierung nach allen Encoder-Blöcken
        hidden_states = self.encoder_norm(hidden_states)
        pooled_hidden = hidden_states.mean(dim=1)
  
        # Vorhersage im letzten Attention Kopf erzeugen
        prediction = self.output_head(pooled_hidden)
        prediction = prediction.view(-1, self.horizon, self.num_quantiles)

        if attention_weights is not None:
            self._last_attention_weights = attention_weights.detach()

        return {
            "prediction": prediction,
            "attention": attention_weights,
        }

    # Funktion zum Start der Epoche - leert alle Validerungslisten für Vorhersage, IST und Serien für die Epoche
    def on_validation_epoch_start(self):
        self._val_pred_batches = []
        self._val_true_batches = []
        self._val_series_batches = []

    # Funktion eines Traningsschrittes
    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y_true_log = y[0].float()
        else:
            y_true_log = y.float()

        network_out = self(x)
        # Quantile Loss als Trainingsziel Log-Space, da y_log als Ziel
        loss = quantile_loss(network_out["prediction"], y_true_log, self.quantiles)

        # Logging des Trainingsverlustes pro Epoche für CSV und pro Schritt für Fortschrittsanzeige
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    # Valsierungsfunktion
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y_true_log = y[0].float()
        else:
            y_true_log = y.float()

        network_out = self(x)
        # Vorhersage ermitteln
        prediction = network_out["prediction"]

        # Validierungsloss im Log-Space wie bei training_step)
        val_loss = quantile_loss(prediction, y_true_log, self.quantiles)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Sanity Check: kurzer Durchlauf vor dem ersten echten Training. Ist ein Feature von Pytorch und dient nur dazu die gesamte Datenpipline zu testen bevor richtig traniert wird.
        if self.trainer is not None and self.trainer.sanity_checking:
            return val_loss

        # Punktprognose aus Quantil-Ausgaben extrahieren
        pred_log = extract_point_forecast(prediction.detach().float().cpu().numpy())
        true_log = y_true_log.detach().float().cpu().numpy()

        # Wieder in den Originalraum transformieren
        pred_y = np.expm1(pred_log).clip(min=0.0)
        true_y = np.expm1(true_log).clip(min=0.0)


        series_ids = extract_series_ids_from_raw_x(x, self.series_mapping)

        # Batch-Ergebnisse in den  Epochenlisten speichern
        self._val_pred_batches.append(pred_y)
        self._val_true_batches.append(true_y)
        self._val_series_batches.append(series_ids)
        return val_loss

    # Funktion für das Ende der Epoche - alle Batch-Ergebnisse aggregieren und finale Metriken berechnen
    def on_validation_epoch_end(self):
        # Beim Sanity Check oder leeren Epochenlisten keine Metrikberechnung durchführen
        if self.trainer is not None and self.trainer.sanity_checking:
            return

        if not self._val_pred_batches:
            return

        # Alle Batches zu einem großen Array zusammenführen
        pred_y = np.concatenate(self._val_pred_batches, axis=0)
        true_y = np.concatenate(self._val_true_batches, axis=0)
        series_ids = np.concatenate(self._val_series_batches, axis=0)

        # Berechnung aller Evaluationsmetriken 
        metrics = eval_mase_mse_wape_weekly_from_arrays(
            pred_y=pred_y,
            true_y=true_y,
            series_ids=series_ids,
            mase_denoms=self.mase_denoms,
        )

        # Logging aller Metriken - val_mase als Steuerungsmetrik für Early Stopping und LR-Scheduler
        self.log("val_mase", float(metrics["mase"]), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mase_w1", float(metrics["mase_w1"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_mase_w2", float(metrics["mase_w2"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_mase_w3", float(metrics["mase_w3"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_mase_w4", float(metrics["mase_w4"]), on_epoch=True, prog_bar=False, logger=True)

        self.log("val_wape", float(metrics["wape"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_wape_w1", float(metrics["wape_w1"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_wape_w2", float(metrics["wape_w2"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_wape_w3", float(metrics["wape_w3"]), on_epoch=True, prog_bar=False, logger=True)
        self.log("val_wape_w4", float(metrics["wape_w4"]), on_epoch=True, prog_bar=False, logger=True)

        self.log("val_mse", float(metrics["mse"]), on_epoch=True, prog_bar=False, logger=True)

        # Zwischenausgabe wären des Tranings
        # Wurde vorallem am Anfang fürs Testen genutzt
        print("val_mase:", self.trainer.callback_metrics.get("val_mase"))
        print("lr:", self.trainer.optimizers[0].param_groups[0]["lr"])

    def configure_optimizers(self):
        # Adam-Optimizer: adaptiver Gradientenabstieg. Grundsätzlich gut für Transformer-Modelle geeignet
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # ReduceLROnPlateau: reduziert die Lernrate wenn keine Verbesserung von val_mase erkannt wird
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_Factor,
            patience=LR_PATIENCE,
            threshold=MIN_DELTA,
            threshold_mode="abs",
            min_lr=LR_MIN,
        )

        # Konfiguration des LR-Schedulers. Überwacht val_mase einmal pro Epoche
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mase",
                "interval": "epoch",
                "frequency": 1,
            },
        }

# Erstellt die TimeSeriesDataSet-Objekte für Training und Validierung
def build_timeseries_datasets(df: pd.DataFrame):
    df = df.copy()
    # Zur Sicherheit allle Datentypen nochmals auf String festlegen
    df["series_id"] = df["series_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    df["state_id"] = df["state_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    # Nimm die angebene Anzahl an Serien aus dem Subset. In diesem Falle identisch, beides 1000 Serien
    if MAX_SERIES is not None:
        allowed_series = sorted(df["series_id"].unique())[:MAX_SERIES] # die ersten 1000 Serien
        df = df[df["series_id"].isin(allowed_series)].copy()  # Kopieren in DF 

    # Featurespalten erstellen
    known_reals, unknown_reals = build_feature_columns()

    # Split-Grenzen(leter Tag eines Splits) bestimmen
    train_cutoff = int(df.loc[df["split"] == "train", "time_idx"].max())
    val_cutoff = int(df.loc[df["split"] == "val", "time_idx"].max())
    #test_cutoff = int(df.loc[df["split"] == "test", "time_idx"].max())  # Wurde für Test ürsprunglich mitentwickelt, ist aber für das Traning nicht nötig

    # Trainingsdatensatz
    training = TimeSeriesDataSet(
        df[df.time_idx <= train_cutoff], # bis zum letzten Traningstag
        time_idx="time_idx",
        target="y_log",
        group_ids=["series_id"],
        static_categoricals=STATIC_CATEGORICALS,
        min_encoder_length=ENCODER_LEN,
        max_encoder_length=ENCODER_LEN,
        min_prediction_length=PRED_LEN,
        max_prediction_length=PRED_LEN,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=False,  # diese drei speziellen Features für den PatchTST werden nicht aktiviert um eine bessere Vergleichbarkeit der Modelle zu gewährleisten.
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    # Validerungsdatensatz - identisch wie Traning nur für Valdierungsdaten
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df.time_idx <= val_cutoff],
        predict=True,
        stop_randomization=True,
        min_prediction_idx=train_cutoff + 1,  # Prognosen starten am ersten val Tag
    )

    return training, validation


# Funktion mit der der Optunasuchlauf gesteuert wird. Enthält den gesamten Optuna Suchraum, auch vorgestellt in der Arbeit.
# Hier wird der Suchraum gepflegt
def suggest_hyperparameters(optuna_trial: optuna.Trial) -> dict:
    return {
        "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "d_model": optuna_trial.suggest_categorical("d_model", [128,256]),
        "attention_head_size": optuna_trial.suggest_categorical("attention_head_size", [2, 4]),
        "dropout": optuna_trial.suggest_float("dropout", 0.0, 0.3),
        "num_transformer_layers": optuna_trial.suggest_categorical("num_transformer_layers", [2, 3]), # wie viele Transformer-Blöcke gestapelt werden. 
    }

# Funktion um die Metrikausgabe für die CSV zu sortieren
def reorder_metrics_columns(metrics_dataframe: pd.DataFrame):
    column_order = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mase",
        "val_mase_w1",
        "val_mase_w2",
        "val_mase_w3",
        "val_mase_w4",
        "val_wape",
        "val_wape_w1",
        "val_wape_w2",
        "val_wape_w3",
        "val_wape_w4",
        "val_mse",
        "lr",
        "epoch_time_sec",
    ]
    existing_columns = [col for col in column_order if col in metrics_dataframe.columns]
    remaining_columns = [col for col in metrics_dataframe.columns if col not in existing_columns]
    # Alle Spalten ausgeben
    return metrics_dataframe[existing_columns + remaining_columns]

# Sammelt Prognosen und wahre Werte über alle Batches eines DataLoaders für die finale Evaluation
def collect_predictions(model: PatchTSTModel, dataloader, series_mapping, device):
    model = model.to(device)
    model.eval()

    prediction_logs = []
    true_logs = []
    series_ids_all = []

    # torch.no_grad() deaktiviert die Gradientenberechnung. Geschwindigkeitsvorteil
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = move_batch_to_device(x, device)
            if isinstance(y, (tuple, list)):
                y_true_log = y[0].to(device).float()
            else:
                y_true_log = y.to(device).float()


            # Erstellung der Prgnosen model() ruft forward_pass auf
            network_out = model(x)
            prediction = network_out["prediction"].detach().float().cpu().numpy()
            true_log = y_true_log.detach().float().cpu().numpy()

            # ersteelle Punktforecast
            prediction_logs.append(extract_point_forecast(prediction))
            true_logs.append(true_log)
            series_ids_all.append(extract_series_ids_from_raw_x(x, series_mapping))
    
    # Alle Batches zu Arrays zusammenführen und zurückgeben
    return {
        "prediction_log": np.concatenate(prediction_logs, axis=0),
        "true_log": np.concatenate(true_logs, axis=0),
        "series_ids": np.concatenate(series_ids_all, axis=0),
    }


# Funktion die einen kompletten Traningsdurchlauf für einen Seed enthält. 
def train_one_seed(
    df: pd.DataFrame,
    mase_denoms: dict,
    run_dir: Path,
    seed_value: int,
    hpo_params: dict | None = None,
) -> dict:
    # Hyperparameter kommen entweder aus der Konfigration oder werden durch Optuna geliefert (siehe nächste If-Schleife)
    learning_rate = LR
    d_model = D_MODEL
    attention_head_size = ATTN_HEAD_SIZE
    hidden_continuous_size = HIDDEN_CONT_SIZE
    dropout_rate = DROPOUT
    batch_size = BATCH_SIZE
    patch_len = PATCH_LEN
    patch_stride = PATCH_STRIDE
    num_transformer_layers = NUM_TRANSFORMER_LAYERS

    # Optuna-Hyperparameter überschreiben die globalen Standardwerte falls Optuna läuft
    if hpo_params is not None:
        learning_rate = float(hpo_params.get("learning_rate", learning_rate))
        d_model = int(hpo_params.get("d_model", d_model))
        attention_head_size = int(hpo_params.get("attention_head_size", attention_head_size))
        hidden_continuous_size = int(d_model / 2)
        dropout_rate = float(hpo_params.get("dropout", dropout_rate))
        num_transformer_layers = int(hpo_params.get("num_transformer_layers", num_transformer_layers))

    # Seed setzen für Reproduzierbarkeit
    set_seed(seed_value)

    # Konfig aller Modellinfos dient der Dokumentation
    seed_config = {
        "model": "PatchTST",
        "encoder_len": ENCODER_LEN,
        "pred_len": PRED_LEN,
        "batch_size": batch_size,
        "lr": learning_rate,
        "d_model": d_model,
        "attn_head_size": attention_head_size,
        "hidden_cont_size": hidden_continuous_size,
        "dropout": dropout_rate,
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "num_transformer_layers": num_transformer_layers,
        "series_emb_dim": SERIES_EMB_DIM,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lags": LAG_LIST,
        "rolling_windows": ROLLING_WINDOWS,
        "max_series": MAX_SERIES,
        "device": DEVICE,
        "base_seed": BASE_SEED,
        "num_seeds": NUM_SEEDS,
        "seed": int(seed_value),
        "known_real_features": KNOWN_REAL_FEATURES,
        "static_categoricals": STATIC_CATEGORICALS,
    }

    # Sichern der JSON Files für die Konfiguration und der Information über das genutze System 
    save_json(run_dir / "config.json", seed_config)
    save_json(run_dir / "system_info.json", get_system_info())

    # Erstellen von Trainings- und Validierungs-Dataset
    training_ds, val_ds = build_timeseries_datasets(df)
    series_mapping = get_series_id_mapping(training_ds)
    # Anzahl der Zeitreihen bestimmen
    num_series = len(series_mapping) if series_mapping is not None else int(df["series_id"].nunique())
    feature_names = list(training_ds.reals)

    # DataLoader konfigurieren: num_workers für paralleles Datenladen - Num Workers wurde viel getestet
    # In der Cloud hat sich num_workers 6 als beste Konfiguration herausgestellt. Ist bei allem Modellen identisch
    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=6,
        persistent_workers=False,
        pin_memory=(TORCH_DEVICE == "cuda"),
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=6,
        persistent_workers=False,
        pin_memory=(TORCH_DEVICE == "cuda"),
    )

    # Input-Dimension dynamisch aus dem ersten Batch bestimmen
    sample_batch = next(iter(train_loader))
    sample_x, _ = sample_batch
    input_dim = int(sample_x["encoder_cont"].shape[-1])

    # Modellerstellung über PatchTST Klasse
    model = PatchTSTModel(
        input_dim=input_dim,
        horizon=PRED_LEN,
        num_series=num_series,
        learning_rate=learning_rate,
        d_model=d_model,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout_rate,
        patch_len=patch_len,
        patch_stride=patch_stride,
        num_transformer_layers=num_transformer_layers,
        series_emb_dim=SERIES_EMB_DIM,
        mase_denoms=mase_denoms,
        series_mapping=series_mapping,
        feature_names=feature_names,
    )

    # Logger um die einzelnen Metriken pro Epoche zu loggen
    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")

    # Checkpoint eines Modells abrufen. Bestes Modell zum derzeigen Stand speichern.
    ckpt = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val_mase", # Speichert den Checkpoint mit dem niedrigsten val_mase
        save_top_k=1,       # Behält nur den besten Checkpoint
        mode="min",
    )

    # Iportierte Lighting Funktion zur Überwachung und Steuerung der Verbesserung während des Tranings aufrufen. 
    early = EarlyStopping(
        monitor="val_mase",
        patience=PATIENCE,
        mode="min",
        min_delta=MIN_DELTA,
    )

    # Überwacht und loggt die aktuelle Lernrate nach jeder Epoche
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer konfigurieren, alle Trainingskomponenten zusammenfassen
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,
        devices=1,
        precision="bf16-mixed" if TORCH_DEVICE == "cuda" else "32-true",
        gradient_clip_val=0.1, # # Gradient Clipping verhindert explodierende Gradienten. Wurde bewusst festgesetzt und nicht per Optuna bestimmt um den Suchraum klein zu halten
        logger=csv_logger,
        callbacks=[ckpt, early, lr_monitor],
        log_every_n_steps=70,
        enable_progress_bar=True, # Gib eine Fortschrittsanzeige aus beim Traning
        enable_model_summary=False,
        profiler=None,
    )

    #  Trainingszeit messen für die Kosten und Zeitberechnung in der Arbeit
    total_start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    total_time = time.perf_counter() - total_start

    # Bestes Modell aus dem Checkpoint laden um die finale Prognose zum Schluss des Tranings auf dem Valsplit zu machen
    best_path = ckpt.best_model_path
    if best_path:
        model = PatchTSTModel.load_from_checkpoint(
            best_path,
            input_dim=input_dim,
            horizon=PRED_LEN,
            num_series=num_series,
            learning_rate=learning_rate,
            d_model=d_model,
            attention_head_size=attention_head_size,
            hidden_continuous_size=hidden_continuous_size,
            dropout=dropout_rate,
            patch_len=patch_len,
            patch_stride=patch_stride,
            num_transformer_layers=num_transformer_layers,
            series_emb_dim=SERIES_EMB_DIM,
            mase_denoms=mase_denoms,
            series_mapping=series_mapping,
            feature_names=feature_names,
        )

    # Ermittelte Metriken speichern
    # Liest die Lightning CSV-Logs und rekonstruiert eine pro-Epoche Tabelle
    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
    epoch_rows = []
    if metrics_csv.exists():
        log_df = pd.read_csv(metrics_csv)
        epochs = sorted(int(ep) for ep in log_df["epoch"].dropna().unique().tolist()) if "epoch" in log_df.columns else []
        for ep in epochs:
            subset = log_df[log_df["epoch"] == ep]

            # Standardzeile mit NaN-Werten initialisieren. Werden nachher mit korrekten Werten befüllt
            row = {
                "epoch": ep + 1,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "val_mase": np.nan,
                "val_mase_w1": np.nan,
                "val_mase_w2": np.nan,
                "val_mase_w3": np.nan,
                "val_mase_w4": np.nan,
                "val_wape": np.nan,
                "val_wape_w1": np.nan,
                "val_wape_w2": np.nan,
                "val_wape_w3": np.nan,
                "val_wape_w4": np.nan,
                "val_mse": np.nan,
                "lr": float(learning_rate),
                "epoch_time_sec": np.nan,
            }

            # verschiedene Spaltennamen je nach Lightning-Version testen, jenachdem wie es von Lightning bennant wurde
            for train_key in ["train_loss_epoch", "train_loss", "train_loss_step"]:
                if train_key in subset.columns:
                    tmp = subset[train_key].dropna()
                    if len(tmp) > 0:
                        row["train_loss"] = float(tmp.iloc[-1])
                        break

            # Validierungsmetriken direkt aus den CSV-Spalten lesen
            for col in [
                "val_loss",
                "val_mase",
                "val_mase_w1",
                "val_mase_w2",
                "val_mase_w3",
                "val_mase_w4",
                "val_wape",
                "val_wape_w1",
                "val_wape_w2",
                "val_wape_w3",
                "val_wape_w4",
                "val_mse",
            ]:
                if col in subset.columns:
                    tmp = subset[col].dropna()
                    if len(tmp) > 0:
                        row[col] = float(tmp.iloc[-1])

            # LR - verschiedene Spaltennamen je nach Optimizer-Konfiguration ausprobieren
            for lr_key in ["lr-Adam", "lr"]:
                if lr_key in subset.columns:
                    tmp = subset[lr_key].dropna()
                    if len(tmp) > 0:
                        row["lr"] = float(tmp.iloc[-1])
                        break

            epoch_rows.append(row)
    
    # Finale Prognosen und Metriken auf dem Validierungsset berechnen
    prediction_output = collect_predictions(model, val_loader, series_mapping, model.device)

    # Nur zu Sicherheit. Nicht nötig
    # if not epoch_rows:
    #     epoch_rows = [{
    #         "epoch": 1,
    #         "train_loss": np.nan,
    #         "val_loss": np.nan,
    #         "val_mase": np.nan,
    #         "val_mase_w1": np.nan,
    #         "val_mase_w2": np.nan,
    #         "val_mase_w3": np.nan,
    #         "val_mase_w4": np.nan,
    #         "val_wape": np.nan,
    #         "val_wape_w1": np.nan,
    #         "val_wape_w2": np.nan,
    #         "val_wape_w3": np.nan,
    #         "val_wape_w4": np.nan,
    #         "val_mse": np.nan,
    #         "lr": float(learning_rate),
    #         "epoch_time_sec": float(total_time),
    #     }]

    # Durchschnittliche Zeit pro Epoche berechnen
    epoch_rows[-1]["epoch_time_sec"] = float(total_time / max(len(epoch_rows), 1))

    # Metriken als CSV mit korrekter Spaltenreihenfolge speichern
    metrics_dataframe = reorder_metrics_columns(pd.DataFrame(epoch_rows))
    metrics_dataframe.to_csv(run_dir / "metrics.csv", index=False)

    # Trainingsplots erstellen und speichern - aus run_logger
    save_plots(run_dir, epoch_rows)

    # Rücktransformation aus Log-Space in Originalraum
    pred_y = np.expm1(prediction_output["prediction_log"]).clip(min=0.0)
    true_y = np.expm1(prediction_output["true_log"]).clip(min=0.0)
    # Metrikberechnung auf den finalen Prognosen des besten Modell-Checkpoints
    val_metrics = eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, prediction_output["series_ids"], mase_denoms)
    external_val_loss = calc_mse_loss_logspace(prediction_output["prediction_log"], prediction_output["true_log"])

    # Finale Metriken in die letzte Epochenzeile schreiben
    if epoch_rows:
        epoch_rows[-1]["val_loss"] = float(external_val_loss)
        epoch_rows[-1]["val_mase"] = float(val_metrics["mase"])
        epoch_rows[-1]["val_mase_w1"] = float(val_metrics["mase_w1"])
        epoch_rows[-1]["val_mase_w2"] = float(val_metrics["mase_w2"])
        epoch_rows[-1]["val_mase_w3"] = float(val_metrics["mase_w3"])
        epoch_rows[-1]["val_mase_w4"] = float(val_metrics["mase_w4"])
        epoch_rows[-1]["val_wape"] = float(val_metrics["wape"])
        epoch_rows[-1]["val_wape_w1"] = float(val_metrics["wape_w1"])
        epoch_rows[-1]["val_wape_w2"] = float(val_metrics["wape_w2"])
        epoch_rows[-1]["val_wape_w3"] = float(val_metrics["wape_w3"])
        epoch_rows[-1]["val_wape_w4"] = float(val_metrics["wape_w4"])
        epoch_rows[-1]["val_mse"] = float(val_metrics["mse"])

    # MetrikCSV nochmals überschreiben wenn die finalen Werte da sind
    metrics_dataframe = reorder_metrics_columns(pd.DataFrame(epoch_rows))
    metrics_dataframe.to_csv(run_dir / "metrics.csv", index=False)

    # Beste Epoche anhand des niedrigsten val_mase bestimmen - Für summary
    if "val_mase" in metrics_dataframe.columns and metrics_dataframe["val_mase"].notna().any():
        best_epoch_idx = metrics_dataframe["val_mase"].astype(float).idxmin()
        best_epoch_row = metrics_dataframe.loc[best_epoch_idx].to_dict()
    else:
        best_epoch_row = metrics_dataframe.iloc[-1].to_dict()

    # Ausgabe der Summary des Tranings als JSON File
    summary = {
        "seed": int(seed_value),
        "best_model_path": best_path,
        "best_epoch": int(best_epoch_row["epoch"]) if pd.notna(best_epoch_row.get("epoch", np.nan)) else None,
        "best_val_mase": float(ckpt.best_model_score) if ckpt.best_model_score is not None else None,
        "total_time_sec": float(total_time),
        "epochs_ran": int(len(epoch_rows)),
        "val_mase": float(best_epoch_row["val_mase"]) if pd.notna(best_epoch_row.get("val_mase", np.nan)) else None,
        "val_mase_w1": float(best_epoch_row["val_mase_w1"]) if pd.notna(best_epoch_row.get("val_mase_w1", np.nan)) else None,
        "val_mase_w2": float(best_epoch_row["val_mase_w2"]) if pd.notna(best_epoch_row.get("val_mase_w2", np.nan)) else None,
        "val_mase_w3": float(best_epoch_row["val_mase_w3"]) if pd.notna(best_epoch_row.get("val_mase_w3", np.nan)) else None,
        "val_mase_w4": float(best_epoch_row["val_mase_w4"]) if pd.notna(best_epoch_row.get("val_mase_w4", np.nan)) else None,
        "val_wape": float(best_epoch_row["val_wape"]) if pd.notna(best_epoch_row.get("val_wape", np.nan)) else None,
        "val_wape_w1": float(best_epoch_row["val_wape_w1"]) if pd.notna(best_epoch_row.get("val_wape_w1", np.nan)) else None,
        "val_wape_w2": float(best_epoch_row["val_wape_w2"]) if pd.notna(best_epoch_row.get("val_wape_w2", np.nan)) else None,
        "val_wape_w3": float(best_epoch_row["val_wape_w3"]) if pd.notna(best_epoch_row.get("val_wape_w3", np.nan)) else None,
        "val_wape_w4": float(best_epoch_row["val_wape_w4"]) if pd.notna(best_epoch_row.get("val_wape_w4", np.nan)) else None,
        "val_mse": float(best_epoch_row["val_mse"]) if pd.notna(best_epoch_row.get("val_mse", np.nan)) else None,
    }

    # Summary der Metriken speichern
    save_json(run_dir / "summary.json", summary)
    save_excel(run_dir, seed_config, epoch_rows, summary)

    print("Gespeichert unter:", run_dir / "metrics.xlsx")
    print("Bester Checkpoint unter:", best_path)
    return summary


# Hilfsfunktion erzeugt eine Liste von Seeds für einen Optuna-Trial
def build_trial_seed_list(trial_base_seed: int, trial_seed_count: int) -> list[int]:
    if trial_seed_count <= 1:
        return [trial_base_seed]
    return [trial_base_seed + i for i in range(trial_seed_count)]


# objective_factory() ist eine Funktion die eine andere Funktion zurückgibt. Das ist nötig,
# weil Optuna intern nur def objective(trial) akzeptiert, die objective-Funktion aber
# zusätzlich df und mase_denoms benötigt — diese werden hier einmalig mitgegeben.
def objective_factory(df: pd.DataFrame, mase_denoms: dict, optuna_base_dir: Path):
    def objective(optuna_trial: optuna.Trial) -> float:
        # Hyperparameter für diesen Trial sampeln
        suggested_hyperparameters = suggest_hyperparameters(optuna_trial)
        # Separates Verzeichnis für jeden Trial anlegen
        trial_run_dir = optuna_base_dir / f"optuna_trial_{optuna_trial.number:04d}"
        trial_run_dir.mkdir(parents=True, exist_ok=True)

        trial_seed_list = build_trial_seed_list(BASE_SEED, OPTUNA_SEEDS_PER_TRIAL)
        validation_mase_values = []

        # Jeden Seed des Trials trainieren und val_mase sammeln
        for seed_value in trial_seed_list:
            seed_run_dir = trial_run_dir / f"seed_{seed_value}"
            seed_run_dir.mkdir(parents=True, exist_ok=True)
            seed_summary = train_one_seed(
                df=df,
                mase_denoms=mase_denoms,
                run_dir=seed_run_dir,
                seed_value=seed_value,
                hpo_params=suggested_hyperparameters,
            )
            validation_mase_values.append(float(seed_summary["best_val_mase"]))

        # Mittleren MASE über alle Seeds als Trial-Ergebnis zurückgeben
        mean_validation_mase = float(np.mean(validation_mase_values)) if validation_mase_values else float("inf")
        # Trial-Metadaten für spätere Analyse speichern
        optuna_trial.set_user_attr("trial_seeds", trial_seed_list)
        optuna_trial.set_user_attr("val_mases", validation_mase_values)
        return mean_validation_mase

    return objective


# Hauptfuntion des Skripts das alle Funktionen aufruft und den kompletten Traningsdurchlauf startet
def main():
    # Daten laden und Zeitreihen-Features hinzufügen
    df = load_preprocessed()
    df = add_time_series_features(df)

    known_reals, unknown_reals = build_feature_columns()

    # Alle kontinuierlichen Features (Encoder + Decoder)
    all_features = known_reals + unknown_reals

    # Training Dataset aus dem gesamten Dataset erstellen (finales Subset), bei dem der Split mit train markiert ist
    train_df = df[df["split"] == "train"].copy()
   
    # MASE Nenner errechnen
    mase_denoms = compute_mase_denominators(train_df, seasonality=7)

    # Dient der Dokumentation
    config = {
        "model": "PatchTST",
        "encoder_len": ENCODER_LEN,
        "pred_len": PRED_LEN,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "d_model": D_MODEL,
        "attn_head_size": ATTN_HEAD_SIZE,
        "hidden_cont_size": HIDDEN_CONT_SIZE,
        "dropout": DROPOUT,
        "patch_len": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "num_transformer_layers": NUM_TRANSFORMER_LAYERS,
        "series_emb_dim": SERIES_EMB_DIM,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lags": LAG_LIST,
        "rolling_windows": ROLLING_WINDOWS,
        "known_real_features": known_reals,
        "unknown_real_features": unknown_reals,
        "static_categoricals": STATIC_CATEGORICALS,
        "all_features": all_features,
        "max_series": MAX_SERIES,
        "device": DEVICE,
        "base_seed": BASE_SEED,
        "num_seeds": NUM_SEEDS,
        "use_optuna": USE_OPTUNA,
        "optuna_trials": OPTUNA_TRIALS,
    }

    # Verzeichnis für den Run erstellen
    ts = time.strftime("%Y%m%d-%H%M%S")
    parent_run_dir = RUNS_DIR / f"{ts}_PatchTST_seed={BASE_SEED}__num_seeds={NUM_SEEDS}"
    parent_run_dir.mkdir(parents=True, exist_ok=True)
    save_json(parent_run_dir / "config.json", config)
    save_json(parent_run_dir / "system_info.json", get_system_info())

    # Wenn Oputuna eingeschaltet ist, dann Optuna Study initialisieren, verschiedene Plots erstellen und Ergebnisse wegspeichern
    if USE_OPTUNA:
        optuna_run_dir = parent_run_dir / "optuna"
        optuna_run_dir.mkdir(parents=True, exist_ok=True)

        objective_function = objective_factory(df=df, mase_denoms=mase_denoms, optuna_base_dir=optuna_run_dir)
        # neue Study erstellen
        optuna_study = optuna.create_study(direction=OPTUNA_DIRECTION)
        # Optimiertung starten
        optuna_study.optimize(objective_function, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_SEC)

        # Plots erstellen
        plot_optimization_history(optuna_study).write_html(optuna_run_dir / "optuna_optimization_history.html")
        plot_param_importances(optuna_study).write_html(optuna_run_dir / "optuna_param_importances.html")
        plot_parallel_coordinate(optuna_study).write_html(optuna_run_dir / "optuna_parallel_coordinate.html")
        plot_slice(optuna_study).write_html(optuna_run_dir / "optuna_slice.html")

        # Beste Kofiguration als JSON Speichern
        best_hyperparameters = optuna_study.best_params
        save_json(optuna_run_dir / "optuna_summary.json",
            {
                "best_value": float(optuna_study.best_value),
                "best_params": best_hyperparameters,
                "n_trials": int(len(optuna_study.trials)),
            },
        )
    # Wenn kein Optuna genutzt wird bleibt es leer
    else:
        best_hyperparameters = None

    # Variabeln für die besten Ergebnisse pro Seed erstellen
    all_seed_summaries = []
    best_overall_val_mase = float("inf")
    best_overall_seed = None
    best_overall_model_path = None

    # Jeden Seed nacheinander trainieren
    for seed_offset in range(NUM_SEEDS):
        current_seed = int(BASE_SEED) + int(seed_offset)
        run_dir = parent_run_dir / f"seed_{current_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        seed_summary = train_one_seed(
            df=df,
            mase_denoms=mase_denoms,
            run_dir=run_dir,
            seed_value=current_seed,
            hpo_params=best_hyperparameters,
        )
        all_seed_summaries.append(seed_summary)

        # Den besten MASE Wert über alle Seeds bestimmen
        if np.isfinite(seed_summary["best_val_mase"]) and seed_summary["best_val_mase"] < best_overall_val_mase:
            best_overall_val_mase = float(seed_summary["best_val_mase"])
            best_overall_seed = int(current_seed)
            best_overall_model_path = seed_summary["best_model_path"]

    # Übergrreifende Summary über alle Seeds erstellen
    overall_summary = {
        "best_overall_seed": best_overall_seed,
        "best_overall_val_mase": float(best_overall_val_mase),
        "best_overall_model_path": best_overall_model_path,
        "seed_summaries": all_seed_summaries,
        "optuna_best_params": best_hyperparameters,
    }
    save_json(parent_run_dir / "overall_summary.json", overall_summary)

    print("Gespeichert unter:", parent_run_dir / "overall_summary.json")

# Aufrufen der Hauptfunktion main() zum starten des Skriptes
if __name__ == "__main__":
    main()
