"""
Gemeinsames Test-Script für 28-Tage-Testhorizont über vier bereits trainierte Modelle:
- Many-to-One LSTM
- Seq-to-Seq LSTM
- Temporal Fusion Transformer (TFT)
- PatchTST

Ziel:
    Modelle werden NICHT neu trainiert. Es werden vorhandene Checkpoints geladen und auf dem
    Test-Split mit demselben Horizont (28 Tage) getestet. Berechnet werden dieselben Metriken
    wie in den Trainingsdateien: MASE, MSE, WAPE sowie Wochenwerte W1-W4.

Voraussetzungen:
    - data/preprocessed/m5_long.csv
    - Trainingsdateien liegen im Projektroot
    - Checkpoint-Pfade werden oben in dieser Datei fest eingetragen

Nutzung:
    1. Oben in der Datei die vier Checkpoint-Pfade anpassen.
    2. Script aus dem Projektroot starten:

       python test_all_models_28d_fixed_paths.py
"""

# Aktiviert moderne Typ-Hints und verhindert, dass alle Typen sofort zur Laufzeit ausgewertet werden müssen.
from __future__ import annotations

print("[START] test_all_models_28d_fixed_paths.py wird ausgeführt", flush=True)

# importlib.util lädt Trainingsdateien dynamisch als Module, damit Modellklassen und Feature-Funktionen wiederverwendet werden können.
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Globale Einstellungen
# -----------------------------------------------------------------------------
# Automatische Gerätewahl: GPU, falls verfügbar; sonst CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Standardpfad zum vorverarbeiteten Datensatz.
PREP_DIR = Path("data") / "preprocessed"
CSV_PATH = PREP_DIR / "m5_long.csv"

# Ausgabeordner für die Testergebnisse.
OUT_DIR = Path("runs") / "test"


# -----------------------------------------------------------------------------
# Feste Pfade zu den Trainingsdateien
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Feste Pfade zu den Trainingsdateien
# -----------------------------------------------------------------------------
# Diese Dateien enthalten die Modellklassen, Feature-Definitionen und Hilfsfunktionen,
# die beim Training verwendet wurden. Das Test-Script lädt diese Dateien als Module.
MANY_TO_ONE_TRAIN_PATH = Path("code/train_lstm_many_to_one_multi_seed.py")
SEQ_TO_SEQ_TRAIN_PATH = Path("code/train_lstm_seq_to_seq_multi_seed.py")
TFT_TRAIN_PATH = Path("code/train_TFT_multi_seed.py")
PATCHTST_TRAIN_PATH = Path("code/train_PatchTST_multi_seed.py")


# -----------------------------------------------------------------------------
# Feste Pfade zu den gespeicherten Modell-Checkpoints
# -----------------------------------------------------------------------------
# WICHTIG:
# Diese vier Pfade musst du an deine tatsächlichen Run-Ordner anpassen.
#
# LSTM-Modelle speichern üblicherweise .pt-Dateien:
#   runs/lstm/.../best_model.pt
#
# TFT und PatchTST speichern üblicherweise Lightning-Checkpoints:
#   runs/tft/.../best.ckpt
#   runs/patchtst/.../best.ckpt
MANY_TO_ONE_CKPT_PATH = Path("Trained Models/best_model_Many_to_One.pt")
SEQ_TO_SEQ_CKPT_PATH = Path("Trained Models/best_model_Seq_to_Seq.pt")
TFT_CKPT_PATH = Path("Trained Models/best_model_TFT.ckpt")
PATCHTST_CKPT_PATH = Path("Trained Models/best_model_PatchTST.ckpt")


# -----------------------------------------------------------------------------
# Auswahl der Modelle, die getestet werden sollen
# -----------------------------------------------------------------------------
# Alle vier Modelle testen:
MODELS_TO_TEST = ["many_to_one", "seq_to_seq", "tft", "patchtst"]
#MODELS_TO_TEST = ["many_to_one", "seq_to_seq"]

# -----------------------------------------------------------------------------
# Detailprognosen speichern?
# -----------------------------------------------------------------------------
# False:
#   Es werden nur die kompakten Metrik-Dateien gespeichert:
#       - test_summary_all_models.csv
#       - test_summary_all_models.json
#
# True:
#   Zusätzlich wird pro Modell eine große test_predictions.csv gespeichert.
#   Diese Datei enthält jede einzelne Prognose pro Serie und Horizonttag.
#   Bei vielen Serien kann das sehr groß werden und unter Windows/VS Code RAM-Probleme verursachen.
SAVE_DETAILED_PREDICTIONS = False

# -----------------------------------------------------------------------------
# Testfenster-Logik
# -----------------------------------------------------------------------------
# True:
#   Pro Serie wird genau EIN 28-Tage-Testfenster ausgewertet:
#       erster Testtag bis erster Testtag + 27.
#
# False:
#   Es werden alle möglichen Sliding-Windows im Testsplit ausgewertet.
#   Das kann sehr viele Samples erzeugen und das Script stark verlangsamen.
USE_ONLY_FIRST_TEST_WINDOW_PER_SERIES = True

# Beispiele:
# Nur TFT testen:
# MODELS_TO_TEST = ["tft"]
#
# Nur die beiden LSTM-Modelle testen:
# MODELS_TO_TEST = ["many_to_one", "seq_to_seq"]


# Länge des historischen Kontextfensters: pro Forecast werden die letzten 56 Tage vor Teststart genutzt.
SEQ_LEN = 56

# Länge des Testhorizonts: es werden 28 Tage prognostiziert und bewertet.
HORIZON = 28

BATCH_SIZE_DEFAULT = 1024
NUM_WORKERS = 0


# -----------------------------------------------------------------------------
# Hilfsfunktionen: IO / Module / Checkpoints
# -----------------------------------------------------------------------------
# Speichert Ergebnisse und Konfigurationen als JSON-Datei.
def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def log_step(message: str) -> None:
    """Einfache Fortschrittsausgabe, damit sichtbar ist, wo das Script gerade steht."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# Lädt eine Trainingsdatei als Python-Modul, damit dieselbe Modellarchitektur und Featurelogik wie im Training genutzt wird.
def load_module(module_path: Path, module_name: str):
    if not module_path.exists():
        raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Konnte Modul nicht laden: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Lädt den vorverarbeiteten Long-Format-Datensatz mit Split-, Ziel- und Feature-Spalten.
def load_preprocessed(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Keine vorverarbeitete Datei gefunden: {csv_path}")
    return pd.read_csv(csv_path, parse_dates=["date"])


def find_latest_checkpoint(search_dir: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    if not search_dir.exists():
        return None
    for pattern in patterns:
        candidates.extend(search_dir.rglob(pattern))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_json_if_exists(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def infer_lstm_hparams_from_run(ckpt_path: Path, fallback: dict) -> dict:
    """Liest optionale Hyperparameter aus Run-Dateien; nutzt Fallbacks aus Trainingsmodul."""
    run_dir = ckpt_path.parent
    parent_dir = run_dir.parent

    hparams = dict(fallback)

    # Normale Run-Konfigs
    for config_path in [run_dir / "run_config.json", parent_dir / "run_config.json"]:
        cfg = read_json_if_exists(config_path)
        if cfg:
            if "hidden_size" in cfg:
                hparams["hidden_size"] = int(cfg["hidden_size"])
            if "batch_size" in cfg:
                hparams["batch_size"] = int(cfg["batch_size"])

    # Optuna / best params, falls vorhanden
    for summary_path in [parent_dir / "overall_summary_bestparams.json", run_dir / "overall_summary_bestparams.json"]:
        summary = read_json_if_exists(summary_path)
        best_params = summary.get("optuna_best_params")
        if isinstance(best_params, dict):
            if "hidden_size" in best_params:
                hparams["hidden_size"] = int(best_params["hidden_size"])
            if "num_layers" in best_params:
                hparams["num_layers"] = int(best_params["num_layers"])
            if "dropout" in best_params:
                hparams["dropout"] = float(best_params["dropout"])

    return hparams


def torch_load_state_dict_safe(ckpt_path: Path) -> dict:
    """
    Lädt einen PyTorch-State-Dict.

    weights_only=True vermeidet die FutureWarning bei reinen Gewichtedateien.
    Falls eine ältere PyTorch-Version dieses Argument nicht kennt, wird auf den
    klassischen Aufruf zurückgefallen.
    """
    try:
        return torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=DEVICE)


def infer_lstm_hparams_from_state_dict(state_dict: dict, hparams: dict) -> dict:
    """
    Leitet LSTM-Hyperparameter direkt aus den gespeicherten Gewichten ab.

    Das ist robuster als nur run_config.json zu lesen, weil bei Optuna- oder
    Best-Parameter-Runs die Architektur manchmal nicht vollständig in der
    direkt benachbarten Config steht.

    Wichtige Ableitungen:
        - hidden_size aus fc.weight:
            fc.weight hat Shape (horizon, hidden_size)
        - num_layers aus vorhandenen lstm.weight_ih_lX Schlüsseln:
            l0, l1, l2 bedeutet 3 Layer
    """
    inferred = dict(hparams)

    if "fc.weight" in state_dict:
        inferred["hidden_size"] = int(state_dict["fc.weight"].shape[1])

    layer_indices = []
    for key in state_dict.keys():
        match = re.match(r"lstm\.weight_ih_l(\d+)$", key)
        if match:
            layer_indices.append(int(match.group(1)))

    if layer_indices:
        inferred["num_layers"] = int(max(layer_indices) + 1)

    return inferred


# -----------------------------------------------------------------------------
# Gemeinsame Metriken analog zu den Trainingsdateien
# -----------------------------------------------------------------------------
# Berechnet MASE-Nenner pro Serie ausschließlich aus dem Trainingssplit. Dadurch bleibt die Testbewertung sauber.
def compute_mase_denominators(train_df: pd.DataFrame, seasonality: int = 7) -> dict:
    denoms = {}
    for series_id, series_values in train_df.groupby("series_id"):
        series_values = series_values.sort_values("time_idx")
        y = series_values["y"].values.astype(np.float32)
        if len(y) <= seasonality:
            denoms[str(series_id)] = 1.0
            continue
        diff = np.abs(y[seasonality:] - y[:-seasonality])
        den = float(np.mean(diff)) if np.mean(diff) > 0 else 1.0
        denoms[str(series_id)] = den
    return denoms


# Zentrale Metrikfunktion für alle Modelle. Erwartet Vorhersagen und Istwerte im Originalraum.
def eval_mase_mse_wape_weekly_from_arrays(
    pred_y: np.ndarray,
    true_y: np.ndarray,
    series_ids: np.ndarray,
    mase_denoms: dict,
) -> dict:
    week_slices = [(0, 7), (7, 14), (14, 21), (21, 28)]

    pred_y = np.asarray(pred_y, dtype=np.float32)
    true_y = np.asarray(true_y, dtype=np.float32)
    series_ids = np.asarray(series_ids, dtype=object)

    pred_y = np.clip(pred_y, a_min=0.0, a_max=None)
    true_y = np.clip(true_y, a_min=0.0, a_max=None)

    abs_err = np.abs(pred_y - true_y)
    mse = float(np.mean((pred_y - true_y) ** 2))

    den = np.array([float(mase_denoms.get(str(sid), 1.0)) for sid in series_ids], dtype=np.float32)
    den = np.where(den > 0, den, 1.0)

    mae_overall = np.mean(abs_err, axis=1)
    mase = float(np.mean(mae_overall / den))

    mase_weeks = []
    for start, end in week_slices:
        mae_week = np.mean(abs_err[:, start:end], axis=1)
        mase_weeks.append(float(np.mean(mae_week / den)))

    wape_num = float(np.sum(abs_err))
    wape_den = float(np.sum(true_y))
    wape = (wape_num / wape_den) if wape_den > 0 else float("nan")

    wape_weeks = []
    for start, end in week_slices:
        num = float(np.sum(abs_err[:, start:end]))
        den_w = float(np.sum(true_y[:, start:end]))
        wape_weeks.append((num / den_w) if den_w > 0 else float("nan"))

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



def eval_mase_wape_by_horizon_day(
    pred_y: np.ndarray,
    true_y: np.ndarray,
    series_ids: np.ndarray,
    mase_denoms: dict,
) -> pd.DataFrame:
    """
    Berechnet MASE und WAPE separat für jeden der 28 Horizonttage.

    MASE pro Horizonttag:
        mean_i(|y_i,h - yhat_i,h| / mase_denom_i)

    WAPE pro Horizonttag:
        sum_i(|y_i,h - yhat_i,h|) / sum_i(y_i,h)
    """
    pred_y = np.asarray(pred_y, dtype=np.float32)
    true_y = np.asarray(true_y, dtype=np.float32)
    series_ids = np.asarray(series_ids, dtype=object)

    pred_y = np.clip(pred_y, a_min=0.0, a_max=None)
    true_y = np.clip(true_y, a_min=0.0, a_max=None)

    abs_err = np.abs(pred_y - true_y)

    den = np.array(
        [float(mase_denoms.get(str(sid), 1.0)) for sid in series_ids],
        dtype=np.float32,
    )
    den = np.where(den > 0, den, 1.0)

    rows = []
    for h in range(pred_y.shape[1]):
        mase_h = float(np.mean(abs_err[:, h] / den))

        wape_den_h = float(np.sum(true_y[:, h]))
        wape_h = float(np.sum(abs_err[:, h]) / wape_den_h) if wape_den_h > 0 else float("nan")

        rows.append(
            {
                "horizon_day": h + 1,
                "mase": mase_h,
                "wape": wape_h,
            }
        )

    return pd.DataFrame(rows)


def save_horizon_metrics(
    run_out_dir: Path,
    model_name: str,
    pred_y: np.ndarray,
    true_y: np.ndarray,
    series_ids: np.ndarray,
    mase_denoms: dict,
) -> dict:
    """
    Speichert MASE und WAPE je Horizonttag für ein Modell.

    Pro Modell entstehen:
        - horizon_metrics_mase_wape.csv
        - horizon_metrics_mase_wape.png

    Zusätzlich gibt die Funktion ein Dictionary zurück:
        mase_day_1 ... mase_day_28
        wape_day_1 ... wape_day_28

    Diese Werte werden in die Modell-Summary übernommen.
    """
    horizon_metrics_df = eval_mase_wape_by_horizon_day(
        pred_y=pred_y,
        true_y=true_y,
        series_ids=series_ids,
        mase_denoms=mase_denoms,
    )
    horizon_metrics_df.insert(0, "model", model_name)

    run_out_dir.mkdir(parents=True, exist_ok=True)
    horizon_metrics_df.to_csv(run_out_dir / "horizon_metrics_mase_wape.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(horizon_metrics_df["horizon_day"], horizon_metrics_df["mase"], marker="o", label="MASE")
    plt.plot(horizon_metrics_df["horizon_day"], horizon_metrics_df["wape"], marker="o", label="WAPE")
    plt.xlabel("Forecast-Horizonttag")
    plt.ylabel("Metrikwert")
    plt.title(f"MASE und WAPE je Horizonttag - {model_name}")
    plt.xticks(range(1, HORIZON + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_out_dir / "horizon_metrics_mase_wape.png", dpi=150)
    plt.close()

    wide_metrics = {}
    for _, row in horizon_metrics_df.iterrows():
        day = int(row["horizon_day"])
        wide_metrics[f"mase_day_{day}"] = float(row["mase"])
        wide_metrics[f"wape_day_{day}"] = float(row["wape"])

    return wide_metrics


def save_combined_horizon_metrics(out_dir: Path, results: dict) -> None:
    """
    Erstellt eine modellübergreifende Übersicht über MASE und WAPE je Horizonttag.

    Dateien:
        - test_horizon_metrics_all_models.csv
        - test_horizon_mase_all_models.png
        - test_horizon_wape_all_models.png
    """
    rows = []

    for model_name, metrics in results.items():
        for day in range(1, HORIZON + 1):
            mase_key = f"mase_day_{day}"
            wape_key = f"wape_day_{day}"

            if mase_key in metrics and wape_key in metrics:
                rows.append(
                    {
                        "model": model_name,
                        "horizon_day": day,
                        "mase": float(metrics[mase_key]),
                        "wape": float(metrics[wape_key]),
                    }
                )

    if not rows:
        return

    df_horizon = pd.DataFrame(rows)
    df_horizon.to_csv(out_dir / "test_horizon_metrics_all_models.csv", index=False)

    plt.figure(figsize=(10, 5))
    for model_name, group in df_horizon.groupby("model"):
        group = group.sort_values("horizon_day")
        plt.plot(group["horizon_day"], group["mase"], marker="o", label=model_name)
    plt.xlabel("Forecast-Horizonttag")
    plt.ylabel("MASE")
    plt.title("MASE je Horizonttag über alle Serien")
    plt.xticks(range(1, HORIZON + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "test_horizon_mase_all_models.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_name, group in df_horizon.groupby("model"):
        group = group.sort_values("horizon_day")
        plt.plot(group["horizon_day"], group["wape"], marker="o", label=model_name)
    plt.xlabel("Forecast-Horizonttag")
    plt.ylabel("WAPE")
    plt.title("WAPE je Horizonttag über alle Serien")
    plt.xticks(range(1, HORIZON + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "test_horizon_wape_all_models.png", dpi=150)
    plt.close()


def eval_loss_logspace_from_arrays(pred_log: np.ndarray, true_log: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred_log) - np.asarray(true_log)) ** 2))


def save_predictions(out_path: Path, pred_y: np.ndarray, true_y: np.ndarray, series_ids: np.ndarray) -> None:
    rows = []
    for i in range(pred_y.shape[0]):
        for h in range(pred_y.shape[1]):
            rows.append({
                "sample_id": i,
                "series_id": str(series_ids[i]),
                "horizon_day": h + 1,
                "y_true": float(true_y[i, h]),
                "y_pred": float(pred_y[i, h]),
                "abs_error": float(abs(pred_y[i, h] - true_y[i, h])),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# -----------------------------------------------------------------------------
# LSTM: Datensatz-Vorbereitung
# -----------------------------------------------------------------------------
# Kodiert kategoriale IDs als Integer, weil LSTM-Embeddings numerische Indizes benötigen.
def add_category_codes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, list, list, list]:
    df = df.copy()
    series_ids = sorted(df["series_id"].unique())
    series_to_idx = {series_id: i for i, series_id in enumerate(series_ids)}
    idx_to_series = {i: series_id for series_id, i in series_to_idx.items()}

    item_ids = sorted(df["item_id"].unique())
    store_ids = sorted(df["store_id"].unique())
    state_ids = sorted(df["state_id"].unique())

    df["item_id_code"] = df["item_id"].map({v: i for i, v in enumerate(item_ids)}).astype(np.int64)
    df["store_id_code"] = df["store_id"].map({v: i for i, v in enumerate(store_ids)}).astype(np.int64)
    df["state_id_code"] = df["state_id"].map({v: i for i, v in enumerate(state_ids)}).astype(np.int64)

    return df, series_to_idx, idx_to_series, item_ids, store_ids, state_ids


# Baut Testfenster für Many-to-One-LSTM: 56 Tage Historie -> 28 Tage Zielhorizont.
def build_many_to_one_test_windows(df: pd.DataFrame, feature_cols: list[str], series_to_idx: dict):
    feature_list, y_log_list, series_list, item_list, store_list, state_list = [], [], [], [], [], []

    for series_id, series_values in df.groupby("series_id"):
        series_values = series_values.sort_values("time_idx").reset_index(drop=True)
        x_feat = series_values[feature_cols].astype(np.float32).values
        y_log = series_values["y_log"].astype(np.float32).values
        splits = series_values["split"].values
        item_code = int(series_values["item_id_code"].iloc[0])
        store_code = int(series_values["store_id_code"].iloc[0])
        state_code = int(series_values["state_id_code"].iloc[0])

        if USE_ONLY_FIRST_TEST_WINDOW_PER_SERIES:
            # Genau ein Testfenster pro Serie:
            # Start ist der erste Tag, an dem split == "test" gilt.
            test_positions = np.where(splits == "test")[0]
            if len(test_positions) == 0:
                continue
            candidate_days = [int(test_positions[0])]
        else:
            # Sliding-Window-Test:
            # Jeder mögliche Starttag im Testsplit wird als eigenes 28-Tage-Fenster bewertet.
            candidate_days = [
                day for day in range(SEQ_LEN, len(series_values) - HORIZON + 1)
                if splits[day] == "test"
            ]

        for day in candidate_days:
            if day < SEQ_LEN or day + HORIZON > len(series_values):
                continue
            feature_list.append(x_feat[day - SEQ_LEN: day])
            y_log_list.append(y_log[day: day + HORIZON])
            series_list.append(series_to_idx[series_id])
            item_list.append(item_code)
            store_list.append(store_code)
            state_list.append(state_code)

    if not feature_list:
        raise RuntimeError("Keine Test-Windows für Many-to-One LSTM gefunden.")

    return (
        np.stack(feature_list, 0),
        np.stack(y_log_list, 0),
        np.array(series_list, dtype=np.int64),
        np.array(item_list, dtype=np.int64),
        np.array(store_list, dtype=np.int64),
        np.array(state_list, dtype=np.int64),
    )


# Baut Testfenster für Seq-to-Seq-LSTM mit getrennten Encoder- und Decoder-Inputs.
def build_seq_to_seq_test_windows(df: pd.DataFrame, enc_cols: list[str], dec_cols: list[str], series_to_idx: dict):
    enc_list, dec_list, y_log_list, series_list, item_list, store_list, state_list = [], [], [], [], [], [], []

    for series_id, series_values in df.groupby("series_id"):
        series_values = series_values.sort_values("time_idx").reset_index(drop=True)
        x_enc = series_values[enc_cols].astype(np.float32).values
        x_dec = series_values[dec_cols].astype(np.float32).values
        y_log = series_values["y_log"].astype(np.float32).values
        splits = series_values["split"].values
        item_code = int(series_values["item_id_code"].iloc[0])
        store_code = int(series_values["store_id_code"].iloc[0])
        state_code = int(series_values["state_id_code"].iloc[0])

        if USE_ONLY_FIRST_TEST_WINDOW_PER_SERIES:
            # Genau ein Testfenster pro Serie:
            # Start ist der erste Tag, an dem split == "test" gilt.
            test_positions = np.where(splits == "test")[0]
            if len(test_positions) == 0:
                continue
            candidate_days = [int(test_positions[0])]
        else:
            # Sliding-Window-Test:
            # Jeder mögliche Starttag im Testsplit wird als eigenes 28-Tage-Fenster bewertet.
            candidate_days = [
                day for day in range(SEQ_LEN, len(series_values) - HORIZON + 1)
                if splits[day] == "test"
            ]

        for day in candidate_days:
            if day < SEQ_LEN or day + HORIZON > len(series_values):
                continue
            enc_list.append(x_enc[day - SEQ_LEN: day])
            dec_list.append(x_dec[day: day + HORIZON])
            y_log_list.append(y_log[day: day + HORIZON])
            series_list.append(series_to_idx[series_id])
            item_list.append(item_code)
            store_list.append(store_code)
            state_list.append(state_code)

    if not enc_list:
        raise RuntimeError("Keine Test-Windows für Seq-to-Seq LSTM gefunden.")

    return (
        np.stack(enc_list, 0),
        np.stack(dec_list, 0),
        np.stack(y_log_list, 0),
        np.array(series_list, dtype=np.int64),
        np.array(item_list, dtype=np.int64),
        np.array(store_list, dtype=np.int64),
        np.array(state_list, dtype=np.int64),
    )


# -----------------------------------------------------------------------------
# Testfunktionen: LSTM Many-to-One / Seq-to-Seq
# -----------------------------------------------------------------------------
def test_many_to_one_lstm(df: pd.DataFrame, train_module, ckpt_path: Path, run_out_dir: Path) -> dict:
    df_coded, series_to_idx, idx_to_series, item_ids, store_ids, state_ids = add_category_codes(df)
    feature_cols = train_module.build_feature_columns()

    hparams = infer_lstm_hparams_from_run(
        ckpt_path,
        fallback={
            "hidden_size": int(getattr(train_module, "HIDDEN_SIZE", 128)),
            "num_layers": int(getattr(train_module, "LAYER", 2)),
            "dropout": float(getattr(train_module, "DROPOUT", 0.1)),
            "batch_size": int(getattr(train_module, "BATCH_SIZE", BATCH_SIZE_DEFAULT)),
        },
    )

    # Checkpoint laden und Architekturparameter direkt aus den gespeicherten Gewichten ableiten.
    # Dadurch passt die rekonstruierte Modellarchitektur auch dann, wenn die Config-Datei
    # nicht alle Optuna-/Final-Run-Parameter enthält.
    state = torch_load_state_dict_safe(ckpt_path)
    hparams = infer_lstm_hparams_from_state_dict(state, hparams)

    # Many-to-One nutzt globale LAYER/DROPOUT in __init__.
    original_layer = getattr(train_module, "LAYER", None)
    original_dropout = getattr(train_module, "DROPOUT", None)
    train_module.LAYER = int(hparams["num_layers"])
    train_module.DROPOUT = float(hparams["dropout"])

    log_step('Many-to-One: Test-Windows werden gebaut')
    x, y_log, series_idx, item_idx, store_idx, state_idx = build_many_to_one_test_windows(df_coded, feature_cols, series_to_idx)
    log_step(f'Many-to-One: {len(y_log)} Test-Samples erzeugt')
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_log, dtype=torch.float32),
        torch.tensor(series_idx, dtype=torch.long),
        torch.tensor(item_idx, dtype=torch.long),
        torch.tensor(store_idx, dtype=torch.long),
        torch.tensor(state_idx, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=int(hparams["batch_size"]), shuffle=False, num_workers=NUM_WORKERS)

    model = train_module.Many_to_One_LSTM(
        n_features=x.shape[-1],
        hidden_size=int(hparams["hidden_size"]),
        horizon=HORIZON,
        num_items=len(item_ids),
        num_stores=len(store_ids),
        num_states=len(state_ids),
    ).to(DEVICE)

    model.load_state_dict(state)
    model.eval()

    if original_layer is not None:
        train_module.LAYER = original_layer
    if original_dropout is not None:
        train_module.DROPOUT = original_dropout

    pred_logs, true_logs, series_ids_all = [], [], []
    with torch.no_grad():
        for xb, yb, sidb, ib, sb, stb in loader:
            xb = xb.to(DEVICE)
            ib = ib.to(DEVICE)
            sb = sb.to(DEVICE)
            stb = stb.to(DEVICE)
            pred_log = model(xb, ib, sb, stb).detach().cpu().numpy()
            pred_logs.append(pred_log)
            true_logs.append(yb.numpy())
            series_ids_all.extend([idx_to_series[int(i)] for i in sidb.numpy().tolist()])

    pred_log = np.concatenate(pred_logs, axis=0)
    true_log = np.concatenate(true_logs, axis=0)
    series_ids_np = np.asarray(series_ids_all, dtype=object)
    pred_y = np.expm1(pred_log).clip(min=0.0)
    true_y = np.expm1(true_log).clip(min=0.0)

    mase_denoms = compute_mase_denominators(df[df["split"] == "train"].copy(), seasonality=7)
    metrics = eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, series_ids_np, mase_denoms)
    metrics.update(save_horizon_metrics(
        run_out_dir=run_out_dir,
        model_name="many_to_one_lstm",
        pred_y=pred_y,
        true_y=true_y,
        series_ids=series_ids_np,
        mase_denoms=mase_denoms,
    ))
    metrics["loss_logspace_mse"] = eval_loss_logspace_from_arrays(pred_log, true_log)
    metrics["n_test_samples"] = int(pred_y.shape[0])
    metrics["checkpoint"] = str(ckpt_path)

    run_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_out_dir / "test_summary.json", metrics)
    if SAVE_DETAILED_PREDICTIONS:
        save_predictions(run_out_dir / "test_predictions.csv", pred_y, true_y, series_ids_np)
    return metrics


def test_seq_to_seq_lstm(df: pd.DataFrame, train_module, ckpt_path: Path, run_out_dir: Path) -> dict:
    df_coded, series_to_idx, idx_to_series, item_ids, store_ids, state_ids = add_category_codes(df)
    enc_cols, dec_cols = train_module.build_feature_columns()

    hparams = infer_lstm_hparams_from_run(
        ckpt_path,
        fallback={
            "hidden_size": int(getattr(train_module, "HIDDEN_SIZE", 128)),
            "num_layers": int(getattr(train_module, "LAYER", 2)),
            "dropout": float(getattr(train_module, "DROPOUT", 0.1)),
            "batch_size": int(getattr(train_module, "BATCH_SIZE", BATCH_SIZE_DEFAULT)),
        },
    )

    # Checkpoint laden und Architekturparameter direkt aus den gespeicherten Gewichten ableiten.
    state = torch_load_state_dict_safe(ckpt_path)
    hparams = infer_lstm_hparams_from_state_dict(state, hparams)

    log_step('Seq-to-Seq: Test-Windows werden gebaut')
    x_enc, x_dec, y_log, series_idx, item_idx, store_idx, state_idx = build_seq_to_seq_test_windows(df_coded, enc_cols, dec_cols, series_to_idx)
    log_step(f'Seq-to-Seq: {len(y_log)} Test-Samples erzeugt')
    ds = TensorDataset(
        torch.tensor(x_enc, dtype=torch.float32),
        torch.tensor(x_dec, dtype=torch.float32),
        torch.tensor(y_log, dtype=torch.float32),
        torch.tensor(series_idx, dtype=torch.long),
        torch.tensor(item_idx, dtype=torch.long),
        torch.tensor(store_idx, dtype=torch.long),
        torch.tensor(state_idx, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=int(hparams["batch_size"]), shuffle=False, num_workers=NUM_WORKERS)

    model = train_module.Seq_to_Seq_LSTM(
        enc_features_num=len(enc_cols),
        dec_features_num=len(dec_cols),
        hidden_size=int(hparams["hidden_size"]),
        horizon=HORIZON,
        num_items=len(item_ids),
        num_stores=len(store_ids),
        num_states=len(state_ids),
        num_layers=int(hparams["num_layers"]),
        dropout=float(hparams["dropout"]),
    ).to(DEVICE)

    model.load_state_dict(state)
    model.eval()

    pred_logs, true_logs, series_ids_all = [], [], []
    with torch.no_grad():
        for xeb, xdb, yb, sidb, ib, sb, stb in loader:
            xeb = xeb.to(DEVICE)
            xdb = xdb.to(DEVICE)
            ib = ib.to(DEVICE)
            sb = sb.to(DEVICE)
            stb = stb.to(DEVICE)
            pred_log = model(xeb, xdb, ib, sb, stb).detach().cpu().numpy()
            pred_logs.append(pred_log)
            true_logs.append(yb.numpy())
            series_ids_all.extend([idx_to_series[int(i)] for i in sidb.numpy().tolist()])

    pred_log = np.concatenate(pred_logs, axis=0)
    true_log = np.concatenate(true_logs, axis=0)
    series_ids_np = np.asarray(series_ids_all, dtype=object)
    pred_y = np.expm1(pred_log).clip(min=0.0)
    true_y = np.expm1(true_log).clip(min=0.0)

    mase_denoms = compute_mase_denominators(df[df["split"] == "train"].copy(), seasonality=7)
    metrics = eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, series_ids_np, mase_denoms)
    metrics.update(save_horizon_metrics(
        run_out_dir=run_out_dir,
        model_name="seq_to_seq_lstm",
        pred_y=pred_y,
        true_y=true_y,
        series_ids=series_ids_np,
        mase_denoms=mase_denoms,
    ))
    metrics["loss_logspace_mse"] = eval_loss_logspace_from_arrays(pred_log, true_log)
    metrics["n_test_samples"] = int(pred_y.shape[0])
    metrics["checkpoint"] = str(ckpt_path)

    run_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_out_dir / "test_summary.json", metrics)
    if SAVE_DETAILED_PREDICTIONS:
        save_predictions(run_out_dir / "test_predictions.csv", pred_y, true_y, series_ids_np)
    return metrics


# -----------------------------------------------------------------------------
# TFT / PatchTST: Test-Dataset analog zur Trainingslogik bauen
# -----------------------------------------------------------------------------
def build_pf_train_test_datasets(df: pd.DataFrame, train_module):
    df = df.copy()
    df["series_id"] = df["series_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    df["state_id"] = df["state_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    max_series = getattr(train_module, "MAX_SERIES", None)
    if max_series is not None:
        allowed_series = sorted(df["series_id"].unique())[: int(max_series)]
        df = df[df["series_id"].isin(allowed_series)].copy()

    known_reals, unknown_reals = train_module.build_feature_columns()
    train_cutoff = int(df.loc[df["split"] == "train", "time_idx"].max())
    val_cutoff = int(df.loc[df["split"] == "val", "time_idx"].max())
    test_cutoff = int(df.loc[df["split"] == "test", "time_idx"].max())

    training = train_module.TimeSeriesDataSet(
        df[df.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="y_log",
        group_ids=["series_id"],
        static_categoricals=train_module.STATIC_CATEGORICALS,
        min_encoder_length=int(getattr(train_module, "ENCODER_LEN", SEQ_LEN)),
        max_encoder_length=int(getattr(train_module, "ENCODER_LEN", SEQ_LEN)),
        min_prediction_length=int(getattr(train_module, "PRED_LEN", HORIZON)),
        max_prediction_length=int(getattr(train_module, "PRED_LEN", HORIZON)),
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=train_module.GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    test = train_module.TimeSeriesDataSet.from_dataset(
        training,
        df[df.time_idx <= test_cutoff],
        predict=True,
        stop_randomization=True,
        min_prediction_idx=val_cutoff + 1,
    )
    return training, test, df


def test_tft(df: pd.DataFrame, train_module, ckpt_path: Path, run_out_dir: Path) -> dict:
    df = train_module.add_time_series_features(df)
    train_df = df[df["split"] == "train"].copy()
    mase_denoms = train_module.compute_mase_denominators(train_df, seasonality=7)

    training_ds, test_ds, _ = build_pf_train_test_datasets(df, train_module)
    series_mapping = train_module.get_series_id_mapping(training_ds)
    batch_size = int(getattr(train_module, "BATCH_SIZE", BATCH_SIZE_DEFAULT))

    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=(TORCH_DEVICE == "cuda"),
    )

    model = train_module.TFT_Model.load_from_checkpoint(
        str(ckpt_path),
        mase_denoms=mase_denoms,
        series_mapping=series_mapping,
    )
    model.to(TORCH_DEVICE)
    model.eval()

    raw = model.predict(test_loader, mode="raw", return_x=True)
    pred_log = train_module.extract_point_forecast(raw.output.prediction.detach().cpu().numpy())
    true_log = raw.x["decoder_target"].detach().cpu().numpy()
    series_ids = train_module.extract_series_ids_from_raw_x(raw.x, series_mapping)

    pred_y = np.expm1(pred_log).clip(min=0.0)
    true_y = np.expm1(true_log).clip(min=0.0)

    metrics = eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, series_ids, mase_denoms)
    metrics.update(save_horizon_metrics(
        run_out_dir=run_out_dir,
        model_name="tft",
        pred_y=pred_y,
        true_y=true_y,
        series_ids=series_ids,
        mase_denoms=mase_denoms,
    ))
    metrics["loss_logspace_mse"] = eval_loss_logspace_from_arrays(pred_log, true_log)
    metrics["n_test_samples"] = int(pred_y.shape[0])
    metrics["checkpoint"] = str(ckpt_path)

    run_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_out_dir / "test_summary.json", metrics)
    if SAVE_DETAILED_PREDICTIONS:
        save_predictions(run_out_dir / "test_predictions.csv", pred_y, true_y, series_ids)
    return metrics


def test_patchtst(df: pd.DataFrame, train_module, ckpt_path: Path, run_out_dir: Path) -> dict:
    df = train_module.add_time_series_features(df)
    train_df = df[df["split"] == "train"].copy()
    mase_denoms = train_module.compute_mase_denominators(train_df, seasonality=7)

    training_ds, test_ds, _ = build_pf_train_test_datasets(df, train_module)
    series_mapping = train_module.get_series_id_mapping(training_ds)
    batch_size = int(getattr(train_module, "BATCH_SIZE", BATCH_SIZE_DEFAULT))
    feature_names = list(training_ds.reals)
    num_series = len(series_mapping) if series_mapping is not None else int(df["series_id"].nunique())

    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=(TORCH_DEVICE == "cuda"),
    )

    sample_x, _ = next(iter(test_loader))
    input_dim = int(sample_x["encoder_cont"].shape[-1])

    model = train_module.PatchTSTModel.load_from_checkpoint(
        str(ckpt_path),
        input_dim=input_dim,
        horizon=int(getattr(train_module, "PRED_LEN", HORIZON)),
        num_series=num_series,
        learning_rate=float(getattr(train_module, "LR", 5e-3)),
        d_model=int(getattr(train_module, "D_MODEL", 32)),
        attention_head_size=int(getattr(train_module, "ATTN_HEAD_SIZE", 4)),
        hidden_continuous_size=int(getattr(train_module, "HIDDEN_CONT_SIZE", 16)),
        dropout=float(getattr(train_module, "DROPOUT", 0.1)),
        patch_len=int(getattr(train_module, "PATCH_LEN", 16)),
        patch_stride=int(getattr(train_module, "PATCH_STRIDE", 8)),
        num_transformer_layers=int(getattr(train_module, "NUM_TRANSFORMER_LAYERS", 3)),
        series_emb_dim=int(getattr(train_module, "SERIES_EMB_DIM", 16)),
        mase_denoms=mase_denoms,
        series_mapping=series_mapping,
        feature_names=feature_names,
    )
    model.to(TORCH_DEVICE)
    model.eval()

    output = train_module.collect_predictions(model, test_loader, series_mapping, model.device)
    pred_log = output["prediction_log"]
    true_log = output["true_log"]
    series_ids = output["series_ids"]
    pred_y = np.expm1(pred_log).clip(min=0.0)
    true_y = np.expm1(true_log).clip(min=0.0)

    metrics = eval_mase_mse_wape_weekly_from_arrays(pred_y, true_y, series_ids, mase_denoms)
    metrics.update(save_horizon_metrics(
        run_out_dir=run_out_dir,
        model_name="patchtst",
        pred_y=pred_y,
        true_y=true_y,
        series_ids=series_ids,
        mase_denoms=mase_denoms,
    ))
    metrics["loss_logspace_mse"] = eval_loss_logspace_from_arrays(pred_log, true_log)
    metrics["n_test_samples"] = int(pred_y.shape[0])
    metrics["checkpoint"] = str(ckpt_path)

    run_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_out_dir / "test_summary.json", metrics)
    if SAVE_DETAILED_PREDICTIONS:
        save_predictions(run_out_dir / "test_predictions.csv", pred_y, true_y, series_ids)
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def require_checkpoint(model_name: str, checkpoint_path: Path) -> Path:
    """
    Prüft, ob der oben fest eingetragene Checkpoint existiert.

    Es gibt keine CLI-Argumente und keine automatische Suche mehr.
    Dadurch ist das Script einfacher, aber die Pfade im Kopf der Datei müssen stimmen.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint für {model_name} nicht gefunden: {checkpoint_path}\n"
            f"Bitte den Pfad oben in der Datei anpassen."
        )
    return checkpoint_path


def main() -> None:
    """
    Einstiegspunkt des Test-Scripts.

    Ablauf:
        1. Vorverarbeiteten Datensatz aus CSV_PATH laden.
        2. Prüfen, ob ein Test-Split vorhanden ist.
        3. Für jedes Modell in MODELS_TO_TEST:
            - Trainingsdatei als Python-Modul laden.
            - Fest eingetragenen Checkpoint laden.
            - Test-Windows bzw. Test-Dataloader bauen.
            - Vorhersagen berechnen.
            - MASE, WAPE, MSE und Wochenmetriken berechnen.
        4. Alle Ergebnisse als CSV und JSON speichern.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_DIR / f"{timestamp}_test_horizon_28d"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_step('Lade vorverarbeiteten Datensatz')
    df = load_preprocessed(CSV_PATH)
    log_step(f'Datensatz geladen: {len(df):,} Zeilen, {df["series_id"].nunique():,} Serien')
    if "test" not in set(df["split"].astype(str).unique()):
        raise RuntimeError("Der Datensatz enthält keinen split == 'test'.")

    results: dict[str, Any] = {}
    rows = []

    if "many_to_one" in MODELS_TO_TEST:
        log_step("Starte Many-to-One LSTM")
        log_step(f"Many-to-One Trainingsdatei: {MANY_TO_ONE_TRAIN_PATH}")
        log_step(f"Many-to-One Checkpoint: {MANY_TO_ONE_CKPT_PATH}")
        module = load_module(MANY_TO_ONE_TRAIN_PATH, "train_many_to_one")
        ckpt = require_checkpoint("many_to_one", MANY_TO_ONE_CKPT_PATH)
        metrics = test_many_to_one_lstm(df, module, ckpt, out_dir / "many_to_one_lstm")
        results["many_to_one_lstm"] = metrics
        rows.append({"model": "many_to_one_lstm", **metrics})

    if "seq_to_seq" in MODELS_TO_TEST:
        log_step("Starte Seq-to-Seq LSTM")
        log_step(f"Seq-to-Seq Trainingsdatei: {SEQ_TO_SEQ_TRAIN_PATH}")
        log_step(f"Seq-to-Seq Checkpoint: {SEQ_TO_SEQ_CKPT_PATH}")
        module = load_module(SEQ_TO_SEQ_TRAIN_PATH, "train_seq_to_seq")
        ckpt = require_checkpoint("seq_to_seq", SEQ_TO_SEQ_CKPT_PATH)
        metrics = test_seq_to_seq_lstm(df, module, ckpt, out_dir / "seq_to_seq_lstm")
        results["seq_to_seq_lstm"] = metrics
        rows.append({"model": "seq_to_seq_lstm", **metrics})

    if "tft" in MODELS_TO_TEST:
        log_step("Starte TFT")
        log_step(f"TFT Trainingsdatei: {TFT_TRAIN_PATH}")
        log_step(f"TFT Checkpoint: {TFT_CKPT_PATH}")
        module = load_module(TFT_TRAIN_PATH, "train_tft")
        ckpt = require_checkpoint("tft", TFT_CKPT_PATH)
        metrics = test_tft(df, module, ckpt, out_dir / "tft")
        results["tft"] = metrics
        rows.append({"model": "tft", **metrics})

    if "patchtst" in MODELS_TO_TEST:
        log_step("Starte PatchTST")
        log_step(f"PatchTST Trainingsdatei: {PATCHTST_TRAIN_PATH}")
        log_step(f"PatchTST Checkpoint: {PATCHTST_CKPT_PATH}")
        module = load_module(PATCHTST_TRAIN_PATH, "train_patchtst")
        ckpt = require_checkpoint("patchtst", PATCHTST_CKPT_PATH)
        metrics = test_patchtst(df, module, ckpt, out_dir / "patchtst")
        results["patchtst"] = metrics
        rows.append({"model": "patchtst", **metrics})

    summary = {
        "created_at": timestamp,
        "device": DEVICE,
        "csv_path": str(CSV_PATH),
        "horizon": HORIZON,
        "seq_len": SEQ_LEN,
        "models_to_test": MODELS_TO_TEST,
        "results": results,
    }
    save_json(out_dir / "test_summary_all_models.json", summary)

    # Zusätzlich modellübergreifende CSV und Plots für MASE/WAPE je Horizonttag speichern.
    save_combined_horizon_metrics(out_dir, results)

    summary_df = pd.DataFrame(rows)
    preferred_cols = [
        "model",
        "mase", "mase_w1", "mase_w2", "mase_w3", "mase_w4",
        "wape", "wape_w1", "wape_w2", "wape_w3", "wape_w4",
        "mse", "loss_logspace_mse", "n_test_samples", "checkpoint",
    ]

    # Tagesmetriken für die grafische Horizont-Auswertung.
    preferred_cols += [f"mase_day_{day}" for day in range(1, HORIZON + 1)]
    preferred_cols += [f"wape_day_{day}" for day in range(1, HORIZON + 1)]

    existing = [c for c in preferred_cols if c in summary_df.columns]
    remaining = [c for c in summary_df.columns if c not in existing]
    summary_df = summary_df[existing + remaining]

    summary_df.to_csv(out_dir / "test_summary_all_models.csv", index=False)

    print("Saved:", out_dir / "test_summary_all_models.csv")
    print("Saved:", out_dir / "test_summary_all_models.json")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
