
# Test-Script für die trainierten Forecasting-Modelle.

# Getestet werden:
# - Many-to-One LSTM
# - Seq-to-Seq LSTM
# - TFT
# - PatchTST

# Die gespeicherten Modelle werdeb geladen und auf dem Test-Split bewertet.


# Importieren der Bibliotheken
from pathlib import Path
import importlib.util
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------
# Konfigurationen
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = Path("data/preprocessed/m5_long.csv")
OUTPUT_DIR = Path("runs/test")

SEQ_LEN = 56
HORIZON = 28
NUM_WORKERS = 6
SAVE_DETAILED_PREDICTIONS = True

# Welche Modelle sollen getestet werden?
MODELS_TO_TEST = [
    "many_to_one",
    "seq_to_seq",
    "tft",
    "patchtst",
]

# ------------------------------------------------------------
# Pfade zu Trainingsdateien und den trainierten Modellen
# ------------------------------------------------------------

MANY_TO_ONE_TRAIN_FILE = Path("Code/train_lstm_many_to_one.py")
SEQ_TO_SEQ_TRAIN_FILE = Path("Code/train_lstm_seq_to_seq.py")
TFT_TRAIN_FILE = Path("Code/train_TFT.py")
PATCHTST_TRAIN_FILE = Path("Code/train_PatchTST.py")

MANY_TO_ONE_MODEL_FILE = Path("Trained_Models/best_model_Many_to_One.pt")
SEQ_TO_SEQ_MODEL_FILE = Path("Trained_Models/best_model_Seq_to_Seq.pt")
TFT_MODEL_FILE = Path("Trained_Models/best_model_TFT.ckpt")
PATCHTST_MODEL_FILE = Path("Trained_Models/best_model_PatchTST.ckpt")



# ------------------------------------------------------------
# Fest eingetragene Modellparameter -> diese Werte stammen aus den Konfigurationen des finalen Tranings
# ------------------------------------------------------------

# Many-to-One LSTM
MANY_TO_ONE_HIDDEN_SIZE = 256
MANY_TO_ONE_NUM_LAYERS = 3
MANY_TO_ONE_DROPOUT = 0.1
MANY_TO_ONE_BATCH_SIZE = 1024

# Seq-to-Seq LSTM
SEQ_TO_SEQ_HIDDEN_SIZE = 128
SEQ_TO_SEQ_NUM_LAYERS = 2
SEQ_TO_SEQ_DROPOUT = 0.1
SEQ_TO_SEQ_BATCH_SIZE = 1024

# TFT
TFT_BATCH_SIZE = 1024

# PatchTST
PATCHTST_BATCH_SIZE = 1024
PATCHTST_D_MODEL = 128
PATCHTST_ATTENTION_HEAD_SIZE = 2
PATCHTST_HIDDEN_CONT_SIZE = 16
PATCHTST_DROPOUT = 0.10827211824776498
PATCHTST_PATCH_LEN = 16
PATCHTST_PATCH_STRIDE = 8
PATCHTST_NUM_TRANSFORMER_LAYERS = 3
PATCHTST_SERIES_EMB_DIM = 16
PATCHTST_LR = 0.0006559644071632867


# ------------------------------------------------------------
# Sammlung an Hilfsfunktionen
# ------------------------------------------------------------

# Gebe beim Start des Testings des Modells die Zeit mit aus.
def print_step(text: str):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

# Laden des Traningsskriptes des jeweiligen Modells
def load_python_file(file_path: Path, module_name: str):
    if not file_path.exists():
        raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Laden des m5_long.CSV Datensatzes
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Datensatz nicht gefunden: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

# Prüfe ob das Modellfile an der richtigen Stelle liegt
def check_model_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {path}")

# Lade die Gewichte der Neuronen aus dem File
def load_torch_weights(path: Path):
    return torch.load(path, map_location=DEVICE)


# ------------------------------------------------------------
# Funktionen zur Berechnung der Metriken
# ------------------------------------------------------------
# Berechnung des MASE Nenners (MAE) - Saisionale Prognose mit 7 Tagen
def calculate_mase_denominators(train_df: pd.DataFrame, seasonality: int = 7) -> dict:
    denominators = {}

    for series_id, group in train_df.groupby("series_id"):
        group = group.sort_values("time_idx")
        y_values = group["y"].to_numpy(dtype=np.float32)

        if len(y_values) <= seasonality:
            denominators[str(series_id)] = 1.0
            continue

        differences = np.abs(y_values[seasonality:] - y_values[:-seasonality])
        denominator = float(np.mean(differences))

        # Falls MAE null sein sollte, wird es auf 1 erhöht, damit nicht durch null geteilt wird
        if denominator <= 0:
            denominator = 1.0

        denominators[str(series_id)] = denominator

    return denominators

# Berechnung des MASE Nenners (MAE) - Saisionale Prognose mit 7 Tagen 
def calculate_metrics(prediction: np.ndarray, actual: np.ndarray, series_ids: np.ndarray, mase_denominators: dict) -> dict:
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, None) # np.clip begrenzt hier alle Werte auf 0. Werte kleiner null werden zu null. None = mach oben keine Grenze. Der Grund dafür liegt darin, dass die Werte wieder in den Orignialraum umgeformt werden. Dabei kann es passieren, dass die Werte leicht negativ sind.
    actual = np.clip(np.asarray(actual, dtype=np.float32), 0.0, None)
    series_ids = np.asarray(series_ids, dtype=object)

    absolute_error = np.abs(prediction - actual)
    squared_error = (prediction - actual) ** 2

    denominator_values = np.array(
        [float(mase_denominators.get(str(series_id), 1.0)) for series_id in series_ids],
        dtype=np.float32,
    )
    denominator_values = np.where(denominator_values > 0, denominator_values, 1.0)
   
    # Gesamtwerte über alle Tage berechnen
    mase = float(np.mean(np.mean(absolute_error, axis=1) / denominator_values))
    mse = float(np.mean(squared_error))

    wape_denominator = float(np.sum(actual))
    wape = float(np.sum(absolute_error) / wape_denominator) if wape_denominator > 0 else float("nan")

    result = {
        "mase": mase,
        "mse": mse,
        "wape": wape,
    }

    # Wochenmetriken für 28 Tage Prognosehorizont
    week_ranges = {
        "w1": (0, 7),
        "w2": (7, 14),
        "w3": (14, 21),
        "w4": (21, 28),
    }

    # Berechnungen der Metriken MASE und WAPE pro Woche
    for week_name, (start, end) in week_ranges.items():
        week_error = absolute_error[:, start:end]
        week_actual = actual[:, start:end]

        result[f"mase_{week_name}"] = float(np.mean(np.mean(week_error, axis=1) / denominator_values))

        week_wape_denominator = float(np.sum(week_actual))
        if week_wape_denominator > 0:
            result[f"wape_{week_name}"] = float(np.sum(week_error) / week_wape_denominator)
        else:
            result[f"wape_{week_name}"] = float("nan")

    return result

# Berechnung der täglichen Metrikwerte
def calculate_day_metrics(prediction: np.ndarray, actual: np.ndarray, series_ids: np.ndarray, mase_denominators: dict):
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, None)  # np.clip begrenzt hier alle Werte auf 0. Werte kleiner null werden zu null. None = mach oben keine Grenze
    actual = np.clip(np.asarray(actual, dtype=np.float32), 0.0, None)

    denominator_values = np.array(
        [float(mase_denominators.get(str(series_id), 1.0)) for series_id in series_ids],
        dtype=np.float32,
    )
    denominator_values = np.where(denominator_values > 0, denominator_values, 1.0)

    rows = []

    # Für jeden Tag des Prognosehorizontes die Metriken ermitteln
    for day in range(HORIZON):
        absolute_error = np.abs(prediction[:, day] - actual[:, day])
        actual_sum = float(np.sum(actual[:, day]))

        rows.append({
            "horizon_day": day + 1,
            "mase": float(np.mean(absolute_error / denominator_values)),
            "wape": float(np.sum(absolute_error) / actual_sum) if actual_sum > 0 else float("nan"),
        })

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Funktion zum Speichern der Modellergebnisse und erzeugen von Plots, die die Performance der Modelle vergleichen
# ------------------------------------------------------------
def save_model_results(output_dir: Path, model_name: str, prediction: np.ndarray, actual: np.ndarray, series_ids: np.ndarray, mase_denominators: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Berechnen der Metriken
    metrics = calculate_metrics(prediction, actual, series_ids, mase_denominators)
    horizon_df = calculate_day_metrics(prediction, actual, series_ids, mase_denominators)
    horizon_df.insert(0, "model", model_name) # An erster Stelle den Modellnamen einfügen
    horizon_df.to_csv(output_dir / "horizon_metrics.csv", index=False)

    # Abbildung erstellen
    plt.figure(figsize=(10, 5))
    plt.plot(horizon_df["horizon_day"], horizon_df["mase"], marker="o", label="MASE")
    plt.plot(horizon_df["horizon_day"], horizon_df["wape"], marker="o", label="WAPE")
    plt.xlabel("Forecast-Tag")
    plt.ylabel("Metrikwert")
    plt.title(f"MASE und WAPE je Forecast-Tag - {model_name}")
    plt.xticks(range(1, HORIZON + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "horizon_metrics.png", dpi=150)
    plt.close()

    for _, row in horizon_df.iterrows():
        day = int(row["horizon_day"])
        metrics[f"mase_day_{day}"] = float(row["mase"])
        metrics[f"wape_day_{day}"] = float(row["wape"])

    return metrics

# ------------------------------------------------------------
# Datenvorbereitung für die LSTM-Modelle
# ------------------------------------------------------------
def add_integer_codes(df: pd.DataFrame):
    # LSTMs brauchen numerische IDs für Embeddings – hier werden
    # die String-IDs z.B  CA_1, FOODS_1) in fortlaufende Integers umgewandelt.
    # Zurückgegeben werden auch die Mapping-Dicts um später wieder
    # von Zahl auf ID zurückrechnen zu können.
    df = df.copy()

    series_ids = sorted(df["series_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    store_ids = sorted(df["store_id"].unique())
    state_ids = sorted(df["state_id"].unique())

    series_to_number = {value: number for number, value in enumerate(series_ids)}
    number_to_series = {number: value for value, number in series_to_number.items()}

    df["item_id_code"] = df["item_id"].map({value: number for number, value in enumerate(item_ids)}).astype(np.int64)
    df["store_id_code"] = df["store_id"].map({value: number for number, value in enumerate(store_ids)}).astype(np.int64)
    df["state_id_code"] = df["state_id"].map({value: number for number, value in enumerate(state_ids)}).astype(np.int64)

    return df, series_to_number, number_to_series, item_ids, store_ids, state_ids

# Funktion zum erzeugen des Test Windowns für Many-To-One
def build_many_to_one_window(df: pd.DataFrame, feature_columns: list[str], series_to_number: dict):
    
    # Baut für jede Serie genau ein Test-Fenster
    # Input:  die 56 Tage vor dem Test-Split (Encoder)
    # Target: die 28 Test-Tage Prognosezeitraum im Testsplit
    
    all_features = []
    all_targets = []
    all_series = []
    all_items = []
    all_stores = []
    all_states = []

    for series_id, group in df.groupby("series_id"):
        group = group.sort_values("time_idx").reset_index(drop=True)

        features = group[feature_columns].to_numpy(dtype=np.float32)
        target = group["y_log"].to_numpy(dtype=np.float32)
        split = group["split"].to_numpy()

        # Ersten Tag des Test-Splits als Startpunkt des Forecast-Fensters nehmen
        test_positions = np.where(split == "test")[0]
        if len(test_positions) == 0:
            continue

        forecast_start = int(test_positions[0])

        # Absicherung: genug Historie für Encoder und genug Zukunft für Horizon
        if forecast_start < SEQ_LEN or forecast_start + HORIZON > len(group):
            continue

        all_features.append(features[forecast_start - SEQ_LEN:forecast_start])
        all_targets.append(target[forecast_start:forecast_start + HORIZON])
        all_series.append(series_to_number[series_id])
        all_items.append(int(group["item_id_code"].iloc[0]))
        all_stores.append(int(group["store_id_code"].iloc[0]))
        all_states.append(int(group["state_id_code"].iloc[0]))

    return (
        np.stack(all_features),
        np.stack(all_targets),
        np.array(all_series, dtype=np.int64),
        np.array(all_items, dtype=np.int64),
        np.array(all_stores, dtype=np.int64),
        np.array(all_states, dtype=np.int64),
    )


def build_seq_to_seq_window(df: pd.DataFrame, encoder_columns: list[str], decoder_columns: list[str], series_to_number: dict):
    # Wie build_many_to_one_window, aber mit getrennten Encoder- und Decoder-Features.
    # Encoder: historische Features der letzten 56 Tage
    # Decoder: bekannte zukünftige Features der nächsten 28 Tage (Preis, Kalender etc.)
    all_encoder_features = []
    all_decoder_features = []
    all_targets = []
    all_series = []
    all_items = []
    all_stores = []
    all_states = []

    for series_id, group in df.groupby("series_id"):
        group = group.sort_values("time_idx").reset_index(drop=True)

        encoder_features = group[encoder_columns].to_numpy(dtype=np.float32)
        decoder_features = group[decoder_columns].to_numpy(dtype=np.float32)
        target = group["y_log"].to_numpy(dtype=np.float32)
        split = group["split"].to_numpy()

        test_positions = np.where(split == "test")[0]
        if len(test_positions) == 0:
            continue

        forecast_start = int(test_positions[0])
        if forecast_start < SEQ_LEN or forecast_start + HORIZON > len(group):
            continue

        all_encoder_features.append(encoder_features[forecast_start - SEQ_LEN:forecast_start])
        all_decoder_features.append(decoder_features[forecast_start:forecast_start + HORIZON])
        all_targets.append(target[forecast_start:forecast_start + HORIZON])
        all_series.append(series_to_number[series_id])
        all_items.append(int(group["item_id_code"].iloc[0]))
        all_stores.append(int(group["store_id_code"].iloc[0]))
        all_states.append(int(group["state_id_code"].iloc[0]))

    return (
        np.stack(all_encoder_features),
        np.stack(all_decoder_features),
        np.stack(all_targets),
        np.array(all_series, dtype=np.int64),
        np.array(all_items, dtype=np.int64),
        np.array(all_stores, dtype=np.int64),
        np.array(all_states, dtype=np.int64),
    )


# ------------------------------------------------------------
# LSTM-Modelle testen
# ------------------------------------------------------------

# Als erstes wird das Many-to-One getestet
def test_many_to_one_lstm(df: pd.DataFrame, output_dir: Path) -> dict:
    print_step("Teste Many-to-One LSTM")
    check_model_file(MANY_TO_ONE_MODEL_FILE)

    # Trainingsskript laden um Modellarchitektur und Features zu bekommen
    train_file = load_python_file(MANY_TO_ONE_TRAIN_FILE, "many_to_one_train_file")
    df, series_to_number, number_to_series, item_ids, store_ids, state_ids = add_integer_codes(df)
    feature_columns = train_file.build_feature_columns() # build_feature_columns() kommt aus dem Trainingsskript

    # Test-Fenster für alle Serien bauen
    x, y_log, series_numbers, item_numbers, store_numbers, state_numbers = build_many_to_one_window(
        df=df,
        feature_columns=feature_columns,
        series_to_number=series_to_number,
    )

    # PyTorch Dataset und DataLoader bauen
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_log, dtype=torch.float32),
        torch.tensor(series_numbers, dtype=torch.long),
        torch.tensor(item_numbers, dtype=torch.long),
        torch.tensor(store_numbers, dtype=torch.long),
        torch.tensor(state_numbers, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=MANY_TO_ONE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) # Shuffle false heißt, dass die Zeitschritte nicht random übergeben werden, sehr wichtig bei Zeitreihen!

    # Modell mit denselben Parametern wie beim Training initialisieren
    train_file.LAYER = MANY_TO_ONE_NUM_LAYERS
    train_file.DROPOUT = MANY_TO_ONE_DROPOUT

    model = train_file.Many_to_One_LSTM(
        n_features=x.shape[-1],
        hidden_size=MANY_TO_ONE_HIDDEN_SIZE,
        horizon=HORIZON,
        num_items=len(item_ids),
        num_stores=len(store_ids),
        num_states=len(state_ids),
    ).to(DEVICE)

    # Gespeicherte Gewichte laden
    model.load_state_dict(load_torch_weights(MANY_TO_ONE_MODEL_FILE))
    model.eval()

    prediction_log_list = []
    actual_log_list = []
    series_id_list = []

    # Batch für Batch durch das Modell schicken
    with torch.no_grad(): # Kein Gradient nötig beim Testen, da wir nicht trainieren
        for batch in loader:
            x_batch, y_batch, series_batch, item_batch, store_batch, state_batch = batch

            prediction_log = model(
                x_batch.to(DEVICE),
                item_batch.to(DEVICE),
                store_batch.to(DEVICE),
                state_batch.to(DEVICE),
            )

            prediction_log_list.append(prediction_log.cpu().numpy())
            actual_log_list.append(y_batch.numpy())
            # Nummer zurück zu Series-ID übersetzen
            series_id_list.extend([number_to_series[int(number)] for number in series_batch.numpy()])

    # Alle Batches zusammenführen
    prediction_log = np.concatenate(prediction_log_list)
    actual_log = np.concatenate(actual_log_list)

    # Log-Raum zurück in Original-Skala transformieren - expm1 ist Umkehrung von log1p
    prediction = np.expm1(prediction_log)
    actual = np.expm1(actual_log)
    series_ids = np.array(series_id_list, dtype=object)

    # Berechnen der Metriken und Speichern der Ergebnisse
    mase_denominators = calculate_mase_denominators(df[df["split"] == "train"])
    metrics = save_model_results(output_dir, "many_to_one_lstm", prediction, actual, series_ids, mase_denominators)
    
    metrics["loss_logspace_mse"] = float(np.mean((prediction_log - actual_log) ** 2))
    metrics["n_test_samples"] = int(prediction.shape[0])
    metrics["checkpoint"] = str(MANY_TO_ONE_MODEL_FILE)

    return metrics

# Nun Testen des Seq-to-Seq-LSTM-Modells
def test_seq_to_seq_lstm(df: pd.DataFrame, output_dir: Path) -> dict:
    # Der grundsätzliche Aufbau ist identisch mit dem Many-to-One
    # Besonderheit ist hier wieder die Encoder-Decoder-Struktur
    print_step("Teste Seq-to-Seq LSTM")
    check_model_file(SEQ_TO_SEQ_MODEL_FILE)

    train_file = load_python_file(SEQ_TO_SEQ_TRAIN_FILE, "seq_to_seq_train_file")
    df, series_to_number, number_to_series, item_ids, store_ids, state_ids = add_integer_codes(df)
    
    # Seq-to-Seq nutzt getrennte Encoder und Decoder Featurespalten
    encoder_columns, decoder_columns = train_file.build_feature_columns()

    x_encoder, x_decoder, y_log, series_numbers, item_numbers, store_numbers, state_numbers = build_seq_to_seq_window(
        df=df,
        encoder_columns=encoder_columns,
        decoder_columns=decoder_columns,
        series_to_number=series_to_number,
    )

    dataset = TensorDataset(
        torch.tensor(x_encoder, dtype=torch.float32),
        torch.tensor(x_decoder, dtype=torch.float32),
        torch.tensor(y_log, dtype=torch.float32),
        torch.tensor(series_numbers, dtype=torch.long),
        torch.tensor(item_numbers, dtype=torch.long),
        torch.tensor(store_numbers, dtype=torch.long),
        torch.tensor(state_numbers, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=SEQ_TO_SEQ_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = train_file.Seq_to_Seq_LSTM(
        enc_features_num=len(encoder_columns),
        dec_features_num=len(decoder_columns),
        hidden_size=SEQ_TO_SEQ_HIDDEN_SIZE,
        horizon=HORIZON,
        num_items=len(item_ids),
        num_stores=len(store_ids),
        num_states=len(state_ids),
        num_layers=SEQ_TO_SEQ_NUM_LAYERS,
        dropout=SEQ_TO_SEQ_DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(load_torch_weights(SEQ_TO_SEQ_MODEL_FILE))
    model.eval()

    prediction_log_list = []
    actual_log_list = []
    series_id_list = []

    with torch.no_grad():
        for batch in loader:
            x_encoder_batch, x_decoder_batch, y_batch, series_batch, item_batch, store_batch, state_batch = batch

            prediction_log = model(
                x_encoder_batch.to(DEVICE),
                x_decoder_batch.to(DEVICE),
                item_batch.to(DEVICE),
                store_batch.to(DEVICE),
                state_batch.to(DEVICE),
            )

            prediction_log_list.append(prediction_log.cpu().numpy())
            actual_log_list.append(y_batch.numpy())
            series_id_list.extend([number_to_series[int(number)] for number in series_batch.numpy()])

    prediction_log = np.concatenate(prediction_log_list)
    actual_log = np.concatenate(actual_log_list)

    prediction = np.expm1(prediction_log)
    actual = np.expm1(actual_log)
    series_ids = np.array(series_id_list, dtype=object)

    mase_denominators = calculate_mase_denominators(df[df["split"] == "train"])
    metrics = save_model_results(output_dir, "seq_to_seq_lstm", prediction, actual, series_ids, mase_denominators)

    metrics["loss_logspace_mse"] = float(np.mean((prediction_log - actual_log) ** 2))
    metrics["n_test_samples"] = int(prediction.shape[0])
    metrics["checkpoint"] = str(SEQ_TO_SEQ_MODEL_FILE)

    return metrics

# ------------------------------------------------------------
# Daten für die Transformer TFT und PatchTST vorbereiten
# ------------------------------------------------------------

def build_transformer_test_dataset(df: pd.DataFrame, train_file):
    # TFT und PatchTST nutzen das PyTorch Forecasting TimeSeriesDataSet, welches
    # ein spezialisiertes Dataset-Format ist das Encoder und Decoder automatisch verwaltet
    # Zuerst wird das Training-Dataset und dann das Test-Dataset abgeleitet


    df = df.copy()
    df["series_id"] = df["series_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    df["state_id"] = df["state_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    known_reals, unknown_reals = train_file.build_feature_columns()

    # Letzten Tag aus dem Split lesen
    train_end = int(df.loc[df["split"] == "train", "time_idx"].max())
    val_end = int(df.loc[df["split"] == "val", "time_idx"].max())
    test_end = int(df.loc[df["split"] == "test", "time_idx"].max())

    # Training-Dataset für Normalisierung
    # Hintergrund dieses Schrittes:
    # TFT und PatchTST nutzen zwei Normalisierungen: y_log und GroupNormalizer was automatisch durch das Framework passiert.
    # Der GroupNormalizer berechnet Mittelwert und Standardabweichung auf den
    # Trainingsdaten – from_dataset() stellt sicher dass beim Test exakt
    # dieselben Parameter verwendet werden, weil das Framework die
    # Rücktransformation intern übernimmt und sonst falsche Werte rauskämen.
    # Daher sind hier andere Schritte nötig als bei LSTM

    training_dataset = train_file.TimeSeriesDataSet(
        df[df["time_idx"] <= train_end],
        time_idx="time_idx",
        target="y_log",
        group_ids=["series_id"],
        static_categoricals=train_file.STATIC_CATEGORICALS,
        min_encoder_length=SEQ_LEN,
        max_encoder_length=SEQ_LEN,
        min_prediction_length=HORIZON,
        max_prediction_length=HORIZON,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=train_file.GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )

    # Test-Dataset von Training-Dataset ableiten damit Normalisierung identisch ist
    test_dataset = train_file.TimeSeriesDataSet.from_dataset(
        training_dataset,
        df[df["time_idx"] <= test_end],
        predict=True,
        stop_randomization=True,
        min_prediction_idx=val_end + 1, # Nur Test Zeitraum vorhersagen
    )

    return training_dataset, test_dataset


# ------------------------------------------------------------
# TFT und PatchTST testen
# ------------------------------------------------------------

# Test des TFT
def test_tft(df: pd.DataFrame, output_dir: Path) -> dict:
    print_step("Teste TFT")
    check_model_file(TFT_MODEL_FILE)

    train_file = load_python_file(TFT_TRAIN_FILE, "tft_train_file")
    df = train_file.add_time_series_features(df)

    training_dataset, test_dataset = build_transformer_test_dataset(df, train_file)
    series_mapping = train_file.get_series_id_mapping(training_dataset)
    mase_denominators = train_file.compute_mase_denominators(df[df["split"] == "train"], seasonality=7)

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=TFT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=(DEVICE == "cuda"),
    )

    # TFT aus Checkpoint laden – Modellparameter werden aus der .ckpt-Datei gelesen
    model = train_file.TFT_Model.load_from_checkpoint(
        str(TFT_MODEL_FILE),
        mase_denoms=mase_denominators,
        series_mapping=series_mapping,
    )
    model.to(DEVICE)
    model.eval()

    # TFT gibt einen rohen Output zurück aus dem  die Punktvorhersage ermittelt wird
    raw_prediction = model.predict(test_loader, mode="raw", return_x=True)

    prediction_log = train_file.extract_point_forecast(raw_prediction.output.prediction.cpu().numpy()) # Mit der Funktion aus dem Traningsskript wird der Punktforecast erstellt
    actual_log = raw_prediction.x["decoder_target"].cpu().numpy()
    series_ids = train_file.extract_series_ids_from_raw_x(raw_prediction.x, series_mapping)

    # Rücktransformation in den Originalraum
    prediction = np.expm1(prediction_log)
    actual = np.expm1(actual_log)

    metrics = save_model_results(output_dir, "tft", prediction, actual, series_ids, mase_denominators)
    metrics["loss_logspace_mse"] = float(np.mean((prediction_log - actual_log) ** 2))
    metrics["n_test_samples"] = int(prediction.shape[0])
    metrics["checkpoint"] = str(TFT_MODEL_FILE)

    return metrics

# Test des PatchTST
def test_patchtst(df: pd.DataFrame, output_dir: Path) -> dict:
    print_step("Teste PatchTST")
    check_model_file(PATCHTST_MODEL_FILE)

    train_file = load_python_file(PATCHTST_TRAIN_FILE, "patchtst_train_file")
    df = train_file.add_time_series_features(df)

    training_dataset, test_dataset = build_transformer_test_dataset(df, train_file)
    series_mapping = train_file.get_series_id_mapping(training_dataset)
    mase_denominators = train_file.compute_mase_denominators(df[df["split"] == "train"], seasonality=7)

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=PATCHTST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        pin_memory=(DEVICE == "cuda"),
    )

    # Input-Dimensionen aus den Daten lesen statt sich hart zu Coden
    sample_x, _ = next(iter(test_loader))
    input_dim = int(sample_x["encoder_cont"].shape[-1])
    feature_names = list(training_dataset.reals)
    num_series = len(series_mapping) if series_mapping is not None else int(df["series_id"].nunique())

    # PatchTST aus Checkpoint laden – alle Architektur-Parameter müssen exakt passen
    model = train_file.PatchTSTModel.load_from_checkpoint(
        str(PATCHTST_MODEL_FILE),
        input_dim=input_dim,
        horizon=HORIZON,
        num_series=num_series,
        #learning_rate=PATCHTST_LR, #LR wird beim Test nicht benötigt
        d_model=PATCHTST_D_MODEL,
        attention_head_size=PATCHTST_ATTENTION_HEAD_SIZE,
        hidden_continuous_size=PATCHTST_HIDDEN_CONT_SIZE,
        dropout=PATCHTST_DROPOUT,
        patch_len=PATCHTST_PATCH_LEN,
        patch_stride=PATCHTST_PATCH_STRIDE,
        num_transformer_layers=PATCHTST_NUM_TRANSFORMER_LAYERS,
        series_emb_dim=PATCHTST_SERIES_EMB_DIM,
        mase_denoms=mase_denominators,
        series_mapping=series_mapping,
        feature_names=feature_names,
    )
    model.to(DEVICE)
    model.eval()

    # PatchTST hat eine eigene collect_predictions-Funktion im Trainingsskript
    output = train_file.collect_predictions(model, test_loader, series_mapping, DEVICE)

    prediction_log = output["prediction_log"]
    actual_log = output["true_log"]
    series_ids = output["series_ids"]
    # Rücktransformation in den Originalraum
    prediction = np.expm1(prediction_log)
    actual = np.expm1(actual_log)

    metrics = save_model_results(output_dir, "patchtst", prediction, actual, series_ids, mase_denominators)
    metrics["loss_logspace_mse"] = float(np.mean((prediction_log - actual_log) ** 2))
    metrics["n_test_samples"] = int(prediction.shape[0])
    metrics["checkpoint"] = str(PATCHTST_MODEL_FILE)

    return metrics


# ------------------------------------------------------------
# Vergleichsplots über alle Modelle
# ------------------------------------------------------------

def save_overall_horizon_plots(output_dir: Path, all_results: dict) -> None:
    # Tagesmetriken aller Modelle in ein gemeinsames DF zusammenführen
    rows = []

    for model_name, metrics in all_results.items():
        for day in range(1, HORIZON + 1):
            rows.append({
                "model": model_name,
                "horizon_day": day,
                "mase": metrics.get(f"mase_day_{day}"),
                "wape": metrics.get(f"wape_day_{day}"),
            })

    horizon_df = pd.DataFrame(rows)
    horizon_df.to_csv(output_dir / "test_horizon_metrics_all_models.csv", index=False)

    # Plot 1: MASE pro Tag für alle Modelle im Vergleich
    plt.figure(figsize=(10, 5))
    for model_name, group in horizon_df.groupby("model"):
        plt.plot(group["horizon_day"], group["mase"], marker="o", label=model_name)
    plt.xlabel("Forecast-Tag")
    plt.ylabel("MASE")
    plt.title("MASE je Forecast-Tag")
    plt.xticks(range(1, HORIZON + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_horizon_mase_all_models.png", dpi=150)
    plt.close()

    # Plot 2: WAPE pro Tag für alle Modelle im Vergleich
    plt.figure(figsize=(10, 5))
    for model_name, group in horizon_df.groupby("model"):
        plt.plot(group["horizon_day"], group["wape"], marker="o", label=model_name)
    plt.xlabel("Forecast-Tag")
    plt.ylabel("WAPE")
    plt.title("WAPE je Forecast-Tag")
    plt.xticks(range(1, HORIZON + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_horizon_wape_all_models.png", dpi=150)
    plt.close()

# ------------------------------------------------------------
# Hautptfunktion des Skriptes: Alle Modelle nacheinander testen und Ergebnisse sammeln
# ------------------------------------------------------------
def main() -> None:
    print("START - Einfaches Test-Script wird ausgeführt", flush=True)
    
    # Ausgabe-Ordner mit Timestamp anlegen damit Runs nicht überschrieben werden
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_DIR / f"{timestamp}_simple_test_28d"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_step("Lade den Datensatz...")
    df = load_data()
    print_step(f"Datensatz geladen: {len(df):,} Zeilen, {df['series_id'].nunique():,} Serien")

    all_results = {} # Metriken pro Modell
    summary_rows = [] # Liste für die finale Vergleichs-CSV

    # Jedes Modell testen das in MODELS_TO_TEST steht
    if "many_to_one" in MODELS_TO_TEST:
        metrics = test_many_to_one_lstm(df, output_dir / "many_to_one_lstm")
        all_results["many_to_one_lstm"] = metrics
        summary_rows.append({"model": "many_to_one_lstm", **metrics}) # **metrics entpackt das Metriken-Dict in einzelne Spalten. Jede Metrik eine eigene Spalte

    if "seq_to_seq" in MODELS_TO_TEST:
        metrics = test_seq_to_seq_lstm(df, output_dir / "seq_to_seq_lstm")
        all_results["seq_to_seq_lstm"] = metrics
        summary_rows.append({"model": "seq_to_seq_lstm", **metrics})

    if "tft" in MODELS_TO_TEST:
        metrics = test_tft(df, output_dir / "tft")
        all_results["tft"] = metrics
        summary_rows.append({"model": "tft", **metrics})

    if "patchtst" in MODELS_TO_TEST:
        metrics = test_patchtst(df, output_dir / "patchtst")
        all_results["patchtst"] = metrics
        summary_rows.append({"model": "patchtst", **metrics})

    summary_df = pd.DataFrame(summary_rows)

    # Spaltenreihenfolge der Metrikausgabe festlegen
    first_columns = [
        "model",
        "mase", "mase_w1", "mase_w2", "mase_w3", "mase_w4",
        "wape", "wape_w1", "wape_w2", "wape_w3", "wape_w4",
        "mse", "loss_logspace_mse", "n_test_samples", "checkpoint",
    ]
    first_columns += [f"mase_day_{day}" for day in range(1, HORIZON + 1)]
    first_columns += [f"wape_day_{day}" for day in range(1, HORIZON + 1)]

    # Nur Spalten nehmen die wirklich existieren
    existing_first_columns = [column for column in first_columns if column in summary_df.columns]
    other_columns = [column for column in summary_df.columns if column not in existing_first_columns]
    summary_df = summary_df[existing_first_columns + other_columns]

    # Speichern der CSV und Diagramme
    summary_df.to_csv(output_dir / "test_summary_all_models.csv", index=False)
    save_overall_horizon_plots(output_dir, all_results)

    print("Gespeichert unter :", output_dir / "test_summary_all_models.csv")


if __name__ == "__main__":
    main()
