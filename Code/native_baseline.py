# Dieses Skript erstellt die Prognosen für die Naivprognose und die Ergebnisse der Modelle auch für den Validierungs-
# und Testsplit vergeleichbar zu machen.

import time
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Allgemeine Konfigurationen
# ---------------------------------------------------------------------
DATA_DIR = Path("data") / "preprocessed"
CSV_PATH = DATA_DIR / "m5_long.csv"

HORIZON = 28
SEASONALITY = 7
SPLIT_NAME = "val"   # "val" or "test"

RUNS_DIR = Path("runs")


# ---------------------------------------------------------------------
# Funktion zum Laden der Daten 
# ---------------------------------------------------------------------
def load_data():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)

# ---------------------------------------------------------------------
# Funktion zur Berechnung des MASE Nenners auf dem Traningssplit. Nenner MASE = MAE der saisonalen Naivprognose
# ---------------------------------------------------------------------
def compute_mase_denominators(df: pd.DataFrame, seasonality: int = 7) -> dict:
    
    # Dictionary vorbereiten welches später alle Nenner von MASE (MAE) pro Serie enthält
    denoms: dict[str, float] = {}

    # Jede Zeitreihe einzeln verarbeiten
    for series_id, series_df in df.groupby("series_id"):
        series_df = series_df.sort_values("time_idx")
        series_df_train = series_df[series_df["split"] == "train"] 

        # Nur die Trainingsdaten für den Nenner verwenden (basiert auf der allgemeinen MASE-Formel)
        y_train = series_df_train["y"].astype(np.float32).values
       
        # Wenn die Reihe zu kurz ist, kann kein 7-Vergleich berechnet werden
        # dann → NaN für die Serie speichern, Reihe wird später beim MASE übersprungen
        if len(y_train) <= seasonality:
            denoms[series_id] = np.nan
            continue

        # Absolute Unterschiede zwischen den aktuellen Werten y_train[seasonality:] und den Werten von vor 7 Tagen berechnen y_train[:-seasonality]
        absolute_differences = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        
        # Nenner (MAE) berechnen - dafür Durchschnitt aller Abweichungen pro Serie
        denom = float(np.mean(absolute_differences)) if len(absolute_differences) else np.nan

        # Sonderfall: Wenn eine Reihe immer 0 verkauft, ist der Nenner auch 0.
        # Division durch 0 geht nicht daher wird es bei inf auf Nan gesetzt
        if not np.isfinite(denom) or denom <= 0.0:
            denoms[series_id] = np.nan
        else:
            denoms[series_id] = denom

    # Rückgabe des Dictionarys mit allen MAE-Werten pro Serie
    return denoms


# ---------------------------------------------------------------------
# Wape Berechnung
# ---------------------------------------------------------------------
def calc_wape(num: float, den: float) -> float:
    # Diese Funktion wird von den seasonal_naive Funktionen aufgerufen und berechnet die WAPE Werte
    # Wenn es gar keine Verkäufe gab (den = 0), gibt es kein sinnvolles WAPE
    # dann wird NaN zurückgeben statt durch 0 zu teilen
    return float(num / den) if den > 0.0 else np.nan


def build_metrics_table(results: dict, weekly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    # Diese Funktion baut eine kompakte Tabelle mit zwei Spalten: Metrik und Wert 
    # Das ist bewusst flach gehalten, damit man das
    # Ergebnis direkt in Excel öffnen und weiterverwenden kann

    #Liste für alle Zeilen aufbauen
    rows = []

    # Gesamtmetriken über alle Serien und alle Fenster
    rows.append({"Metrik": "MASE", "Wert": results["mase"]})
    rows.append({"Metrik": "WAPE", "Wert": results["wape"]})
    rows.append({"Metrik": "MSE", "Wert": results["mse"]})

    # MASE pro Woche - Range 1 bis 5, also 4 Wochen (letzter Wert in Range wird nicht genutzt)
    for week in range(1, 5):
        val = weekly_df.loc[weekly_df["week"] == week, "mase"].iloc[0] 
        rows.append({"Metrik": f"MASE Woche {week}", "Wert": val})

    # WAPE pro Woche
    for week in range(1, 5):
        val = weekly_df.loc[weekly_df["week"] == week, "wape"].iloc[0]
        rows.append({"Metrik": f"WAPE Woche {week}", "Wert": val})

    # MASE pro Tag - Range 1 bis Horizon (28) +1, also 1-28
    for day in range(1, HORIZON + 1):
        val = daily_df.loc[daily_df["horizon"] == day, "mase"].iloc[0]
        rows.append({"Metrik": f"MASE Tag {day}", "Wert": val})

    # WAPE pro Tag
    for day in range(1, HORIZON + 1):
        val = daily_df.loc[daily_df["horizon"] == day, "wape"].iloc[0]
        rows.append({"Metrik": f"WAPE Tag {day}", "Wert": val})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Naivprognose in Summe über den gesamten Horizont - Durchschnitt über alle Serien
# ---------------------------------------------------------------------
def seasonal_naive_baseline(df: pd.DataFrame, split_name: str, denoms: dict) -> dict:

    # Folgt der Sliding-Window logik. 
    # Die Logik ist so, dass es mehrere Fenster gibt die verlichen werden.
    # Es wird immer ein 7 Tage Fenster aus dem Val oder Testsplit mit einem Fenster welches 7 Tage zurückliegt verglichen
    # Dabei rückt das Fenster immer einen Tag weiter in den Val oder Testsplit. Start des ersten Fensters ist der erste Tag von Val oder Test.
    # Da es 28 Val- oder Test-Tage gibt pro Serie, gibt es demnach auch 28 Fenster pro Serie

    # Liste für alle Fenster, am Ende wird der Mittelwert gebildet.
    maes = []
    mses = []
    mases = []

    wape_num = 0.0
    wape_den = 0.0

    n_windows = 0
    n_windows_mase = 0

    for series_id, series_df in df.groupby("series_id"):
        series_df = series_df.sort_values("time_idx").reset_index(drop=True)

        y = series_df["y"].astype(np.float32).values
        splits = series_df["split"].values
        den = denoms.get(series_id, np.nan)  # MASE-Nenner der Serie holen

        # Startpunkte der Fenster ermitteln.
        for t in range(SEASONALITY, len(series_df) - HORIZON + 1):
            if splits[t] != split_name:
                continue

            y_true = y[t : t + HORIZON]  # Echte Werte
            y_pred = y[t - SEASONALITY : t - SEASONALITY + HORIZON]   # Vorhersage: Werte von vor einer Woche

            ### Debug
            # print (n_windows, y_true)
            # print(n_windows, y_pred)

            if len(y_pred) != HORIZON:
                continue

            # Fehlerwerte ermittlen
            error = y_true - y_pred
            absolut_error = np.abs(error)
            squared_error = error ** 2

            mean_absolut_error = float(np.mean(absolut_error))
            mean_squared_error = float(np.mean(squared_error))

            # Fehler pro Fenster in die Liste eintragen
            maes.append(mean_absolut_error)
            mses.append(mean_squared_error)

            # WAPE Zähler und Nenner ermitteln
            wape_num += float(np.sum(squared_error))
            wape_den += float(np.sum(np.abs(y_true)))

            # Window hochzählen
            n_windows += 1
           
            ### Debug
            # print (n_windows)

            # MASE berechnen und mases Liste anfügen, wenn Nenner MAE größer Null ist
            if np.isfinite(den) and den > 0.0:
                mases.append(absolut_error / den)
                n_windows_mase += 1


    return {
        "split": split_name,
        "series": int(df["series_id"].nunique()),
        "n_windows": int(n_windows),
        "n_windows_mase": int(n_windows_mase),
        "mae": float(np.mean(maes)) if maes else np.nan,
        "mse": float(np.mean(mses)) if mses else np.nan,
        "mase": float(np.mean(mases)) if mases else np.nan,
        "wape": calc_wape(wape_num, wape_den),
        "wape_num": float(wape_num),
        "wape_den": float(wape_den),
    }

# ---------------------------------------------------------------------
# Naivprognose pro Tag über den Prognosehorizont = 28 Tage
# ---------------------------------------------------------------------
def seasonal_naive_by_day(df: pd.DataFrame, split_name: str, denoms: dict):

    # Für jeden der 28 Vorhersagetage eine eigene Liste für die Fehler. mae_day enthält somit weitere 28 Listen im "Bauch"
    # mae_day[0] sammelt alle MAE-Werte für Tag 1, mae_day[1] für Tag 2 usw.
    mae_day = [[] for _ in range(HORIZON)] # _ steht für Variable brauch man nicht mehr
    mse_day = [[] for _ in range(HORIZON)]
    mase_day = [[] for _ in range(HORIZON)]

    # das Gleiche für Zähler und Nenner der WAPE-Berechnung
    wape_num_day = [0.0 for _ in range(HORIZON)]
    wape_den_day = [0.0 for _ in range(HORIZON)]

    for series_id, series_df in df.groupby("series_id"):
        series_df = series_df.sort_values("time_idx").reset_index(drop=True)

        y = series_df["y"].astype(np.float32).values
        splits = series_df["split"].values
        den = denoms.get(series_id, np.nan)

        for t in range(SEASONALITY, len(series_df) - HORIZON + 1):
            if splits[t] != split_name:
                continue

            y_true = y[t : t + HORIZON]
            y_pred = y[t - SEASONALITY : t - SEASONALITY + HORIZON]

            if len(y_pred) != HORIZON:
                continue

            error = y_true - y_pred
            absolut_error = np.abs(error)
            squared_error = error ** 2

            # Jeden der 28 Tage einzeln in die zugehörige Liste eintragen.
            # So landen Fehler von Tag 1 in mae_day[0],
            # und Tag 2 in mae_day[1] usw
            # h steht für einen Tag aus dem Horizont 
            for h in range(HORIZON):
                mae_day[h].append(float(absolut_error[h]))
                mse_day[h].append(float(squared_error[h]))

                # WAPE pro Tag 
                wape_num_day[h] += float(absolut_error[h])
                wape_den_day[h] += float(abs(y_true[h]))

                # MASE pro Tag, wieder nur wenn Nenner (MAE) größer null 
                if np.isfinite(den) and den > 0.0:
                    mase_day[h].append(float(absolut_error[h] / den))
   
    # Für jeden der 28 Tage den Mittelwert über alle Serien und Fenster bilden  
    rows = []
    for h in range(HORIZON):
        rows.append(
            {
                "horizon": h + 1,
                "mae": float(np.mean(mae_day[h])) if mae_day[h] else np.nan,
                "mse": float(np.mean(mse_day[h])) if mse_day[h] else np.nan,
                "mase": float(np.mean(mase_day[h])) if mase_day[h] else np.nan,
                "wape": calc_wape(wape_num_day[h], wape_den_day[h]),
                "wape_num": float(wape_num_day[h]),
                "wape_den": float(wape_den_day[h]),
                "n": int(len(mae_day[h])),
                "n_mase": int(len(mase_day[h])),
            }
        )
    
    # Rückgabe als Dataframe 
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Naivprognose pro Tag über den Prognosehorizont = 28 Tage
# ---------------------------------------------------------------------
def seasonal_naive_by_week(df: pd.DataFrame, split_name: str, denoms: dict):

    # Berechnet MAE, MSE, MASE und WAPE getrennt für jede der 4 Wochen
    # Im Grunde identische logik wie vorherige Funktion nur mit Wochen statt Tagen
    # Kommentierung weniger Umfangreich da fast identische Logik

    weeks = [(0, 7), (7, 14), (14, 21), (21, 28)]
    rows = []

    # Unterschied zu den Tagen: Hier wird über die Wochen iteriert. Die Liste weeks enthält die Tage für die Wochen.
    for week_idx, (start, end) in enumerate(weeks, start=1):
        maes = []
        mses = []
        mases = []

        wape_num = 0.0
        wape_den = 0.0

        n_windows = 0
        n_windows_mase = 0

        for series_id, series_df in df.groupby("series_id"):
            series_df = series_df.sort_values("time_idx").reset_index(drop=True)

            y = series_df["y"].astype(np.float32).values
            splits = series_df["split"].values
            den = denoms.get(series_id, np.nan)

            for t in range(SEASONALITY, len(series_df) - HORIZON + 1):
                if splits[t] != split_name:
                    continue

                y_true = y[t : t + HORIZON]
                y_pred = y[t - SEASONALITY : t - SEASONALITY + HORIZON]

                if len(y_pred) != HORIZON:
                    continue

                y_true_week = y_true[start:end]
                y_predict_week = y_pred[start:end]

                error = y_true_week - y_predict_week
                absolut_error = np.abs(error)
                squared_error = error ** 2

                mean_absolut_error = float(np.mean(absolut_error))
                mean_squared_error = float(np.mean(squared_error))

                maes.append(mean_absolut_error)
                mses.append(mean_squared_error)

                # WAPE pro Woche
                wape_num += float(np.sum(squared_error))
                wape_den += float(np.sum(np.abs(y_true_week)))

                n_windows += 1

                # MASE pro Woche, wenn MAE größer Null
                if np.isfinite(den) and den > 0.0:
                    mases.append(mean_absolut_error / den)
                    n_windows_mase += 1

        rows.append(
            {
                "week": week_idx,
                "mae": float(np.mean(maes)) if maes else np.nan,
                "mse": float(np.mean(mses)) if mses else np.nan,
                "mase": float(np.mean(mases)) if mases else np.nan,
                "wape": calc_wape(wape_num, wape_den),
                "wape_num": float(wape_num),
                "wape_den": float(wape_den),
                "n_windows": int(n_windows),
                "n_windows_mase": int(n_windows_mase),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Logging der Ergebnisse und schreiben der CSV
# ---------------------------------------------------------------------

# Erstellen eines "runs" Ordners wenn noch nicht vorhanden
def create_run_folder(base_dir: Path):
    run_dir = base_dir / "Native_Baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

# Funktion zum speichern der JSON
def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    total_start = time.time()

    # Einlesen der Daten
    print("Laden der Daten...")
    df = load_data()

    # Nur für Dokumentation
    config = {
        "model": "SaisonalNaive7",
        "data_dir": str(DATA_DIR),
        "split": SPLIT_NAME,
        "series": int(df["series_id"].nunique()),
        "horizon": HORIZON,
        "seasonality": SEASONALITY,
        "csv_path": str(CSV_PATH),
        "mase_denominator": "mean(|y_t - y_{t-7}|) auf dem Traningssplit pro Serie",
    }

    run_dir = create_run_folder(RUNS_DIR)
    save_json(run_dir / "config.json", config)

    # Aufrufen der Berechnungsfunktion für MASE Nenner also MAE
    print("Berechnung des MASE Nenners auf dem Traninssplit...")
    denoms = compute_mase_denominators(df, seasonality=SEASONALITY)

    # Start der Evaulation auf dem gewählten Split und zusammenfügen der Ergebnisse
    print(f"Evaluieren des Modells auf dem Split ='{SPLIT_NAME}'")
    results = seasonal_naive_baseline(df, SPLIT_NAME, denoms)
    daily_df = seasonal_naive_by_day(df, SPLIT_NAME, denoms)
    weekly_df = seasonal_naive_by_week(df, SPLIT_NAME, denoms)
    metrics_table_df = build_metrics_table(results, weekly_df, daily_df)

    # Nochmal alles übersichtlich ausgeben um schnell Infos zu bekommen. Insbesondere für Debugging genutzt.
    print("\n=== Ergebnisse der Naivprognose ===")
    print(f"Split         : {results['split']}")
    print(f"Series        : {df['series_id'].nunique()}")
    print(f"Windows       : {results['n_windows']}")
    print(f"Windows (MASE): {results['n_windows_mase']}")
    print(f"MAE           : {results['mae']:.4f}")
    print(f"MSE           : {results['mse']:.4f}")
    print(f"MASE          : {results['mase']:.4f}")
    print(f"WAPE          : {results['wape']:.4f}")
    print("=====================================")

    # Zusammenfassung als JSON ausgeben
    summary = {
        "split": results["split"],
        "mae": results["mae"],
        "mse": results["mse"],
        "mase": results["mase"],
        "wape": results["wape"],
        "total_time_sec": float(time.time() - total_start),
        "run_dir": str(run_dir),
    }

    save_json(run_dir / "summary.json", summary)

    # Export der Übersichtstabelle mit Gesamt, Wochen und Tagen aller Metriken
    metrics_table_df.to_csv(run_dir / "metrics_table.csv", index=False)
    
    print("Gepspeichert:", run_dir / "metrics_table.csv")
    print("Ordner:", run_dir)


if __name__ == "__main__":
    main()
