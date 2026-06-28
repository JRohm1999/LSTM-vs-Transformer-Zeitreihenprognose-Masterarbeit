# Skript für das Preprocessing des Datensatzes
# Erstellung der m5_long.csv aus dem Subset und den anderen CSV Dateien wie calendar.csv und sell_prices.csv

# Import der notwendigen Bibliotheken
from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Konfigurationen
# ------------------------------------------------------------

RAW_DIR = Path("data/raw")
SUBSET_DIR = Path("data/preprocessed/subsets")
OUT_DIR = Path("data/preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = SUBSET_DIR / "subset_1000_series.csv" # Hier das jeweilige Subset angeben. Das Subset muss vorher erstellt werden.
CAL_FILE = RAW_DIR / "calendar.csv"
PRICES_FILE = RAW_DIR / "sell_prices.csv"

# Subset
MAX_SERIES = 1000
SEED = 42  #Random_Seed für das zufällige ziehen von weiteren Serien, wenn notwendig

# Forecast-Horizont
HORIZON = 28

# Autoregressive Features - werden von allem Modellen genutzt
LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]


# Rohdaten laden
sales = pd.read_csv(SALES_FILE)
calendar = pd.read_csv(CAL_FILE)
prices = pd.read_csv(PRICES_FILE)

# ------------------------------------------------------------
# Hilfsfunktion zum logarithmieren von Werten. Zusätzlich werden sie in das Format Float32 umgewandelt um die Datenmenge zu reduzieren
# ------------------------------------------------------------
def log_transform_values(x: pd.Series):
    return np.log1p(x).astype(np.float32)

# ------------------------------------------------------------
# Subset verkleinern bei Bedarf
# ------------------------------------------------------------
# Wurde für die ersten Tests genutzt, damit nicht jedes mal das Subset verkleinert werden muss um etwas zu testen.
if MAX_SERIES is not None and MAX_SERIES < len(sales):
    sales = sales.sample(n=MAX_SERIES, random_state=SEED).reset_index(drop=True)

# ------------------------------------------------------------
#  Transormieren von Wide zu Long Format
# ------------------------------------------------------------

# Hintergrund: Subset enthält die Tageswerte als Spalten d_1 ... d_1913.
# Für Modellierung und Join-Operationen ist Long-Format (eine Zeile pro Tag) einfacher.

# Alle ID Spalten
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Alle d_ Spalten, also die einzelnen Tage
d_cols = [c for c in sales.columns if c.startswith("d_")]

# In Long-Format umwandeln. Tage werden von Spalten in Zeilen transformiert.
# Spalte d enthält die Tageskennung (z.B. d_1), y die Verkaufszahl.
df = sales.melt(
    id_vars=id_cols,
    value_vars=d_cols,
    var_name="d",
    value_name="y"
)

# ------------------------------------------------------------
# Kalender Join (calendar.csv)
# ------------------------------------------------------------

# Nur die wichtigen Spalten aus dem Kalender werden behalten.
cal_cols = [
    "d", "date", "wm_yr_wk",
    "wday", "month", "year",
    "event_name_1", "event_name_2",
    "snap_CA", "snap_TX", "snap_WI",
]

# Kleiner Kalender-Datensatz mit den definierten Spalten für Join
calendar_small = calendar[cal_cols].copy()

# Join auf d-Spalte (Tag)
df = df.merge(calendar_small, on="d", how="left")


# ------------------------------------------------------------
# Preis Join (sell_prices.csv)
# ------------------------------------------------------------
# Preise sind auf Woche (wm_yr_wk) + item_id + store_id definiert
# Der Join der Preise erfolgt daher auf diese drei Spalten und so bekommt jeder Verkauf den passenden Preis

df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

## Join-Lücken der Preise analysieren
# Es gibt nicht für jede item_id + store_id + week einen Preis-Eintrag.Da manche Produkte erst ab einem bestimmten Datum verkauft wurden
# # Die Verkaufszahlen sind aber für alle Tage definiert. Somit entstehen Lücken (NaNs) in der sell_price Spalte
na_rate = df["sell_price"].isna().mean()
df_na = df[df["sell_price"].isna()]
na_count = df["sell_price"].isna().sum()

# Ausgeben des Wertes zur Information
print("sell_price NaN rate:", na_rate, "NaN count:", na_count)

# ------------------------------------------------------------
# Datentypen anpassen und leere Verkaufspreise mit 0 auffüllen
# ------------------------------------------------------------

# Datum als datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# y: Falls im Datensatz NaNs  vorhanden sind, auf 0 setzen
df["y"] = df["y"].fillna(0.0).astype(np.float32)

# Anpassen des Datentyps auf float32. Speicherersparnis gegenüber float64.
df["sell_price"] = df["sell_price"].astype("float32")


# ------------------------------------------------------------
# Serien-ID und Sortierung
# ------------------------------------------------------------

# Erstellung einer eindeutigen Serien-ID pro (store_id, item_id) Kombination.
# Jede Store Item Kombination ist eine eigene Zeitreihe.
df["series_id"] = (df["store_id"].astype(str) + "_" + df["item_id"].astype(str))

# Sortierung nach series_id
df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

# time_idx: fortlaufender Zeitindex pro Serie
df["time_idx"] = df.groupby("series_id").cumcount().astype(np.int32)


# ------------------------------------------------------------
# Fehlende Preise mit null füllen und price_missing Feature hinzufügen
# ------------------------------------------------------------
# Allgemeines price_missing Flag (1.0 wenn Preis ursprünglich fehlte, sonst 0.0)
df["price_missing"] = df["sell_price"].isna().astype(np.float32)

# Alle Nans mit 0.0 ersetzen und zusätzliches price_missing Flag setzen
df["sell_price"] = df["sell_price"].fillna(0.0).astype(np.float32)

# ------------------------------------------------------------
# SNAP und Event-Flags erzeugen
# ------------------------------------------------------------
# SNAP ist staatsspezifisch. Für eine einzige Feature-Spalte wird je nach state_id
# die passende SNAP-Spalte gewählt.

# SNAP wird zunächst auf 0.0 gesetzt (kein SNAP)
df["snap"] = 0.0

# Je nach state_id wird die passende SNAP-Spalte zugewiesen
df.loc[df["state_id"] == "CA", "snap"] = df.loc[df["state_id"] == "CA", "snap_CA"]
df.loc[df["state_id"] == "TX", "snap"] = df.loc[df["state_id"] == "TX", "snap_TX"]
df.loc[df["state_id"] == "WI", "snap"] = df.loc[df["state_id"] == "WI", "snap_WI"]

# SNAP NaNs falls vorhanden auf 0.0 setzen und in float32 umwandeln
df["snap"] = df["snap"].fillna(0.0).astype(np.float32)

# Wenn event_name_1 oder event_name_2 nicht NaN ist, wird das entsprechende Flag auf 1.0 gesetzt, sonst 0.0.    
df["has_event_1"] = df["event_name_1"].notna().astype(np.float32)
df["has_event_2"] = df["event_name_2"].notna().astype(np.float32)


# ------------------------------------------------------------
# Splits train/val/test pro Serie markieren
# ------------------------------------------------------------
#   Val = vorletzte 28 Tage, Test = letzte 28 Tage (je Serie).

parts = []
kept_series = 0

for sid, g in df.groupby("series_id"):
    g = g.sort_values("time_idx").reset_index(drop=True)
    n = len(g)

    # Mindestlänge: train + val + test + Puffer
    # Im M5-Datensatz sind alle Serien 1941 Tage lang, daher greift
    # diese Prüfung nie – sie bleibt aber als Absicherung für kürzere Subsets.  
    if n < (3 * HORIZON + 20):
        continue

    # Startzeitpunkte der Splits festlegen
    test_start = n - HORIZON
    val_start = n - 2 * HORIZON

    g_train = g.iloc[:val_start].copy()
    g_val = g.iloc[val_start:test_start].copy()
    g_test = g.iloc[test_start:].copy()

    g_train["split"] = "train"
    g_val["split"] = "val"
    g_test["split"] = "test"

    parts.append(pd.concat([g_train, g_val, g_test], axis=0))
    kept_series += 1

# Ausgeben des DF welches nun die drei Splits enthält
df = pd.concat(parts, axis=0).reset_index(drop=True)


# ------------------------------------------------------------
# Logarithmische Transformation von Sales und Price
# ------------------------------------------------------------
# Hintergrund:
# y_log stabilisiert die Verteilung für Training
# price_s betont relative Preisunterschiede

df["y_log"] = log_transform_values(df["y"])
df["price_s"] = log_transform_values(df["sell_price"])


# ------------------------------------------------------------
# Kalenderfeatures skalieren
# ------------------------------------------------------------
# Hintergrund:
# Kalenderfeatures werden auf [0,1] skaliert, um numerische Dominanz zu vermeiden.
# Es handelt sich um einfache lineare Skalierung.

df["wday_s"] = (df["wday"].fillna(1).astype(np.float32) / 7.0).astype(np.float32)
df["month_s"] = (df["month"].fillna(1).astype(np.float32) / 12.0).astype(np.float32)

y_min = float(df["year"].min())
y_max = float(df["year"].max())
denom = max(1.0, y_max - y_min)
df["year_s"] = ((df["year"].fillna(y_min).astype(np.float32) - y_min) / denom).astype(np.float32)


# ------------------------------------------------------------
# y_z berechnen - Diese Funktion ist ein Überbleibsel eines ersten Vorgehens, welche die Sales Werte als y_z skaliert.
# Dieses Vorgehen wurde nicht weiter verfolgt, sondern mit y_log garbeitet
# ------------------------------------------------------------
# train_only = df[df["split"] == "train"].copy()

# stats = (
#     train_only.groupby("series_id")["y_log"]
#     .agg(["mean", "std"])
#     .rename(columns={"mean": "mu", "std": "sigma"})
# )

# stats["sigma"] = stats["sigma"].replace(0.0, 1.0).fillna(1.0)

# df = df.merge(stats, on="series_id", how="left")
# df["mu"] = df["mu"].fillna(0.0).astype(np.float32)
# df["sigma"] = df["sigma"].fillna(1.0).astype(np.float32)

# df["y_z"] = ((df["y_log"] - df["mu"]) / df["sigma"]).astype(np.float32)


# ------------------------------------------------------------
# Autoregressive Features aus y_log (Lags / Rollings)
# ------------------------------------------------------------
df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

# Lags
for lag in LAGS:
    df[f"y_log_lag_{lag}"] = df.groupby("series_id")["y_log"].shift(lag)

# Rolling (immer nur Vergangenheit -> shift(1))
for w in ROLL_WINDOWS:
    shifted = df.groupby("series_id")["y_log"].shift(1)

    df[f"y_log_roll_mean_{w}"] = (
        shifted.groupby(df["series_id"])
        .rolling(window=w, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"y_log_roll_std_{w}"] = (
        shifted.groupby(df["series_id"])
        .rolling(window=w, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

# NaNs am Serienanfang mit null auffüllen, falls es welche geben sollte
lag_cols = [f"y_log_lag_{l}" for l in LAGS]
roll_cols = [f"y_log_roll_mean_{w}" for w in ROLL_WINDOWS] + [f"y_log_roll_std_{w}" for w in ROLL_WINDOWS]
df[lag_cols + roll_cols] = df[lag_cols + roll_cols].fillna(0.0).astype(np.float32)


# ------------------------------------------------------------
# Spaltenauswahl für den finalen Traningsdatensatz festlegen
# ------------------------------------------------------------

keep_cols = [
    "series_id", "time_idx", "date", "split",
    "store_id", "item_id", "dept_id", "cat_id", "state_id",
    "y", "y_log",
    "sell_price", "price_s", "price_missing", "snap","wday",
    "wday_s", "month", "month_s", "year_s",
    "has_event_1", "has_event_2",
]

keep_cols += lag_cols
keep_cols += [f"y_log_roll_mean_{w}" for w in ROLL_WINDOWS]
keep_cols += [f"y_log_roll_std_{w}" for w in ROLL_WINDOWS]

df = df[keep_cols].copy()


# ------------------------------------------------------------
# Speichern der Datei
# ------------------------------------------------------------

out_csv = OUT_DIR / "m5_long.csv"
df.to_csv(out_csv, index=False)
print("Gespeichert unter:", str(out_csv))
