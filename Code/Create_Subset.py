# In diesem Skript wird aus dem gesamten M5-Datensatz das Subset erstellt.

# Importieren der notwendigen Bibliotheken
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Allgemeine Konfigurationen
# ---------------------------------------------------------------------

INPUT_PATH = Path("data/raw/sales_train_evaluation.csv")
OUTPUT_DIR = Path("data/preprocessed/subsets")
OUTPUT_NAME = "subset_1000_series.csv"

TARGET_NUMBER_OF_SERIES = 1000 # Anazhl der Serien festlegen
RANDOM_SEED = 42 # Ist zufällig auf 42 gesetzt (keine Bedeutung) / Wird genutzt wenn eine zufällige Serie gezogen werden muss. (Wird weiter unten eingesetzt)

# Nachfrageklassen über Anteil aktiver Tage (y > 0)
# Wichtige Festlegung, für die statifizierte Auswahl der Serien
LOW_THRESHOLD = 0.30   # <= 30% aktive Tage 
MID_THRESHOLD = 0.70   # 31-70% aktive Tage, Rest => high

# Zielanteile pro Nachfrageklasse
# drei Klassen: low, mid, high. 
# Konkret: 25% der Serien im Subset sollen low Serien sein (low_Threshold), 50% mittlere, 25% hohe.
# Enscheidung daher um zu vermeiden, dass zu viele Null Serien enthalten sind. So wird ein ausgelichenes Subset erstellt.
DEMAND_CLASS_SHARES = {
    "low": 0.25,
    "mid": 0.50,
    "high": 0.25,
}

# ---------------------------------------------------------------------
# Funktion zum laden der Daten aus der Roh CSV (sales_train_evaluation.csv) - Wide Format (pro Serie eine Spalte)
# Gleichzeitig Valdidieren ob alle benötigten Spalten enthalten sind
# ---------------------------------------------------------------------

def load_data(input_path: Path):
    
    wide_data = pd.read_csv(input_path)
    required_columns = {"id", "state_id", "cat_id"}
    missing_columns = required_columns - set(wide_data.columns)

    # Fehlerausgabe wenn in der CSV die benötigten Spalten "id", "state_id", "cat_id" fehlen
    if missing_columns:
        raise ValueError(f"Fehlende Spalten: {sorted(missing_columns)}")

    # Tagesspalten holen starten immer mit d_
    day_columns = [c for c in wide_data.columns if c.startswith("d_")]
    if not day_columns:
        raise ValueError("Keine Tagesspalten gefunden")

    # Rückgabe als DF 
    return wide_data

# ---------------------------------------------------------------------
# Funktion zum Berechnen der Aktiven Tage pro Serie. Aktiv bedeutet Sales > 0
# ---------------------------------------------------------------------
def compute_series_activity_table(wide_data: pd.DataFrame):
    
    day_columns = [c for c in wide_data.columns if c.startswith("d_")]

    day_values = wide_data[day_columns].to_numpy()
   
    # Zähle aktive Tage pro Serie
    active_day_share = (day_values > 0).mean(axis=1)

    # Tabelle mit allen wichtigen Informationen zu den jeweiligen Serien, um sie nach den Demand Classes zuzuordnen
    series_activity_table = pd.DataFrame({
        "series_id": wide_data["id"].astype(str),
        "active_day_share": active_day_share.astype(float),
        "state_id": wide_data["state_id"].astype(str),
        "cat_id": wide_data["cat_id"].astype(str),
    })

    return series_activity_table

# ---------------------------------------------------------------------
# Funktion zur Einordnung der Serien in die Demand Classes. Hier findet keine Einordnung in Listen statt nur Hilfsfunktion mit return der Klasse
# ---------------------------------------------------------------------
def return_demand_class(active_day_share: float) -> str:
    
    # Wenn aktive Tage kleiner gleich LOW_THRESHOLD sind, dann gibt die Funktion low wieder, usw.
    if active_day_share <= LOW_THRESHOLD:
        return "low"
    if active_day_share <= MID_THRESHOLD:
        return "mid"
    return "high"

# ---------------------------------------------------------------------
# Funktion zur Einordnung der Serien in die Demand Classes. Hier findet keine Einordnung in Listen statt nur Hilfsfunktion mit return der Klasse
# ---------------------------------------------------------------------
def sample_stratified_by_state_and_category(series_table: pd.DataFrame, target_count: int):

    selected_rows = []
    # Anzahl der Gruppen (Staat und Produktkategorie) ermitteln
    # 3 States × 3 Produktkategorie = 9 Gruppen
    number_of_groups = series_table.groupby(["state_id", "cat_id"]).ngroups
    
    # Gleichmäßige Gruppengröße ermitteln: Zielanzahl / Anzahl Gruppen
    # min. 1 pro Gruppe damit keine Gruppe leer bleibt
    target_per_group = max(1, target_count // number_of_groups)

    for (state_id, category_id), group_dataframe in series_table.groupby(["state_id", "cat_id"]):  # Für jede Gruppe einmal in die For-Schleife
        
        # Wenn die Gruppe kleiner ist als die Zielanzahl,
        # einfach alle Zeilen nehmen statt auszuwählen, sonst so viele wie pro Gruppe vorgesehen
        if len(group_dataframe) < target_per_group:
            number_to_sample = len(group_dataframe)
        else:
            number_to_sample = target_per_group

        # Zufällig ohne Zurücklegen sampeln – RANDOM_SEED für Reproduzierbarkeit
        sampled_rows = group_dataframe.sample(
            n=number_to_sample,
            replace=False,
            random_state=RANDOM_SEED
        )

        # Der großen Liste hinzufügen
        selected_rows.append(sampled_rows)
    
    # Alle Gruppen in ein DF zusammenfügen und die Indexe der einzelnen Listen ignorieren (ansonsten würde es mehrmals Index 0, 1, 2 unsw. geben)
    selected_table = pd.concat(selected_rows, ignore_index=True)

    # Falls zu wenig Serien: zufällig auffüllen -> Hierfür wird der Random_Seed vom Anfang genutzt
    # Das passiert wenn einzelne Gruppen kleiner waren als target_per_group
    if len(selected_table) < target_count:
        
        # Nur Serien nehmen die noch nicht im Sample sind, das ~ bedeutet NOT.
        remaining_table = series_table.loc[
            ~series_table["series_id"].isin(selected_table["series_id"])
        ]
        
        # Berechnet wie viele Serien noch fehlen
        missing_count = target_count - len(selected_table)

        # Wenn noch welche fehlen werden sie aus der remaining_table aufgefüllt. Hier auch wieder per Random_Seed sample
        if missing_count > 0 and len(remaining_table) > 0:
            extra_rows = remaining_table.sample(
                n=min(missing_count, len(remaining_table)),
                replace=False,
                random_state=RANDOM_SEED
            )
            selected_table = pd.concat([selected_table, extra_rows], ignore_index=True)

    # Falls zu viele: zufällig kürzen - sollte eigentlich nicht passieren, nur sicherheitshalber eingebaut
    if len(selected_table) > target_count:
        selected_table = selected_table.sample(
            n=target_count,
            replace=False,
            random_state=RANDOM_SEED
        )

    return selected_table

# ---------------------------------------------------------------------
# Funktion zum Speichern der fertigen CSV im Wide Format
# ---------------------------------------------------------------------
def save_subset(wide_data_subset: pd.DataFrame, output_path: Path) -> None:
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide_data_subset.to_csv(output_path, index=False)


# ---------------------------------------------------------------------
# Funktion Main ist die Hauptfunktion die alle anderen Funktionen ausführt
# ---------------------------------------------------------------------

def main():
    # 1) Daten laden (Wide)
    wide_data = load_data(INPUT_PATH)

    # 2) Pro Serie Anteil aktiver Tage berechnen
    series_activity_table = compute_series_activity_table(wide_data)
    series_activity_table["demand_class"] = series_activity_table["active_day_share"].apply(return_demand_class)

    # 3) Zielanzahl pro Nachfrageklasse bestimmen
    target_count_low = int(TARGET_NUMBER_OF_SERIES * DEMAND_CLASS_SHARES["low"])
    target_count_mid = int(TARGET_NUMBER_OF_SERIES * DEMAND_CLASS_SHARES["mid"])
    target_count_high = TARGET_NUMBER_OF_SERIES - target_count_low - target_count_mid

    # 4) Pro Nachfrageklasse stratifiziert nach (state_id, cat_id) sampeln
    low_series_table = series_activity_table[series_activity_table["demand_class"] == "low"]
    mid_series_table = series_activity_table[series_activity_table["demand_class"] == "mid"]
    high_series_table = series_activity_table[series_activity_table["demand_class"] == "high"]

    selected_low = sample_stratified_by_state_and_category(low_series_table, target_count_low)
    selected_mid = sample_stratified_by_state_and_category(mid_series_table, target_count_mid)
    selected_high = sample_stratified_by_state_and_category(high_series_table, target_count_high)

    selected_series_table = pd.concat([selected_low, selected_mid, selected_high], ignore_index=True)

    # 5) Wide-Daten auf die ausgewählten Serien filtern
    selected_series_ids = set(selected_series_table["series_id"].tolist())
    wide_data_subset = wide_data[wide_data["id"].astype(str).isin(selected_series_ids)].copy()

    # 6) Speichern des Subsets
    output_path = OUTPUT_DIR / OUTPUT_NAME
    save_subset(wide_data_subset, output_path)

    # 7) Zusammengefassung  per Print ausgeben
    print("Subset erstellt.")
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {output_path}")
    print(f"Anzahl Serien (soll): {TARGET_NUMBER_OF_SERIES}")
    print(f"Anzahl Serien (ist):  {wide_data_subset.shape[0]}")

    summary_by_class = selected_series_table["demand_class"].value_counts()
    print("\nVerteilung Nachfrageklassen (Series-Level):")
    print(summary_by_class)

    summary_by_state_cat = (
        selected_series_table
        .groupby(["demand_class", "state_id", "cat_id"])
        .size()
        .reset_index(name="n_series")
        .sort_values(["demand_class", "state_id", "cat_id"])
    )
    print("\nStichprobe je (Klasse, State, Cat) erste 30 Zeilen:")
    print(summary_by_state_cat.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
