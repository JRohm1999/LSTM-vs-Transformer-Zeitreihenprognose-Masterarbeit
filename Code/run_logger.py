# run_logger.py
# -----------------------------------------------------------------------------
# Zweck dieses Skriptes:
#   Diese Datei enthält Hilfsfunktionen für Logging in den Hauptskripten zum Traning und Testen der Modelle.

# Importieren sämtlicher Bibliotheken die für die Funktionen gebracht werden
import json
import platform
import socket
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _make_folder_safe(text):
    # Diese Funktion erzeugt einen dateisystemfreundlichen String.
    # Sonderzeichen werden ersetzt, um Probleme bei der Ordnererstellung zu vermeiden.
    text = str(text)
    out = []
    for ch in text:
        if ch.isalnum() or ch in "-_=.+":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)

# Systeminfos sammeln und zurückgeben
def get_system_info():
    # Es werden Systeminformationen gesammelt, die für eine Reproduktion der Ergebnisse
    # relevant sein können. Dies ist insbesondere bei GPU-Training sinnvoll.
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # Torch- und CUDA-Informationen werden ergänzt, da diese häufig die Performance
    # und das Verhalten der Modelle beeinflussen können.
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_info_error"] = str(e)

    return info

# Sichern eines JSON Files 
def save_json(path, data):
    # Speicherung einer Python-Datenstruktur im JSON-Format.
    # Das Format ist für Menschen lesbar und für spätere Analysen einfach nutzbar.
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# Sichern einer Excel-Datei
def save_excel(run_dir, config, epoch_rows, summary):
    run_dir = Path(run_dir)
    out_path = run_dir / "metrics.xlsx"

    df_config = pd.DataFrame([config])
    df_epochs = pd.DataFrame(epoch_rows)
    df_summary = pd.DataFrame([summary])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_config.to_excel(writer, sheet_name="config", index=False)
        df_epochs.to_excel(writer, sheet_name="epochs", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    return out_path

# Erstellung einfacher Visualisierungen des Trainingsverlaufs mit Plots
def save_plots(run_dir, epoch_rows):
    run_dir = Path(run_dir)
    df = pd.DataFrame(epoch_rows)

    if df.empty:
        return

    # Plot 1: Loss-Verlauf
    if "train_loss" in df.columns or "val_loss" in df.columns:
        plt.figure()

        if "train_loss" in df.columns and df["train_loss"].notna().any():
            plt.plot(df["epoch"], df["train_loss"], label="train_loss")

        if "val_loss" in df.columns and df["val_loss"].notna().any():
            plt.plot(df["epoch"], df["val_loss"], label="val_loss", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss-Verlauf pro Epoche")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss.png", dpi=150)
        plt.close()

    # Plot 2: Metrik-Verlauf (MASE)
    metric_cols = [ "val_mase"]
    if any(c in df.columns for c in metric_cols):
        plt.figure()

        if "val_mase" in df.columns and df["val_mase"].notna().any():
            plt.plot(df["epoch"], df["val_mase"], label="MASE", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Wert")
        plt.title("MASE pro Epoche")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "metrics.png", dpi=150)
        plt.close()
