"""Fitur fisika. Dipakai notebook training (P2) dan FastAPI (P3), jadi jangan diubah di satu tempat saja."""
import numpy as np
import pandas as pd


def tambah_fitur(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daya_W"] = df["torsi_Nm"] * 2 * np.pi * df["kecepatan_rpm"] / 60
    df["delta_suhu_K"] = df["suhu_proses_K"] - df["suhu_udara_K"]
    df["strain_tool"] = df["keausan_tool_menit"] * df["torsi_Nm"]
    return df
