"""UI Streamlit untuk teknisi pabrik.

Jalankan dari root repo:
    streamlit run app/ui/streamlit_app.py

Dua mode:
- Mode A (API_URL kosong): model di-load langsung di sini. Paling gampang di-deploy.
- Mode B (API_URL diisi, mis. https://xxx.hf.space): memanggil FastAPI lewat HTTP.
  Isi lewat environment variable API_URL, atau di Streamlit Cloud: Settings → Secrets → API_URL = "https://..."
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Streamlit menjalankan file ini dari app/ui, jadi root repo ditambah manual
from app import predictor  # noqa: E402


def _api_url() -> str:
    try:
        url = st.secrets.get("API_URL", "")
    except Exception:  # tidak ada secrets.toml: wajar saat lokal
        url = ""
    return (url or os.getenv("API_URL", "")).rstrip("/")


API_URL = _api_url()
TIMEOUT = 60  # detik; free tier yang sedang tidur butuh waktu untuk bangun


def prediksi(rows: list[dict]) -> list[dict]:
    if not API_URL:
        return predictor.predict_rows(rows)
    path = "/predict" if len(rows) == 1 else "/predict/batch"
    r = requests.post(API_URL + path, json=rows[0] if len(rows) == 1 else rows, timeout=TIMEOUT)
    if r.status_code == 422:
        raise ValueError(f"Input ditolak API (422): {r.json().get('detail')}")
    r.raise_for_status()
    hasil = r.json()
    return [hasil] if len(rows) == 1 else hasil


def info_model() -> dict:
    if not API_URL:
        return predictor.load()[1]
    r = requests.get(API_URL + "/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Prediksi Kegagalan Mesin CNC", layout="centered")
st.title("Prediksi kegagalan mesin CNC")
st.caption("Isi kondisi mesin, dapatkan peluang mesin gagal dalam 24 jam ke depan.")

try:
    META = info_model()
except Exception as e:
    st.error(f"Model belum siap: {e}")
    st.stop()

FITUR = META["fitur_input"]
mode = f"Mode B · memanggil API {API_URL}" if API_URL else "Mode A · model di-load langsung"
st.info(f"{mode} · versi model {META['versi_model']} · threshold {META['threshold']}")

tab_satu, tab_batch = st.tabs(["Satu mesin", "Banyak mesin (CSV)"])

with tab_satu:
    with st.form("input"):
        c1, c2 = st.columns(2)
        tipe = c1.selectbox("Tipe produk", ["L", "M", "H"], help="Kualitas produk yang sedang dikerjakan")
        shift = c2.selectbox("Shift operator", ["pagi", "siang", "malam"], index=2)
        suhu_udara = c1.number_input("Suhu udara (K)", 250.0, 350.0, 301.2, 0.1)
        suhu_proses = c2.number_input("Suhu proses (K)", 250.1, 400.0, 310.8, 0.1)
        rpm = c1.number_input("Kecepatan (rpm)", 500.0, 4000.0, 1290.0, 10.0)
        torsi = c2.number_input("Torsi (Nm)", 0.0, 150.0, 62.5, 0.5)
        aus = c1.number_input("Keausan tool (menit)", 0.0, 500.0, 228.0, 1.0)
        getar = c2.number_input("Getaran (mm/s)", 0.0, 50.0, 4.1, 0.1)
        jam = c1.number_input("Jam sejak servis", 0.0, 10_000.0, 410.0, 1.0)
        kirim = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

    if kirim:
        row = {"tipe_produk": tipe, "suhu_udara_K": suhu_udara, "suhu_proses_K": suhu_proses,
               "kecepatan_rpm": rpm, "torsi_Nm": torsi, "keausan_tool_menit": aus,
               "getaran_mm_s": getar, "jam_sejak_servis": jam, "shift_operator": shift}
        try:
            hasil = prediksi([row])[0]
        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")
        else:
            p = hasil["probabilitas_gagal"]
            teks = f"Peluang gagal dalam 24 jam: **{p:.1%}** (threshold {hasil['threshold']})"
            if hasil["status"] == "SEGERA SERVIS":
                st.error(f"### SEGERA SERVIS\n{teks}")
            else:
                st.success(f"### AMAN\n{teks}")
            st.progress(min(p, 1.0))
            with st.expander("Lihat request & response JSON"):
                st.json({"request": row, "response": hasil})

with tab_batch:
    st.write("Upload CSV dengan kolom: " + ", ".join(f"`{k}`" for k in FITUR))
    contoh = ROOT / "data" / "contoh_batch.csv"
    if contoh.exists():
        st.download_button("Unduh contoh CSV", contoh.read_bytes(), "contoh_batch.csv", "text/csv")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        kurang = [k for k in FITUR if k not in df.columns]
        if kurang:
            st.error(f"Kolom kurang: {kurang}")
        else:
            df["shift_operator"] = df["shift_operator"].astype(str).str.strip().str.lower()
            X = df[FITUR]
            rows = X.astype(object).where(X.notna(), None).to_dict("records")  # NaN → null supaya valid JSON
            try:
                hasil = pd.DataFrame(prediksi(rows))
            except Exception as e:
                st.error(f"Gagal memprediksi: {e}")
            else:
                out = pd.concat([df.reset_index(drop=True), hasil[["probabilitas_gagal", "status"]]], axis=1)
                out = out.sort_values("probabilitas_gagal", ascending=False)
                n = int((out["status"] == "SEGERA SERVIS").sum())
                st.metric("Mesin yang perlu segera diservis", f"{n} dari {len(out)}")
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.download_button("Unduh hasil", out.to_csv(index=False).encode(), "hasil_prediksi.csv", "text/csv")
