"""Load artifact sekali, lalu prediksi. Dipakai FastAPI (app/api/main.py) dan Streamlit mode A (app/ui/streamlit_app.py)."""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# model.joblib menyimpan referensi ke src.features.tambah_fitur, jadi root repo harus bisa di-import
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "artifacts"


@lru_cache(maxsize=1)
def load():
    """(model, metadata). lru_cache = di-load sekali per proses, bukan per request."""
    model_path, meta_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} tidak ada. Sudah di-commit? Cek .gitignore.")
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return model, meta


def predict_rows(rows: list[dict]) -> list[dict]:
    """Prediksi beberapa baris sekaligus. Tiap baris berisi kolom fitur_input dari metadata."""
    model, meta = load()
    mulai = time.perf_counter()
    X = pd.DataFrame(rows)[meta["fitur_input"]]
    proba = model.predict_proba(X)[:, 1]
    latency = round((time.perf_counter() - mulai) * 1000, 1)
    return [{
        "probabilitas_gagal": round(float(p), 4),
        "status": "SEGERA SERVIS" if p >= meta["threshold"] else "AMAN",
        "threshold": meta["threshold"],
        "versi_model": meta["versi_model"],
        "latency_ms": latency,
    } for p in proba]


def predict_one(row: dict) -> dict:
    return predict_rows([row])[0]
