"""FastAPI untuk model prediksi kegagalan mesin CNC.

Jalankan dari root repo:
    uvicorn app.api.main:app --reload
Lalu buka http://127.0.0.1:8000/docs
"""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # supaya `app` dan `src` bisa di-import
from app import predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prediksi")

MAX_BATCH = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()  # 1 · load sekali saat startup, bukan per request
    yield


app = FastAPI(title="Prediksi Kegagalan Mesin CNC", version="1.0.0", lifespan=lifespan)


class InputData(BaseModel):
    """2 · Validasi: input di luar batas ditolak 422 sebelum menyentuh model.
    Sensor boleh null (sensor rusak): model mengisinya dengan median saat training."""
    tipe_produk: Literal["L", "M", "H"]
    suhu_udara_K: float | None = Field(ge=250, le=350)
    suhu_proses_K: float | None = Field(gt=250, le=400)
    kecepatan_rpm: float | None = Field(ge=500, le=4000)
    torsi_Nm: float | None = Field(ge=0, le=150)
    keausan_tool_menit: float | None = Field(ge=0, le=500)
    getaran_mm_s: float | None = Field(ge=0, le=50)
    jam_sejak_servis: float | None = Field(ge=0, le=10_000)
    shift_operator: Literal["pagi", "siang", "malam"]

    @field_validator("shift_operator", mode="before")
    @classmethod
    def rapikan_shift(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    model_config = {"json_schema_extra": {"examples": [{
        "tipe_produk": "L", "suhu_udara_K": 301.2, "suhu_proses_K": 310.8, "kecepatan_rpm": 1290,
        "torsi_Nm": 62.5, "keausan_tool_menit": 228, "getaran_mm_s": 4.1, "jam_sejak_servis": 410,
        "shift_operator": "malam"}]}}


class Prediksi(BaseModel):
    probabilitas_gagal: float
    status: Literal["SEGERA SERVIS", "AMAN"]
    threshold: float
    versi_model: str
    latency_ms: float


def _log(inputs: list[dict], outputs: list[dict]):
    """Setiap prediksi di-log (checklist P3): input + output, supaya bisa diaudit dan dipantau drift-nya."""
    for x, y in zip(inputs, outputs):
        log.info("prediksi %s", json.dumps({"input": x, "output": y, "ts": time.time()}))


@app.get("/health")
def health():
    _, meta = predictor.load()
    return {"status": "ok", "versi_model": meta["versi_model"], "threshold": meta["threshold"],
            "fitur_input": meta["fitur_input"]}


@app.post("/predict", response_model=Prediksi)
def predict(x: InputData):
    """3 · JSON masuk, prediksi keluar."""
    row = x.model_dump()
    hasil = predictor.predict_one(row)
    _log([row], [hasil])
    return hasil


@app.post("/predict/batch", response_model=list[Prediksi])
def predict_batch(xs: list[InputData]):
    if len(xs) > MAX_BATCH:
        raise HTTPException(413, f"Maksimal {MAX_BATCH} baris per request.")
    rows = [x.model_dump() for x in xs]
    if not rows:
        return []
    hasil = predictor.predict_rows(rows)
    _log(rows, hasil)
    return hasil
