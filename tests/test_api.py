"""Smoke test API (checklist P3): input normal, ekstrem, dan tidak valid. Jalankan: pytest -q"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

NORMAL = {
    "tipe_produk": "L", "suhu_udara_K": 301.2, "suhu_proses_K": 310.8, "kecepatan_rpm": 1290,
    "torsi_Nm": 62.5, "keausan_tool_menit": 228, "getaran_mm_s": 4.1, "jam_sejak_servis": 410,
    "shift_operator": "malam",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` menjalankan startup: model di-load sekali
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and len(r.json()["fitur_input"]) == 9


def test_normal_matches_p2_production_test(client):
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probabilitas_gagal"] <= 1
    assert body["status"] in ("SEGERA SERVIS", "AMAN")
    assert body["status"] == ("SEGERA SERVIS" if body["probabilitas_gagal"] >= body["threshold"] else "AMAN")


def test_healthy_machine_is_safe(client):
    sehat = {**NORMAL, "torsi_Nm": 35, "keausan_tool_menit": 10, "getaran_mm_s": 1.0, "jam_sejak_servis": 20}
    assert client.post("/predict", json=sehat).json()["status"] == "AMAN"


def test_extreme_but_valid_still_answers(client):
    ekstrem = {**NORMAL, "kecepatan_rpm": 4000, "torsi_Nm": 150, "keausan_tool_menit": 500, "getaran_mm_s": 50}
    assert client.post("/predict", json=ekstrem).status_code == 200


def test_broken_sensor_null_still_answers(client):
    rusak = {**NORMAL, "torsi_Nm": None, "getaran_mm_s": None}
    assert client.post("/predict", json=rusak).status_code == 200


def test_messy_shift_is_cleaned(client):
    assert client.post("/predict", json={**NORMAL, "shift_operator": " MALAM "}).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("tipe_produk", "X"), ("kecepatan_rpm", 9000), ("torsi_Nm", -1), ("suhu_proses_K", 0),
    ("shift_operator", "subuh"), ("kecepatan_rpm", "cepat"),
])
def test_invalid_is_rejected_422(client, field, value):
    r = client.post("/predict", json={**NORMAL, field: value})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == field


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in NORMAL.items() if k != "torsi_Nm"}
    assert client.post("/predict", json=body).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[NORMAL, {**NORMAL, "tipe_produk": "H"}])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json=[]).json() == []
