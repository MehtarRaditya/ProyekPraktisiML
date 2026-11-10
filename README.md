# Machine Learning · MKB UPI · P3 Ship

Bahan hands-on P3 seri kuliah praktisi **From notebook to production**: model dari P2 kita kasih API dan
tampilan, lalu di-deploy supaya bisa dipakai teknisi pabrik lewat link.

**Kasus:** pabrik komponen otomotif ingin tahu mesin CNC mana yang kemungkinan **gagal dalam 24 jam**
(predictive maintenance), supaya bisa diservis duluan.

```
P2 · Act (Colab, repo mkb-ml-p2)        P3 · Ship (repo ini)
notebook ──► artifact_p2.zip ──► artifacts/ + src/ ──► FastAPI /predict ──► deploy
             model.joblib                        └──► Streamlit UI ──────► link live
             metadata.json
             src/features.py
```

Daftar isi:
[sebelum mulai](#sebelum-mulai-artifact-dari-p2) ·
[siapkan repo](#p3--ship-1-siapkan-repo) ·
[jalankan API](#p3--ship-2-jalankan-fastapi) ·
[jalankan UI](#p3--ship-3-jalankan-streamlit) ·
[checklist](#p3--ship-4-checklist-sebelum-online) ·
[deploy](#p3--ship-5-deploy-gratis) ·
[Docker](#docker) ·
[error umum](#error-yang-hampir-pasti-kalian-temui)

---

## Sebelum mulai: artifact dari P2

Di akhir P2 kalian mengunduh **`artifact_p2.zip`**. Isinya tiga file, dan cuma itu yang dibutuhkan untuk deploy:

```
artifacts/model.joblib     pipeline lengkap: fitur + preprocessing + model
artifacts/metadata.json    versi model, fitur_input, threshold, metrik, versi library
src/features.py            fitur fisika; dipakai saat training DAN saat prediksi
```

Zip-nya hilang atau P2 belum selesai? Buka lagi notebook P2 di
[Colab](https://colab.research.google.com/github/nurhikam/mkb-ml-p2/blob/main/notebooks/P2_Act_STUDENT.ipynb).
Atau langsung lanjut saja: repo ini sudah membawa **contoh artifact** di `artifacts/`, jadi P3 tetap bisa
diikuti sampai akhir.

---

## P3 · Ship 1: siapkan repo

**Fork** repo ini ke akun GitHub kalian (tombol *Fork* kanan atas), lalu clone fork-nya:

```bash
git clone https://github.com/<username>/mkb-ml-p3.git
cd mkb-ml-p3
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                             # 14 tes harus lolos dengan contoh artifact
```

> Tidak mau install apa pun di laptop? Lewati bagian lokal: langsung ke [deploy](#p3--ship-5-deploy-gratis).
> Streamlit Community Cloud bisa deploy langsung dari fork kalian.

### Pakai model kalian sendiri

1. Ekstrak `artifact_p2.zip` **di root repo**. Path di dalam zip sudah cocok, jadi file lama tertimpa:
   `artifacts/model.joblib`, `artifacts/metadata.json`, `src/features.py`.
2. Buka `artifacts/metadata.json` → `versi_library`, lalu samakan versi di `requirements.txt`:

   ```json
   "versi_library": {"python": "3.12.x", "sklearn": "1.6.1", "xgboost": "2.1.4", "pandas": "2.2.2"}
   ```
   ```text
   scikit-learn==1.6.1
   xgboost-cpu==2.1.4
   pandas==2.2.2
   ```
   Model yang di-load dengan versi scikit-learn berbeda bisa memberi hasil lain tanpa error yang jelas.
   Itu sebabnya versinya di-pin, bukan dibiarkan "terbaru".
3. `pip install -r requirements-dev.txt` lagi, lalu `pytest`. Kalau lolos, model kalian siap.
4. Commit dan push:

   ```bash
   git add artifacts src requirements.txt
   git commit -m "model P2 saya"
   git push
   ```

### Struktur repo

| Path | Isi |
|---|---|
| `data/contoh_batch.csv` | 12 mesin untuk mencoba prediksi batch di UI |
| `artifacts/` | `model.joblib` + `metadata.json` (contoh; ganti dengan punya kalian) |
| `src/features.py` | fitur fisika, dipakai training dan serving |
| `app/predictor.py` | load artifact sekali, lalu prediksi (dipakai API dan UI) |
| `app/api/main.py` | FastAPI: `/predict`, `/predict/batch`, `/health`, `/docs` |
| `app/ui/streamlit_app.py` | UI Streamlit: form, hasil berwarna, JSON, upload CSV |
| `tests/test_api.py` | smoke test: input normal, ekstrem, tidak valid |
| `Dockerfile` | satu image untuk API (default) atau UI |
| `requirements.txt` | versi library di-pin sama dengan saat training |

---

## P3 · Ship 2: jalankan FastAPI

```bash
uvicorn app.api.main:app --reload
```

Buka **http://127.0.0.1:8000/docs**. Swagger UI ini dibuat otomatis oleh FastAPI: klik `POST /predict` →
*Try it out* → *Execute*.

Satu endpoint FastAPI berisi tiga bagian (lihat `app/api/main.py`):

1. **Load sekali saat startup**, bukan per request (`lifespan` → `predictor.load()`).
2. **Validasi** dengan Pydantic (`InputData`): `tipe_produk` harus L/M/H, `kecepatan_rpm` 500–4000, torsi tidak
   boleh negatif, dst. Input yang salah ditolak **422** sebelum menyentuh model.
3. **JSON keluar** dari `POST /predict`.

Request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "tipe_produk": "L", "suhu_udara_K": 301.2, "suhu_proses_K": 310.8, "kecepatan_rpm": 1290,
  "torsi_Nm": 62.5, "keausan_tool_menit": 228, "getaran_mm_s": 4.1, "jam_sejak_servis": 410,
  "shift_operator": "malam"}'
```

Response:

```json
{"probabilitas_gagal": 0.9781, "status": "SEGERA SERVIS", "threshold": 0.16, "versi_model": "1.0.0", "latency_ms": 12.9}
```

Coba kirim `"tipe_produk": "X"` atau `"kecepatan_rpm": 9000`: hasilnya **422 Unprocessable Entity**, dan field
`detail` menyebut field mana yang salah. Itu validasi yang sedang bekerja, bukan bug.

Sensor rusak boleh dikirim sebagai `null`; pipeline mengisinya dengan median data training.

| Endpoint | Fungsi |
|---|---|
| `GET /health` | cek server hidup + versi model, threshold, daftar fitur |
| `POST /predict` | satu mesin |
| `POST /predict/batch` | list mesin (maks 1000 per request) |
| `GET /docs` | dokumentasi interaktif |

Setiap prediksi di-log ke terminal (input + output) supaya bisa diaudit.

---

## P3 · Ship 3: jalankan Streamlit

```bash
streamlit run app/ui/streamlit_app.py
```

Buka **http://localhost:8501**. Isinya form kondisi mesin, hasil berwarna (merah = SEGERA SERVIS, hijau = AMAN),
tampilan request/response JSON, dan tab **Banyak mesin (CSV)** untuk upload `data/contoh_batch.csv`.

| | Mode A (default) | Mode B (poin bonus) |
|---|---|---|
| Cara pakai | `API_URL` kosong | `API_URL=https://...` |
| Model jalan di | Streamlit itu sendiri | FastAPI terpisah |
| Deploy | 1 (Streamlit Cloud) | 2 (UI + API) |
| Cocok untuk | demo, hands-on hari ini | arsitektur industri: UI dan model di-scale terpisah |

Mode B lokal (API dan UI di dua terminal):

```bash
uvicorn app.api.main:app --port 8000                                  # terminal 1
API_URL=http://127.0.0.1:8000 streamlit run app/ui/streamlit_app.py   # terminal 2
```

---

## P3 · Ship 4: checklist sebelum online

| # | Cek | Di repo ini |
|---|---|---|
| 1 | Model mengalahkan baseline di test set | `metadata.json` → `metrik_test` (PR-AUC 0.695 vs acak ~0.055) |
| 2 | Metadata lengkap | `artifacts/metadata.json` |
| 3 | Versi library di-pin = saat training | `requirements.txt` vs `metadata.json` → `versi_library` |
| 4 | Fitur diambil dari `src/features.py` | pipeline memanggil `tambah_fitur` dari sana |
| 5 | Input divalidasi Pydantic | `InputData` di `app/api/main.py` |
| 6 | Ada endpoint `/health` | `GET /health` |
| 7 | Setiap prediksi di-log | `_log()` di `app/api/main.py` |
| 8 | Smoke test: normal, ekstrem, tidak valid | `pytest` (`tests/test_api.py`) |

---

## P3 · Ship 5: deploy gratis

Syarat free tier sering berubah; cek lagi sebelum dipakai.

### A. Streamlit Community Cloud (UI, paling gampang; ini yang dipakai di kelas)

1. Pastikan `artifacts/model.joblib` sudah ter-push ke fork kalian (buka di GitHub dan lihat apakah file-nya ada).
2. Buka **https://share.streamlit.io** dan login dengan GitHub.
3. **Create app** → *Deploy a public app from GitHub*.
4. Isi: repository `<username>/mkb-ml-p3`, branch `main`, main file path **`app/ui/streamlit_app.py`**.
5. **Advanced settings** → Python version: samakan dengan `versi_library.python` di metadata (contoh: 3.12).
6. **Deploy**. Build pertama butuh beberapa menit karena menginstal `requirements.txt`.
7. Smoke test dari link-nya: input normal, input ekstrem (rpm 4000, keausan 500), lalu upload CSV.
8. Tempel link app kalian di chat kelas.

### B. Hugging Face Spaces (API dengan Docker)

1. **https://huggingface.co/new-space** → nama bebas, SDK **Docker** → *Blank* → Create.
2. Clone Space-nya, salin isi repo ini ke dalamnya, lalu push:

   ```bash
   git clone https://huggingface.co/spaces/<username>/<nama-space>
   cp -r app src artifacts requirements.txt Dockerfile ../<nama-space>/
   mkdir -p ../<nama-space>/data && cp data/contoh_batch.csv ../<nama-space>/data/
   cd ../<nama-space> && git add . && git commit -m "API prediksi mesin" && git push
   ```
   Jangan timpa `README.md` milik Space: header YAML di dalamnya (`sdk: docker`) yang memberi tahu
   Hugging Face cara menjalankan Space. Dockerfile repo ini sudah memakai port 7860, port bawaan Spaces.
3. Setelah status *Running*, API ada di `https://<username>-<nama-space>.hf.space/docs`.
4. **Mode B:** di Streamlit Cloud → app kalian → **Settings → Secrets**, isi:

   ```toml
   API_URL = "https://<username>-<nama-space>.hf.space"
   ```
   Simpan; app restart dan banner atas berubah jadi *Mode B · memanggil API*.

### C. Render (API dari repo)

1. **https://render.com** → *New* → *Web Service* → pilih fork kalian.
2. Runtime **Docker** (Render mendeteksi `Dockerfile` otomatis), instance type *Free* → Deploy.
3. API ada di `https://<nama>.onrender.com/docs`. Free tier **tidur** kalau lama tidak dipakai, jadi request
   pertama setelah sepi bisa butuh ~1 menit. UI Streamlit sudah menunggu sampai 60 detik.

---

## Docker

"Works on my machine." "Then we'll ship your machine." Docker mengirim seluruh environment (versi Python
dan library), bukan cuma file `.joblib`.

```bash
docker build -t mesin-cnc .
docker run -p 7860:7860 mesin-cnc                 # FastAPI  → http://localhost:7860/docs
docker run -p 7860:7860 -e APP=ui mesin-cnc       # Streamlit → http://localhost:7860
```

---

## Error yang hampir pasti kalian temui

| Error | Penyebab | Solusi |
|---|---|---|
| `No module named 'src'` | folder `src/` tidak ada, atau dijalankan bukan dari root repo | pastikan `src/features.py` ada; jalankan perintah dari root repo |
| `InconsistentVersionWarning` | versi scikit-learn beda dengan saat training | samakan `requirements.txt` dengan `metadata.json` → `versi_library` |
| `KeyError: 'torsi_Nm'` | nama kolom/field beda | cocokkan dengan `fitur_input` di `metadata.json` |
| `422 Unprocessable Entity` | input gagal validasi | baca field `detail`: ia menyebut field dan alasannya |
| `model.joblib tidak ada` di server | artifact tidak ter-commit | cek file-nya ada di GitHub; jangan masukkan `artifacts/` ke `.gitignore` |
| `AttributeError: Can't get attribute 'tambah_fitur'` | `src/features.py` beda dengan saat training | pakai `src/features.py` dari `artifact_p2.zip` yang sama dengan modelnya |

Setiap pesan error adalah petunjuk. Mentok? Tanya di QnA kelas dan **tempel pesan error lengkapnya**.

---

## Tentang data

Datanya **sintetis**: data pabrik asli hampir selalu rahasia. Strukturnya meniru data sensor mesin perkakas
(terinspirasi dataset AI4I 2020) dan sengaja dikotori: duplikat, nilai sensor mustahil, nilai kosong, typo
kategori. Dataset dan skrip pembuatnya ada di repo [mkb-ml-p2](https://github.com/nurhikam/mkb-ml-p2).
