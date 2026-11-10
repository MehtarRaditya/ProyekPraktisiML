# Satu image untuk API (default) atau UI. "Then we'll ship your machine."
#   docker build -t mesin-cnc .
#   docker run -p 7860:7860 mesin-cnc                         -> FastAPI di http://localhost:7860/docs
#   docker run -p 7860:7860 -e APP=ui mesin-cnc               -> Streamlit di http://localhost:7860
# Hugging Face Spaces (Docker SDK) memakai port 7860.
FROM python:3.12-slim

WORKDIR /code
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY artifacts ./artifacts
COPY data/contoh_batch.csv ./data/contoh_batch.csv

ENV PORT=7860 APP=api
EXPOSE 7860
RUN useradd -m appuser
USER appuser
CMD if [ "$APP" = "ui" ]; then \
      streamlit run app/ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0; \
    else \
      uvicorn app.api.main:app --host 0.0.0.0 --port $PORT; \
    fi
