# Ein Image für Bot, Datensammler und Dashboard — gebaut aus dem Repo-Wurzelverzeichnis:
#   docker build -f deploy/vps/Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Vienna

WORKDIR /app

# Abhängigkeiten zuerst — die Schicht bleibt im Cache, solange sich nur Code ändert
COPY deploy/vps/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY config/settings.yaml config/settings.yaml
COPY core core
COPY data data
COPY strategies strategies
COPY notifications notifications
COPY tools/data_recorder.py tools/data_recorder.py
COPY dashboard dashboard
COPY main.py main.py

# Nicht als root laufen; /state ist das gemeinsame Volume
RUN useradd --uid 1000 --create-home bot && mkdir -p /state && chown bot:bot /state
USER bot

ENV BOT_STATE_DIR=/state
CMD ["python3", "-u", "main.py"]
