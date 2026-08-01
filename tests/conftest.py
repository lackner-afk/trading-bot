"""
Gemeinsame Test-Fixtures.

Wichtigste Regel hier: Tests gehen nie ins Netz. Der SentimentFactor macht
in calculate() einen echten HTTP-Call gegen alternative.me — ohne den Block
wären Tests langsam, flaky und vom Fear-&-Greed-Index des Tages abhängig.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo-Wurzel importierbar machen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.portfolio import Portfolio  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Blockt jeden ausgehenden HTTP-Verkehr in Tests."""

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Netzwerkzugriff im Test - bitte mocken statt echte API aufrufen"
        )

    import requests
    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    monkeypatch.setattr("socket.socket.connect", _blocked)


@pytest.fixture
def portfolio(tmp_path):
    """
    Frisches Portfolio mit eigener DB im tmp_path.

    Ohne db_path legt Portfolio eine trades.db im aktuellen Arbeitsverzeichnis
    an und Tests würden sich gegenseitig den Zustand zerschießen.
    """
    return Portfolio(
        start_capital=100.0,
        db_path=str(tmp_path / "test_trades.db"),
        snapshot_interval_seconds=0,  # in Tests jeden Snapshot schreiben
    )


@pytest.fixture
def sample_candles():
    """
    Deterministischer OHLCV-DataFrame mit Indikatoren (200 5m-Kerzen).

    Aufwärtstrend mit etwas Rauschen — genug Historie für EMA21/BB20/RSI14
    und für den RegimeDetector (braucht >= 40 Kerzen).
    """
    rng = np.random.default_rng(42)
    n = 200
    base = 50_000.0
    drift = np.linspace(0, 2_000, n)
    noise = rng.normal(0, 120, n).cumsum()
    close = base + drift + noise

    idx = [datetime(2026, 1, 1) + timedelta(minutes=5 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - rng.normal(0, 20, n),
            "high": close + np.abs(rng.normal(60, 20, n)),
            "low": close - np.abs(rng.normal(60, 20, n)),
            "close": close,
            "volume": np.abs(rng.normal(1_000, 200, n)),
        }
    )

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["rsi"] = df["rsi"].fillna(50.0)

    df["bb_middle"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * std
    df["bb_lower"] = df["bb_middle"] - 2 * std
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["volume_delta"] = df["volume"].diff().fillna(0.0)

    return df.bfill()
