"""Gemeinsame Test-Helfer: ein TradingBot ohne Netzwerk, mit eigener Datenbank."""
import asyncio
import logging
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeFeed:
    """Feed-Ersatz: feste Preise, kein Netzwerk."""

    def __init__(self, prices):
        self.prices = dict(prices)
        self.min_order_amount = {}

    def get_prices(self, max_age_seconds=None):
        return dict(self.prices)

    def get_price(self, symbol, max_age_seconds=None):
        return self.prices.get(symbol)

    def get_price_age(self, symbol):
        return 0.0


@pytest.fixture
def make_bot(tmp_path, monkeypatch):
    """Baut einen Paper-Bot aus settings.yaml plus Abweichungen (immer mode: paper)."""
    import main

    def _make(overrides=None, prices=None):
        config = yaml.safe_load((ROOT / 'config' / 'settings.yaml').read_text())
        config = main._deep_merge(config, {
            'general': {'mode': 'paper', 'data_feed': 'kraken'},
            'dashboard': {'enabled': False},
            'notifications': {'console': False, 'telegram': {'enabled': False}},
        })
        config = main._deep_merge(config, overrides or {})
        path = tmp_path / 'settings.yaml'
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(main, 'STATE_DIR', tmp_path)
        monkeypatch.delenv('BOT_CONFIG_OVERLAY', raising=False)
        monkeypatch.setattr(main.TradingBot, '_setup_logging', lambda self: None)
        bot = main.TradingBot(config_path=str(path))
        bot.crypto_feed = FakeFeed(prices or {'BTC_EUR': 50000.0, 'ETH_EUR': 2000.0, 'SOL_EUR': 100.0})
        # Simulierte Latenz abschalten — Tests sollen nicht schlafen
        bot.order_engine.latency_min_ms = bot.order_engine.latency_max_ms = 0
        return bot

    return _make


def run(coro):
    return asyncio.run(coro)


def long_signal(symbol='BTC_EUR', price=50000.0, stop_loss=0.0, take_profit=0.0):
    from strategies.crypto_scalper import ScalperSignal, SignalType
    return ScalperSignal(
        signal_type=SignalType.LONG, symbol=symbol, price=price, confidence=1.0,
        reason='Test', take_profit=take_profit, stop_loss=stop_loss, suggested_leverage=1,
    )
