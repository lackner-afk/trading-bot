"""
Integrationstest: TradingBot + DailyTrendStrategy mit einem Fake-Feed.

Prüft die Verdrahtung in main.py ohne Netzwerk: Kauf bei intaktem Trend,
kein Kauf unter der Mindestorder, Ausstieg bei Trendbruch, Notstopp im
Haupt-Loop, laufende Tageskerze wird ignoriert.
"""

import asyncio
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as bot_main  # noqa: E402


class FakeFeed:
    """Liefert vorgegebene Tageskerzen und einen festen Preis."""

    def __init__(self, candles_by_symbol, price_by_symbol):
        self.candles = candles_by_symbol
        self.prices = price_by_symbol
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    def get_candles(self, symbol, timeframe='1m', n=50):
        df = self.candles.get(symbol)
        if df is None or timeframe != '1d':
            return None
        return df.tail(n)

    def get_price(self, symbol, max_age_seconds=None):
        return self.prices.get(symbol)

    def get_prices(self, max_age_seconds=None):
        return dict(self.prices)

    def get_price_age(self, symbol):
        return 0.0


def daily_candles(closes, end_today=True):
    """Tageskerzen, deren letzte Kerze der HEUTIGE (noch laufende) Tag ist."""
    closes = np.asarray(closes, dtype=float)
    today = pd.Timestamp(datetime.utcnow().date())
    start = today - pd.Timedelta(days=len(closes) - 1)
    dates = pd.date_range(start, periods=len(closes), freq='D')
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        'timestamp': dates, 'open': opens,
        'high': np.maximum(opens, closes) * 1.005,
        'low': np.minimum(opens, closes) * 0.995,
        'close': closes, 'volume': 1.0,
    })


def uptrend(n_flat=60, n_up=80, up=0.01):
    closes = [100.0] * n_flat
    for _ in range(n_up):
        closes.append(closes[-1] * (1 + up))
    return closes


def broken_trend():
    closes = uptrend()
    for _ in range(40):
        closes.append(closes[-1] * 0.97)
    return closes


@pytest.fixture
def bot_factory(tmp_path, monkeypatch):
    """Baut einen TradingBot im Temp-Verzeichnis mit angepasster Config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', '')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', '')

    def make(start_capital, ma_days=50):
        cfg = yaml.safe_load((ROOT / 'config' / 'settings.yaml').read_text())
        cfg['general']['start_capital'] = start_capital
        cfg['general']['data_feed'] = 'kraken'
        cfg['notifications']['telegram']['enabled'] = False
        cfg['strategies']['daily_trend'].update({
            'enabled': True, 'pairs': ['BTC_EUR'], 'ma_days': ma_days,
            'interval_seconds': 1,
        })
        cfg_path = tmp_path / 'settings.yaml'
        cfg_path.write_text(yaml.safe_dump(cfg))
        if (tmp_path / 'trades.db').exists():
            (tmp_path / 'trades.db').unlink()
        bot = bot_main.TradingBot(config_path=str(cfg_path))
        return bot

    return make


async def run_loop_once(bot):
    bot.running = True
    task = asyncio.create_task(bot._daily_trend_loop())
    await asyncio.sleep(0.5)
    bot.running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_config_wires_daily_trend_and_disables_confluence(bot_factory):
    bot = bot_factory(start_capital=250)
    assert bot.daily_trend is not None
    assert bot.daily_trend.pairs == ['BTC_EUR']
    assert bot.confluence_strategy is None
    assert bot.crypto_feed.pairs == ['BTC_EUR']
    assert bot.order_engine.fees['crypto_taker'] == pytest.approx(0.0025)


def test_buys_spot_when_trend_is_intact(bot_factory):
    bot = bot_factory(start_capital=250)
    closes = uptrend()
    price = closes[-1]
    bot.crypto_feed = FakeFeed({'BTC_EUR': daily_candles(closes)}, {'BTC_EUR': price})

    asyncio.run(run_loop_once(bot))

    pos = bot.portfolio.positions.get('BTC_EUR')
    assert pos is not None
    assert pos.side == 'long' and pos.leverage == 1 and pos.market_type == 'daily_trend'
    # 20 %-Kappung des RiskManagers: 250 € -> 50 € Notional
    assert pos.size == pytest.approx(50.0, rel=0.02)
    assert pos.stop_loss == pytest.approx(price * 0.92, rel=0.01)
    assert pos.take_profit is None
    # Spot: Margin = Notional, Gebühr 0,25 % gebucht
    assert pos.entry_fees == pytest.approx(pos.size * 0.0025, rel=1e-6)


def test_no_buy_below_min_order(bot_factory, caplog):
    bot = bot_factory(start_capital=100)
    closes = uptrend()
    bot.crypto_feed = FakeFeed({'BTC_EUR': daily_candles(closes)}, {'BTC_EUR': closes[-1]})

    asyncio.run(run_loop_once(bot))

    assert 'BTC_EUR' not in bot.portfolio.positions
    assert any('Mindestorder' in r.getMessage() for r in caplog.records)


def test_running_candle_is_ignored(bot_factory):
    """Die heutige Kerze ist noch offen — sie darf das Signal nicht liefern."""
    bot = bot_factory(start_capital=250)
    closes = [100.0] * 140
    closes[-1] = 130.0   # nur die laufende Kerze schießt hoch
    bot.crypto_feed = FakeFeed({'BTC_EUR': daily_candles(closes)}, {'BTC_EUR': 130.0})

    asyncio.run(run_loop_once(bot))

    assert 'BTC_EUR' not in bot.portfolio.positions


def test_exits_when_trend_breaks(bot_factory):
    bot = bot_factory(start_capital=250)
    closes = broken_trend()
    price = closes[-1]
    bot.crypto_feed = FakeFeed({'BTC_EUR': daily_candles(closes)}, {'BTC_EUR': price})
    bot.portfolio.open_position(
        symbol='BTC_EUR', side='long', size=50.0, price=price * 1.3, leverage=1,
        strategy='daily_trend', market_type='daily_trend', stop_loss=None, fees=0.125,
    )

    asyncio.run(run_loop_once(bot))

    assert 'BTC_EUR' not in bot.portfolio.positions
    trade = bot.portfolio.trades[-1]
    assert trade.strategy == 'daily_trend'
    # Exit-Gebühr aus der Config (0,25 %) plus Entry-Gebühr
    assert trade.fees == pytest.approx(50.0 * 0.0025 + 0.125, rel=1e-6)
    assert trade.pnl < 0


def test_emergency_stop_in_main_loop(bot_factory):
    bot = bot_factory(start_capital=250)
    bot.crypto_feed = FakeFeed({}, {'BTC_EUR': 90.0})
    bot.portfolio.open_position(
        symbol='BTC_EUR', side='long', size=50.0, price=100.0, leverage=1,
        strategy='daily_trend', market_type='daily_trend', stop_loss=92.0, fees=0.125,
    )

    asyncio.run(bot._check_exit_conditions({'BTC_EUR': 90.0}))

    assert 'BTC_EUR' not in bot.portfolio.positions
    assert bot.portfolio.trades[-1].exit_price == 90.0

    # Über dem Stopp passiert nichts — kein Take-Profit, kein Trailing
    bot.portfolio.open_position(
        symbol='BTC_EUR', side='long', size=50.0, price=100.0, leverage=1,
        strategy='daily_trend', market_type='daily_trend', stop_loss=92.0, fees=0.125,
    )
    asyncio.run(bot._check_exit_conditions({'BTC_EUR': 250.0}))
    assert 'BTC_EUR' in bot.portfolio.positions
