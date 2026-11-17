"""Portfolio: Persistenz über Neustarts, PnL-Rechnung, Kennzahlen."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from core.portfolio import Portfolio


def _db(tmp_path):
    return str(tmp_path / 'trades.db')


def test_rundreise_pnl_und_gebuehren(tmp_path):
    p = Portfolio(100, _db(tmp_path))
    p.open_position('BTC_EUR', 'long', 30, 100.0, 1, 's', fees=0.075)
    assert p.balance == 70
    p.update_position_prices({'BTC_EUR': 110.0})
    assert p.equity == pytest.approx(103.0)
    trade = p.close_position('BTC_EUR', 110.0, fees=0.0825, strategy='s')
    assert trade.pnl == pytest.approx(3.0 - 0.1575)
    assert p.balance == pytest.approx(100 + trade.pnl)
    assert p.equity == pytest.approx(p.balance)


def test_positionen_ueberleben_neustart(tmp_path):
    p = Portfolio(100, _db(tmp_path))
    p.open_position('BTC_EUR', 'long', 30, 100.0, 1, 's', market_type='daily_trend')
    q = Portfolio(100, _db(tmp_path))
    assert q.positions['BTC_EUR'].market_type == 'daily_trend'
    assert q.balance == 70 and q.equity == pytest.approx(100)


def test_tagesstart_ueberlebt_neustart_am_selben_tag(tmp_path):
    p = Portfolio(100, _db(tmp_path))
    p.open_position('BTC_EUR', 'long', 30, 100.0, 1, 's')
    p.close_position('BTC_EUR', 60.0, fees=0.0, strategy='s')   # −12 €
    assert p.get_daily_drawdown() == pytest.approx(0.12)
    q = Portfolio(100, _db(tmp_path))
    # Früher: Tagesstart = geladene Equity → Drawdown 0, das Tageslimit war weg
    assert q.get_daily_drawdown() == pytest.approx(0.12)
    assert q.consecutive_losses == 1


def test_tagesstart_von_gestern_gilt_nicht(tmp_path):
    p = Portfolio(100, _db(tmp_path))
    p.open_position('BTC_EUR', 'long', 30, 100.0, 1, 's')
    p.close_position('BTC_EUR', 60.0, fees=0.0, strategy='s')
    p.day_start = (datetime.now() - timedelta(days=1)).date()
    p._save_state()
    q = Portfolio(100, _db(tmp_path))
    assert q.get_daily_drawdown() == pytest.approx(0.0)


def test_sharpe_rechnet_auf_stundenwerten(tmp_path):
    p = Portfolio(100, _db(tmp_path))
    rng = np.random.default_rng(1)
    t0 = datetime(2026, 1, 1)
    eq, hist = 100.0, []
    hourly_returns = rng.normal(0.001, 0.01, 23)
    for h in range(24):
        if h:
            eq *= 1 + hourly_returns[h - 1]
        # viele Zwischenwerte je Stunde — dürfen das Ergebnis nicht verändern
        hist += [(t0 + timedelta(hours=h, seconds=s), eq) for s in range(0, 3600, 600)]
    p.equity_history = hist
    expected = hourly_returns.mean() / hourly_returns.std() * np.sqrt(24 * 365)
    assert p.get_sharpe_ratio() == pytest.approx(expected, rel=1e-6)
