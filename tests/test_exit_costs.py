"""Ausstiegskosten im Paper-Modus: Gebühr auf den Verkaufswert, Slippage auch beim Verkauf."""
import pytest

from tests.conftest import run, long_signal


def test_slippage_abschnitt_wird_gelesen(make_bot):
    bot = make_bot({'slippage': {'min': 0.002, 'max': 0.003}})
    assert bot.order_engine.slippage_min == 0.002
    assert bot.order_engine.slippage_max == 0.003


def test_exit_gebuehr_auf_verkaufswert(make_bot):
    bot = make_bot({'general': {'start_capital': 100}, 'slippage': {'min': 0.0, 'max': 0.0}})
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    pos = bot.portfolio.positions['BTC_EUR']
    assert pos.size == 30.0 and pos.entry_price == 50000.0
    run(bot._close_position('BTC_EUR', 75000.0, 'Test'))   # +50 %
    trade = bot.portfolio.trades[-1]
    # Einstieg 30 € · 0,25 % + Verkauf 45 € · 0,25 %
    assert trade.fees == pytest.approx(30 * 0.0025 + 45 * 0.0025)
    assert trade.pnl == pytest.approx(15.0 - trade.fees)


def test_exit_bekommt_slippage(make_bot):
    bot = make_bot({'general': {'start_capital': 100}, 'slippage': {'min': 0.001, 'max': 0.001}})
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    assert bot.portfolio.positions['BTC_EUR'].entry_price == pytest.approx(50000 * 1.001)
    run(bot._close_position('BTC_EUR', 50000.0, 'Test'))
    assert bot.portfolio.trades[-1].exit_price == pytest.approx(50000 * 0.999)


def test_rundreise_ohne_kursbewegung_kostet_gebuehr_plus_slippage(make_bot):
    """Kurs unverändert: Verlust = 2 × 0,25 % Gebühr + 2 × Slippage auf 30 €."""
    bot = make_bot({'general': {'start_capital': 100}, 'slippage': {'min': 0.0005, 'max': 0.0005}})
    run(bot._execute_signal(long_signal(), strategy_name='daily_trend'))
    run(bot._close_position('BTC_EUR', 50000.0, 'Test'))
    pnl = bot.portfolio.trades[-1].pnl
    assert pnl == pytest.approx(-30 * (0.005 + 0.001), rel=0.02)
    assert bot.portfolio.equity == pytest.approx(100 + pnl)
