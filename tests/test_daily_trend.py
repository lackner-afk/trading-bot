"""
Tests für die DailyTrendStrategy und ihren Backtester.

Alles auf synthetischen Kursen — keine Netzwerkabhängigkeit. Geprüft wird,
was einen Backtest sonst still zu gut aussehen lässt: Blick in die Zukunft,
Gebühren-Arithmetik, Mindestorder, Notstopp.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.daily_trend import DailyTrendStrategy, TrendParams, compute_state  # noqa: E402
from strategies.crypto_scalper import SignalType  # noqa: E402
from tools.backtest_daily_trend import SimConfig, position_notional, simulate  # noqa: E402


def make_candles(closes, start='2024-01-01', spread=0.01):
    closes = np.asarray(closes, dtype=float)
    dates = pd.date_range(start, periods=len(closes), freq='D')
    opens = np.concatenate([[closes[0]], closes[:-1]])   # Eröffnung = Vortagsschluss
    return pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': np.maximum(opens, closes) * (1 + spread),
        'low': np.minimum(opens, closes) * (1 - spread),
        'close': closes,
        'volume': 1.0,
    })


def trend_series(flat_days=60, up_days=60, down_days=40, base=100.0, up=0.01, down=-0.02):
    """Seitwärts, dann klarer Aufwärtstrend, dann Einbruch."""
    closes = [base] * flat_days
    for _ in range(up_days):
        closes.append(closes[-1] * (1 + up))
    for _ in range(down_days):
        closes.append(closes[-1] * (1 + down))
    return closes


PARAMS = TrendParams(ma_days=50, entry_buffer_pct=0.01, exit_buffer_pct=0.02,
                     require_rising_ma=True, slope_days=10, max_loss_pct=0.08)


# ---------------------------------------------------------------------------
# compute_state
# ---------------------------------------------------------------------------

def test_state_is_causal_no_lookahead():
    rng = np.random.default_rng(7)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    df = make_candles(closes)
    full = compute_state(df, PARAMS)
    for i in (60, 120, 200, 299):
        prefix = compute_state(df.iloc[:i + 1], PARAMS)
        assert bool(prefix['entry_ok'].iloc[-1]) == bool(full['entry_ok'].iloc[i])
        assert bool(prefix['exit_ok'].iloc[-1]) == bool(full['exit_ok'].iloc[i])


def test_entry_then_exit_flags_on_trend_reversal():
    df = make_candles(trend_series())
    st = compute_state(df, PARAMS)
    # Aufwärmphase: nichts
    assert not st['entry_ok'].iloc[:PARAMS.ma_days - 1].any()
    # Im Aufwärtstrend wird eingestiegen ...
    assert st['entry_ok'].iloc[60:120].any()
    # ... und im Einbruch ausgestiegen
    assert st['exit_ok'].iloc[120:].any()
    # Hysterese: nie beides zugleich
    assert not (st['entry_ok'] & st['exit_ok']).any()


def test_rising_ma_filter_blocks_flat_market():
    df = make_candles([100.0] * 80 + [101.5] * 5)  # 1,5 % über flacher SMA, aber SMA steigt kaum
    with_slope = compute_state(df, PARAMS)
    without_slope = compute_state(df, TrendParams(ma_days=50, entry_buffer_pct=0.01,
                                                  require_rising_ma=False))
    assert without_slope['entry_ok'].iloc[-1]
    # Die SMA steigt durch die letzten 5 Kerzen minimal; die Steigungsbedingung
    # darf hier nicht laxer sein als die reine Schwellenbedingung.
    assert with_slope['entry_ok'].sum() <= without_slope['entry_ok'].sum()


# ---------------------------------------------------------------------------
# DailyTrendStrategy
# ---------------------------------------------------------------------------

def test_analyze_emits_spot_long_with_stop():
    strat = DailyTrendStrategy({'ma_days': 50, 'max_loss_pct': 0.08, 'pairs': ['BTC_EUR']})
    df = make_candles(trend_series(down_days=0))
    price = float(df['close'].iloc[-1])
    sig = strat.analyze('BTC_EUR', df, price)
    assert sig is not None
    assert sig.signal_type == SignalType.LONG
    assert sig.suggested_leverage == 1
    assert sig.take_profit == 0.0
    assert math.isclose(sig.stop_loss, price * 0.92)


def test_analyze_returns_none_without_warmup_or_trend():
    strat = DailyTrendStrategy({'ma_days': 50, 'pairs': ['BTC_EUR']})
    assert strat.analyze('BTC_EUR', make_candles([100.0] * 30), 100.0) is None
    assert strat.analyze('BTC_EUR', make_candles([100.0] * 120), 100.0) is None


def test_trend_exit_detected():
    strat = DailyTrendStrategy({'ma_days': 50, 'pairs': ['BTC_EUR']})
    df = make_candles(trend_series())
    should_exit, reason = strat.check_trend_exit('BTC_EUR', df)
    assert should_exit and 'Trend gebrochen' in reason


def test_min_equity_for_trade():
    strat = DailyTrendStrategy({'allocation_pct': 0.20, 'min_order_amount': 25.0})
    assert math.isclose(strat.min_equity_for_trade(0.20), 125.0)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

def test_position_notional_respects_hard_limits():
    cfg = SimConfig(start_capital=1000, allocation_pct=0.5)
    # 50 % gewünscht, aber 20 %-Kappung und 2 %/8 % = 25 % Risiko-Kappung -> 200 €
    assert math.isclose(position_notional(1000, 1000, PARAMS, cfg), 200.0)
    # Enger Notstopp: 2 % / 4 % = 50 %, dann greift die 20 %-Kappung
    tight = TrendParams(ma_days=50, max_loss_pct=0.04)
    assert math.isclose(position_notional(1000, 1000, tight, cfg), 200.0)
    # Weiter Notstopp: 2 % / 20 % = 10 % -> die Risiko-Regel ist bindend
    wide = TrendParams(ma_days=50, max_loss_pct=0.20)
    assert math.isclose(position_notional(1000, 1000, wide, cfg), 100.0)


def test_simulate_single_round_trip_fee_accounting():
    df = make_candles(trend_series(), spread=0.0)
    cfg = SimConfig(start_capital=1000.0, fee_pct=0.0025, slippage_pct=0.0,
                    min_order_amount=25.0, allocation_pct=0.20)
    params = TrendParams(ma_days=50, entry_buffer_pct=0.01, exit_buffer_pct=0.02,
                         require_rising_ma=True, max_loss_pct=0.50)  # Stop weit weg
    res = simulate({'BTC': df}, params, cfg)

    assert res.metrics['n_trades'] == 1
    t = res.trades[0]
    assert t['reason'] == 'trend'

    # Kein Blick in die Zukunft: Einstieg einen Tag NACH dem ersten Signal
    st = compute_state(df.assign(date=df['timestamp']).set_index('date'), params)
    first_signal = st.index[st['entry_ok']][0]
    assert t['entry_date'] == first_signal + pd.Timedelta(days=1)
    entry_row = df[df['timestamp'] == t['entry_date']].iloc[0]
    assert math.isclose(t['entry_price'], float(entry_row['open']))

    # Gebühren-Arithmetik auf den Cent
    notional = 1000.0 * min(0.20, 0.02 / 0.50)   # 2 %/50 % = 4 % -> 40 €
    assert math.isclose(t['notional'], notional)
    qty = notional / t['entry_price']
    proceeds = qty * t['exit_price']
    expected_fees = notional * 0.0025 + proceeds * 0.0025
    assert math.isclose(t['fees'], expected_fees, rel_tol=1e-9)
    assert math.isclose(t['pnl'], proceeds - notional - expected_fees, rel_tol=1e-9)
    assert math.isclose(res.equity_curve.iloc[-1], 1000.0 + t['pnl'], rel_tol=1e-9)


def test_min_order_blocks_small_account_but_not_larger_one():
    df = make_candles(trend_series(down_days=0))
    params = TrendParams(ma_days=50, max_loss_pct=0.08)
    small = simulate({'BTC': df}, params, SimConfig(start_capital=100.0))
    assert small.metrics['n_trades'] == 0
    assert small.skipped_min_order > 0
    assert math.isclose(small.equity_curve.iloc[-1], 100.0)

    larger = simulate({'BTC': df}, params, SimConfig(start_capital=250.0))
    assert larger.skipped_min_order == 0
    assert larger.exposure > 0


def test_intraday_stop_is_honoured():
    closes = trend_series(flat_days=60, up_days=30, down_days=0)
    df = make_candles(closes, spread=0.0)
    # Ein Tag mit Crash-Tief weit unter dem Notstopp, Schluss aber wieder oben
    crash_day = len(df) - 5
    df.loc[crash_day, 'low'] = df.loc[crash_day, 'close'] * 0.70
    params = TrendParams(ma_days=50, max_loss_pct=0.08)
    res = simulate({'BTC': df}, params, SimConfig(start_capital=1000.0, slippage_pct=0.0))
    stops = [t for t in res.trades if t['reason'] == 'stop']
    assert len(stops) == 1
    t = stops[0]
    assert math.isclose(t['exit_price'], t['entry_price'] * 0.92, rel_tol=1e-9)
    assert t['exit_date'] == df.loc[crash_day, 'timestamp']


def test_windows_do_not_leak_history_but_use_it_for_warmup():
    df = make_candles(trend_series(flat_days=80, up_days=80, down_days=0))
    params = TrendParams(ma_days=50, max_loss_pct=0.10)   # 2 %/10 % = 20 % -> Kappung greift exakt
    cfg = SimConfig(start_capital=1000.0)
    start = df['timestamp'].iloc[100]
    res = simulate({'BTC': df}, params, cfg, start=start)
    # Kurve beginnt am Fensteranfang, Signal war schon vorher aufgewärmt -> sofort investiert
    assert res.equity_curve.index[0] == start
    assert len(res.equity_curve) == len(df) - 100
    assert res.trades == []           # Position bleibt offen bis zum Ende
    assert res.exposure > 0.15        # ~20 % investiert, mehr erlaubt der RiskManager nicht


def test_buy_and_hold_reference_matches_manual():
    df = make_candles([100.0] * 10 + [120.0] * 10, spread=0.0)
    cfg = SimConfig(start_capital=1000.0, fee_pct=0.0, slippage_pct=0.0)
    res = simulate({'BTC': df}, TrendParams(ma_days=5), cfg)
    # Eröffnung Tag 1 = 100, Schluss letzter Tag = 120
    assert math.isclose(res.buy_hold_return, 0.20)
