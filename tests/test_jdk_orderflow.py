"""Tests für die JDK-Orderflow-Strategie mit konstruierten Kerzen"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.jdk_orderflow import (  # noqa: E402
    JDKOrderflowStrategy, VolumeProfile, estimate_delta, naked_pocs, volume_profile,
)
from strategies.crypto_scalper import SignalType  # noqa: E402


def _candles(closes, spread=0.4, volume=10.0, start='2026-09-01'):
    """Kerzen aus Schlusskursen: Open = vorheriger Close, Docht ± spread"""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        'timestamp': pd.date_range(start, periods=len(closes), freq='1h'),
        'open': opens,
        'high': np.maximum(opens, closes) + spread,
        'low': np.minimum(opens, closes) - spread,
        'close': closes,
        'volume': np.full(len(closes), volume),
    })


def _balance_range(n=220, center=100.0, amp=4.0, seed=1):
    """Seitwärts-Range um `center` — die meiste Zeit nahe der Mitte"""
    rng = np.random.default_rng(seed)
    return center + amp * np.sin(np.arange(n) / 12.0) * rng.uniform(0.3, 1.0, n)


def _failed_auction(side='long'):
    """Range, Fehlausbruch knapp unter die VAL, Rückeroberung mit Kaufdelta"""
    base = _candles(_balance_range(), spread=0.15)
    val = volume_profile(base).val
    closes = list(base['close']) + [val - 0.3, val - 0.8, val - 1.0, val - 0.7]
    df = _candles(closes, spread=0.15)
    # Reclaim: zwei kräftige Kerzen, Schluss am Hoch → positives Delta
    for c in (val + 0.3, val + 0.6):
        prev = df['close'].iloc[-1]
        df.loc[len(df)] = {
            'timestamp': df['timestamp'].iloc[-1] + pd.Timedelta(hours=1),
            'open': prev, 'high': c + 0.05, 'low': prev - 0.2, 'close': c, 'volume': 25.0,
        }
    if side == 'short':
        # Spiegelung um 200: aus dem Fehlausbruch nach unten wird einer nach oben
        mirrored = df.copy()
        mirrored['open'] = 200 - df['open']
        mirrored['close'] = 200 - df['close']
        mirrored['high'] = 200 - df['low']
        mirrored['low'] = 200 - df['high']
        return mirrored
    return df


def test_volume_profile_poc_at_volume_cluster():
    df = _candles([100.0] * 50 + [110.0] * 5 + [90.0] * 5)
    prof = volume_profile(df, bins=40)
    assert prof is not None
    assert abs(prof.poc - 100.0) < 1.0
    assert prof.val <= 100.0 <= prof.vah
    assert prof.range_low < prof.val and prof.vah < prof.range_high


def test_estimate_delta_sign():
    df = pd.DataFrame({'open': [10, 10], 'high': [11, 11], 'low': [9, 9],
                       'close': [11, 9], 'volume': [5, 5]})
    delta = estimate_delta(df)
    assert delta.iloc[0] == pytest.approx(5.0)
    assert delta.iloc[1] == pytest.approx(-5.0)


def test_naked_poc_only_untouched_days():
    # Tag 1 handelt bei 100, Tag 2 bei 120 — der Tag-1-POC wird danach nie wieder erreicht
    df = _candles([100.0] * 23 + [120.0] * 25 + [121.0] * 5, spread=0.2)
    npocs = naked_pocs(df)
    assert any(abs(p - 100.0) < 1.0 for p in npocs)


def test_failed_auction_reclaim_gives_long():
    strat = JDKOrderflowStrategy({'min_rr': 1.5})
    df = _failed_auction('long')
    sig = strat.evaluate('BTC_EUR', df)
    assert sig is not None
    assert sig.side == 'long' and sig.setup == 'failed_auction'
    assert sig.stop_loss < df['low'].iloc[-8:].min() < sig.entry < sig.take_profit
    assert sig.rr >= 1.5


def test_short_is_mirror_of_long():
    long_sig = JDKOrderflowStrategy({'min_rr': 1.5}).evaluate('X', _failed_auction('long'))
    short_sig = JDKOrderflowStrategy({'min_rr': 1.5, 'long_only': False}) \
        .evaluate('X', _failed_auction('short'))
    assert short_sig is not None and short_sig.side == 'short'
    assert short_sig.take_profit < short_sig.entry < short_sig.stop_loss
    assert short_sig.entry == pytest.approx(200 - long_sig.entry)
    assert short_sig.stop_loss == pytest.approx(200 - long_sig.stop_loss)
    assert short_sig.rr == pytest.approx(long_sig.rr, rel=0.05)


def test_long_only_blocks_shorts():
    strat = JDKOrderflowStrategy({'min_rr': 1.5, 'long_only': True})
    assert strat.evaluate('X', _failed_auction('short')) is None


def test_cooldown_blocks_repeat_signal():
    strat = JDKOrderflowStrategy({'min_rr': 1.5, 'cooldown_bars': 6})
    df = _failed_auction('long')
    assert strat.evaluate('BTC_EUR', df) is not None
    assert strat.evaluate('BTC_EUR', df) is None


def test_no_signal_in_quiet_range():
    strat = JDKOrderflowStrategy()
    assert strat.evaluate('BTC_EUR', _candles(_balance_range())) is None


def test_analyze_returns_spot_signal():
    strat = JDKOrderflowStrategy({'min_rr': 1.5, 'leverage': 1})
    df = _failed_auction('long')
    sig = strat.analyze('BTC_EUR', df, float(df['close'].iloc[-1]))
    assert sig.signal_type == SignalType.LONG
    assert sig.suggested_leverage == 1
    assert sig.atr_value > 0


def test_breakeven_stop():
    strat = JDKOrderflowStrategy({'breakeven_at_r': 1.0, 'fee_pct': 0.0025})
    # Einstieg 100, Stop 98 → 1R = 2
    assert strat.breakeven_stop('long', 100.0, 98.0, 101.0) is None
    assert strat.breakeven_stop('long', 100.0, 98.0, 102.0) == pytest.approx(100.5)
    assert strat.breakeven_stop('short', 100.0, 102.0, 98.0) == pytest.approx(99.5)
    # Bereits auf Break-Even → nichts mehr ändern
    assert strat.breakeven_stop('long', 100.0, 100.5, 105.0) is None


def _level_test_setup(levels, wick_close=100.4):
    """Range um 100 mit Test von 99 per Docht, Schluss am Hoch; Levels vorgegeben"""
    df = _candles(_balance_range(center=101.0, amp=1.5), spread=0.15)
    df.loc[len(df)] = {
        'timestamp': df['timestamp'].iloc[-1] + pd.Timedelta(hours=1),
        'open': 99.6, 'high': wick_close + 0.02, 'low': 98.95, 'close': wick_close, 'volume': 30.0,
    }
    strat = JDKOrderflowStrategy({'min_rr': 1.5, 'min_tp_pct': 0.005})
    strat._support_levels = lambda df_, profile_: levels
    return strat, df


def test_level_test_needs_confluence():
    levels = [('VAL', 99.0), ('nPOC', 99.1), ('Range-Hoch', 105.0)]
    strat, df = _level_test_setup(levels)
    prof = VolumeProfile(poc=101, vah=102, val=98.5, range_high=105.0, range_low=97.0)
    sig = strat._level_test_long(df, 100.4, 0.5, prof, estimate_delta(df))
    assert sig is not None and sig.setup == 'level_test'
    assert set(sig.levels) == {'VAL', 'nPOC'}
    assert 'Ablehnung' in sig.confirmations
    assert sig.stop_loss < 98.95 and sig.take_profit == pytest.approx(105.0)

    # Nur ein Level in der Zone → kein Trade
    strat, df = _level_test_setup([('VAL', 99.0), ('Range-Hoch', 105.0)])
    prof = VolumeProfile(poc=101, vah=102, val=98.5, range_high=105.0, range_low=97.0)
    assert strat._level_test_long(df, 100.4, 0.5, prof, estimate_delta(df)) is None


def test_level_test_blocked_below_val():
    levels = [('POC', 99.0), ('nPOC', 99.1)]
    strat, df = _level_test_setup(levels)
    prof = VolumeProfile(poc=101, vah=102, val=100.8, range_high=105.0, range_low=97.0)
    assert strat._level_test_long(df, 100.4, 0.5, prof, estimate_delta(df)) is None
