"""Zufalls-Baseline: gemischte Signale behalten Zahl und Länge der Phasen."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('ccxt')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import robustness_daily as rb  # noqa: E402


def test_mischen_erhaelt_phasen():
    rng = np.random.default_rng(0)
    sig = pd.Series([False] * 5 + [True] * 3 + [False] * 10 + [True] * 7 + [False] * 2 + [True] * 1)
    for _ in range(20):
        mixed = rb.shuffled(sig, rng)
        assert len(mixed) == len(sig)
        assert int(mixed.sum()) == int(sig.sum())
        assert sorted(n for hold, n in rb.spells(mixed) if hold) == [1, 3, 7]


def test_simulator_bucht_gebuehren_und_mindestorder():
    """Flacher Kurs, ein Kauf und ein Verkauf: Verlust = zwei Gebühren auf 30 €."""
    dates = pd.date_range('2024-01-01', periods=10)
    flat = pd.DataFrame({c: 100.0 for c in ['open', 'high', 'low', 'close']}, index=dates)
    sym = rb.bt.SYMBOLS[0]
    data = {s: flat.copy() for s in rb.bt.SYMBOLS}
    sig = {id(data[s]): pd.Series(False, index=dates) for s in rb.bt.SYMBOLS}
    sig[id(data[sym])] = pd.Series([False, True, True, True, False, False, False, False, False, False], index=dates)
    r = rb.bt.simulate(data, sig, dates[0], dates[-1])
    order = max(rb.bt.MIN_ORDER_AMOUNT, rb.bt.START_CAPITAL * rb.bt.POSITION_FRACTION)
    assert r['trades'] == 2
    assert r['final'] == pytest.approx(rb.bt.START_CAPITAL - order * rb.bt.TAKER_FEE * (2 - rb.bt.TAKER_FEE), rel=1e-9)
