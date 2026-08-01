"""
Tests für den Confluence-Backtest-Adapter und den Backtester.

Der wichtigste Test hier ist die Look-ahead-Freiheit: sähe die Strategie
im Backtest auch nur eine Kerze aus der Zukunft, wären alle Ergebnisse zu
gut und das Profitabilitäts-Gate würde auf falscher Grundlage öffnen — die
teuerste Fehlerklasse im ganzen Vorhaben.
"""

import pytest

from data.backtester import Backtester
from data.confluence_backtest_adapter import (
    build_strategy_funcs,
    make_confluence_func,
    prepare_candles_for_confluence,
)


class SpyStrategy:
    """Zeichnet auf, welche Daten die Strategie zu sehen bekommt."""

    def __init__(self, signal=None):
        self.calls = []
        self.signal = signal

    def analyze(self, symbol, candles, current_price):
        self.calls.append({
            "symbol": symbol,
            "len": len(candles),
            "last_close": float(candles["close"].iloc[-1]),
            "max_close": float(candles["close"].max()),
            "price": current_price,
        })
        return self.signal


class FakeSignal:
    def __init__(self, direction="long", confidence=0.7, is_exit=False):
        self.direction = direction
        self.confidence = confidence
        self.is_exit_signal = is_exit
        self.stop_loss = 99.0
        self.take_profit = 110.0


class TestKeinLookAhead:
    def test_strategie_sieht_nie_die_zukunft(self, sample_candles):
        """
        Der entscheidende Test. Die Kerzen steigen im Trend, also ist das
        Maximum der Zukunft grösser als das der Vergangenheit. Sähe der
        Adapter zu weit, wäre max_close > dem Wert bei idx.
        """
        spy = SpyStrategy()
        f = make_confluence_func(spy, "BTC_EUR", warmup=80, window=200)

        for idx in range(80, len(sample_candles)):
            f(sample_candles, idx)

        assert spy.calls, "Strategie wurde nie aufgerufen"
        for call, idx in zip(spy.calls, range(80, len(sample_candles))):
            erlaubtes_max = float(sample_candles["close"].iloc[: idx + 1].max())
            assert call["max_close"] <= erlaubtes_max + 1e-9
            assert call["last_close"] == pytest.approx(
                float(sample_candles["close"].iloc[idx])
            )

    def test_preis_entspricht_dem_index(self, sample_candles):
        spy = SpyStrategy()
        f = make_confluence_func(spy, "BTC_EUR", warmup=80)
        f(sample_candles, 120)
        assert spy.calls[0]["price"] == pytest.approx(
            float(sample_candles["close"].iloc[120])
        )

    def test_warmup_wird_eingehalten(self, sample_candles):
        spy = SpyStrategy()
        f = make_confluence_func(spy, "BTC_EUR", warmup=100)
        for idx in range(0, 100):
            assert f(sample_candles, idx) is None
        assert spy.calls == []

    def test_fenster_wird_begrenzt(self, sample_candles):
        """Begrenzt den Aufwand und spiegelt das Live-Verhalten (n=80)."""
        spy = SpyStrategy()
        f = make_confluence_func(spy, "BTC_EUR", warmup=10, window=50)
        f(sample_candles, 150)
        assert spy.calls[0]["len"] == 51   # window + die aktuelle Kerze

    def test_determinismus(self, sample_candles):
        spy_a, spy_b = SpyStrategy(), SpyStrategy()
        fa = make_confluence_func(spy_a, "BTC_EUR", warmup=80)
        fb = make_confluence_func(spy_b, "BTC_EUR", warmup=80)
        for idx in range(80, 150):
            fa(sample_candles, idx)
            fb(sample_candles, idx)
        assert spy_a.calls == spy_b.calls


class TestSignalUebersetzung:
    def test_long_signal(self, sample_candles):
        f = make_confluence_func(SpyStrategy(FakeSignal("long")), "BTC_EUR", warmup=10)
        out = f(sample_candles, 100)
        assert out["action"] == "long"
        assert out["stop_loss"] == 99.0
        assert out["take_profit"] == 110.0

    def test_exit_signal_wird_zu_close(self, sample_candles):
        """Spot-Semantik: SHORT heisst 'verkauf, was du hast'."""
        f = make_confluence_func(
            SpyStrategy(FakeSignal("short", is_exit=True)), "BTC_EUR", warmup=10
        )
        assert f(sample_candles, 100)["action"] == "close"

    def test_kein_signal(self, sample_candles):
        f = make_confluence_func(SpyStrategy(None), "BTC_EUR", warmup=10)
        assert f(sample_candles, 100) is None

    def test_confidence_schwelle(self, sample_candles):
        f = make_confluence_func(
            SpyStrategy(FakeSignal("long", confidence=0.3)), "BTC_EUR",
            warmup=10, min_confidence=0.55
        )
        assert f(sample_candles, 100) is None

    def test_exception_wird_geschluckt(self, sample_candles):
        """Ein Faktor-Fehler darf nicht den ganzen Backtest abbrechen."""
        class Boom:
            def analyze(self, *a, **kw):
                raise ValueError("kaputt")

        f = make_confluence_func(Boom(), "BTC_EUR", warmup=10)
        assert f(sample_candles, 100) is None

    def test_build_strategy_funcs_isoliert_pro_symbol(self):
        created = []

        def factory():
            s = SpyStrategy()
            created.append(s)
            return s

        funcs = build_strategy_funcs(factory, ["BTC_EUR", "ETH_EUR"])
        assert set(funcs) == {"BTC_EUR", "ETH_EUR"}
        # Eigene Instanz je Symbol — sonst vermischt sich Strategie-Zustand
        assert len(created) == 2
        assert created[0] is not created[1]


class TestSignalNormalisierung:
    def test_legacy_strings(self):
        assert Backtester._normalize_signal("long") == {"action": "long"}
        assert Backtester._normalize_signal("close") == {"action": "close"}
        assert Backtester._normalize_signal(None) is None
        assert Backtester._normalize_signal("bloedsinn") is None

    def test_dict_form(self):
        sig = {"action": "long", "stop_loss": 99.0, "take_profit": 110.0}
        assert Backtester._normalize_signal(sig) == sig

    def test_dict_mit_ungueltiger_action(self):
        assert Backtester._normalize_signal({"action": "hodl"}) is None


class TestSynthetikSchutz:
    def test_synthetik_ist_standardmaessig_verboten(self):
        """
        Ein stiller Random-Walk-Fallback könnte einen 'erfolgreichen'
        Backtest vortäuschen, auf dem dann das Gate öffnet.
        """
        bt = Backtester(initial_capital=100)
        assert bt.allow_synthetic is False
        with pytest.raises(RuntimeError, match="Random Walk"):
            bt._fallback_to_synthetic(["BTC/EUR"], 30, "Test")

    def test_synthetik_bewusst_erlaubt(self):
        bt = Backtester(initial_capital=100, config={"allow_synthetic": True})
        data = bt._fallback_to_synthetic(["BTC/USDT"], 5, "Test")
        assert data
        assert bt.data_source == "simulated"

    def test_kapital_defaults_skalieren_mit_startkapital(self):
        """Die alten Defaults (1000/100) ergaben bei 100 EUR null Trades."""
        bt = Backtester(initial_capital=100)
        assert bt.min_capital <= 100
        assert bt.min_margin <= 100

    def test_timeframe_konfigurierbar(self):
        bt = Backtester(initial_capital=100, config={"timeframe": "5m"})
        assert bt.timeframe == "5m"

    def test_run_backtest_braucht_eine_strategie(self):
        bt = Backtester(initial_capital=100)
        with pytest.raises(ValueError):
            bt.run_backtest()


class TestIndikatorVorbereitung:
    def test_ergaenzt_fehlende_spalten(self, sample_candles):
        roh = sample_candles[["timestamp", "open", "high", "low", "close", "volume"]]
        out = prepare_candles_for_confluence(roh)
        for col in ("rsi", "bb_upper", "bb_middle", "bb_lower", "ema_9", "ema_21"):
            assert col in out.columns

    def test_laesst_vorhandene_unveraendert(self, sample_candles):
        out = prepare_candles_for_confluence(sample_candles)
        assert out is sample_candles
