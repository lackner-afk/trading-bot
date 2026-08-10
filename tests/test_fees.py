"""
Tests für die Gebührenannahme.

Bitpanda Fusion hat 7 volumenabhängige Stufen, Level 1 = 0,25 % — und anders
als bei den meisten Börsen kostet Maker dasselbe wie Taker. Dazu ~0,05 %
Spread. Die Config stand auf 0,04 %/0,06 % (Futures-Struktur), also um
Faktor ~4,5 zu niedrig.

Das ist keine Kosmetik: bei 0,6 % Round-Trip steigt die für Break-even nötige
Win-Rate von 37,7 % auf über 50 %, und typische Trailing-Exits liegen unter
den Gebühren — sie wären real Verluste. Eine zu optimistische Gebühr lässt
eine Strategie profitabel aussehen, die Geld verbrennt.
"""

from pathlib import Path

import pytest
import yaml

from core.performance import DEFAULT_ROUND_TRIP_FEE, round_trip_fee_from_config


def settings() -> dict:
    return yaml.safe_load(Path('config/settings.yaml').read_text())


class TestRoundTripAusConfig:
    def test_maker_gleich_taker(self):
        """Fusion-Besonderheit: keine Maker-Vergünstigung."""
        fees = settings()['fees']
        assert fees['crypto_maker'] == fees['crypto_taker']

    def test_level_1_ist_konfiguriert(self):
        fees = settings()['fees']
        assert fees['crypto_taker'] == pytest.approx(0.0025)
        assert fees['spread_estimate'] == pytest.approx(0.0005)

    def test_round_trip_enthaelt_beide_seiten_und_spread(self):
        rt = round_trip_fee_from_config(settings())
        # 2 x (0.25% + 0.05%)
        assert rt == pytest.approx(0.006)

    def test_alte_annahme_war_deutlich_zu_niedrig(self):
        """Dokumentiert das Ausmass: Faktor 5 gegenueber der frueheren 0.0012."""
        rt = round_trip_fee_from_config(settings())
        assert rt / 0.0012 > 4.0

    def test_fallback_ohne_config(self):
        assert round_trip_fee_from_config({}) == pytest.approx(0.006)
        assert round_trip_fee_from_config(None) == pytest.approx(0.006)

    def test_konstante_passt_zum_default(self):
        assert DEFAULT_ROUND_TRIP_FEE == pytest.approx(round_trip_fee_from_config({}))

    def test_guenstigere_stufe_wird_uebernommen(self):
        """Level 7: 0,02 % — wer das Volumen hat, soll es rechnen duerfen."""
        rt = round_trip_fee_from_config({
            'fees': {'crypto_taker': 0.0002, 'spread_estimate': 0.0005}
        })
        assert rt == pytest.approx(0.0014)


class TestOekonomischeKonsequenz:
    """
    Diese Tests rechnen keine Implementierung nach, sondern halten fest,
    was die Gebuehren mit der Strategie machen — damit die Zahlen bei einer
    Parameteraenderung nicht unbemerkt kippen.
    """

    @staticmethod
    def break_even_win_rate(tp_pct: float, sl_pct: float, round_trip: float) -> float:
        netto_gewinn = tp_pct - round_trip
        netto_verlust = sl_pct + round_trip
        if netto_gewinn <= 0:
            return 1.0      # Gewinne decken die Kosten nicht
        return netto_verlust / (netto_gewinn + netto_verlust)

    def test_break_even_steigt_deutlich(self):
        tp, sl = 0.023, 0.012      # typische Werte aus dem SignalAggregator

        alt = self.break_even_win_rate(tp, sl, 0.0012)
        neu = self.break_even_win_rate(tp, sl, round_trip_fee_from_config(settings()))

        assert alt == pytest.approx(0.377, abs=0.01)
        assert neu > 0.50
        assert neu - alt > 0.10

    def test_trailing_arm_muss_ueber_den_gebuehren_liegen(self):
        """
        Der Kern des Problems: wird der Trailing-Stop bei +0,5 % scharf und
        folgt mit ~0,2 % Abstand, liegt der Exit bei ~+0,3 % brutto — unter
        den 0,6 % Round-Trip. Jeder "Gewinntrade" waere real ein Verlust.
        """
        rt = round_trip_fee_from_config(settings())
        cfg = settings()['strategies']['confluence'].get('exit', {})
        arm = cfg.get('trailing_arm_profit_pct', 0.005)

        # Der Stop folgt mit ATR-Abstand (~0,2 % auf 5m bei BTC/ETH/SOL).
        # Entscheidend ist also nicht arm > rt, sondern was nach dem
        # Rueckfall uebrig bleibt.
        typischer_trail_abstand = 0.002
        exit_brutto = arm - typischer_trail_abstand

        assert exit_brutto > rt, (
            f"trailing_arm_profit_pct={arm:.3%} minus Trail-Abstand "
            f"{typischer_trail_abstand:.3%} ergibt einen Exit bei "
            f"{exit_brutto:.3%} brutto — unter dem Round-Trip von {rt:.3%}. "
            f"Jeder 'Gewinntrade' waere real ein Verlust."
        )

    def test_take_profit_traegt_die_kosten(self):
        """Das TP-Ziel muss die Gebuehren deutlich uebersteigen."""
        rt = round_trip_fee_from_config(settings())
        # tp_pct = 0.016 + confidence * 0.012, minimal bei confidence 0
        min_tp = 0.016
        assert min_tp > rt * 2, (
            f"Minimales TP-Ziel {min_tp:.2%} traegt {rt:.2%} Gebuehren nicht"
        )

    def test_kosten_ueber_die_testphase(self):
        """100 Trades a 15 EUR — was die Testphase an Gebuehren kostet."""
        rt = round_trip_fee_from_config(settings())
        kosten = 100 * 15.0 * rt
        kapital = settings()['general']['start_capital']

        assert kosten == pytest.approx(9.0, abs=0.5)
        # 9 % des Kapitals allein an Gebuehren — muss die Strategie erst verdienen
        assert kosten / kapital < 0.15


class TestBacktestNutztDieConfig:
    def test_backtester_default_entspricht_fusion(self):
        from data.backtester import Backtester

        bt = Backtester(initial_capital=100)
        assert bt.taker_fee == pytest.approx(0.003)
        assert bt.maker_fee == pytest.approx(0.003)

    def test_config_schlaegt_default(self):
        from data.backtester import Backtester

        bt = Backtester(initial_capital=100,
                        config={'taker_fee': 0.001, 'maker_fee': 0.001})
        assert bt.taker_fee == pytest.approx(0.001)


class TestGateNutztDieConfig:
    def test_gate_liest_gebuehr_aus_settings(self, tmp_path):
        """
        Das Kriterium "Erwartungswert > 2x Round-Trip-Fee" haengt direkt an
        diesem Wert — mit der alten 0.0012 waere es viel zu leicht erfuellbar.
        """
        from tools.profitability_gate import evaluate_gate

        result = evaluate_gate(str(tmp_path / "leer.db"))
        assert result.passed is False   # keine DB, aber kein Absturz

    def test_erwartungswert_schwelle_steigt(self, tmp_path):
        import sqlite3
        from datetime import datetime, timedelta

        from tools.profitability_gate import evaluate_gate

        db = tmp_path / "t.db"
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT,
            size REAL, entry_price REAL, exit_price REAL, leverage REAL,
            pnl REAL, fees REAL, entry_time TEXT, exit_time TEXT,
            strategy TEXT, market_type TEXT)""")
        conn.execute("CREATE TABLE equity_snapshots (id INTEGER PRIMARY KEY, ts TEXT, equity REAL, balance REAL)")

        # Erwartungswert 0,4 % vom Notional: reicht bei 0,12 % Round-Trip,
        # aber nicht bei 0,6 %
        base = datetime(2026, 1, 1)
        for i in range(10):
            conn.execute(
                "INSERT INTO trades (symbol, side, size, entry_price, exit_price, leverage,"
                " pnl, fees, entry_time, exit_time, strategy, market_type)"
                " VALUES ('BTC_EUR','long',100.0,100.0,100.4,1.0,0.4,0.01,?,?,'confluence','confluence')",
                ((base + timedelta(hours=i)).isoformat(),
                 (base + timedelta(hours=i, minutes=30)).isoformat()),
            )
        conn.commit()
        conn.close()

        locker = evaluate_gate(str(db), round_trip_fee=0.0012)
        streng = evaluate_gate(str(db), round_trip_fee=0.006)

        def kriterium(res, key):
            return next(c for c in res.criteria if c.key == key)

        assert kriterium(locker, "expectancy").passed is True
        assert kriterium(streng, "expectancy").passed is False
