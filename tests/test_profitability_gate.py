"""
Tests für core/performance.py und tools/profitability_gate.py.

Kernanspruch: das Gate darf einen verlierenden oder zufällig gewinnenden
Bot nicht durchwinken. Die bisherige Go-Live-Checkliste prüfte nur
Config-Dateien per Substring und hätte genau das getan.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

from core.performance import compute_performance, load_daily_equity, load_trades
from tools.profitability_gate import DEFAULT_CRITERIA, evaluate_gate, format_report


def make_db(tmp_path, trades, equity=None, name="trades.db"):
    """Baut eine trades.db mit den gegebenen Trades und einer Equity-Kurve."""
    db = tmp_path / name
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT,
            size REAL, entry_price REAL, exit_price REAL, leverage REAL,
            pnl REAL, fees REAL, entry_time TEXT, exit_time TEXT,
            strategy TEXT, market_type TEXT)
    """)
    conn.execute("""
        CREATE TABLE equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, equity REAL, balance REAL)
    """)

    base = datetime(2026, 1, 1)
    for i, t in enumerate(trades):
        exit_time = t.get("exit_time", base + timedelta(hours=i * 8))
        conn.execute(
            "INSERT INTO trades (symbol, side, size, entry_price, exit_price, "
            "leverage, pnl, fees, entry_time, exit_time, strategy, market_type) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.get("symbol", "BTC_EUR"), t.get("side", "long"), t.get("size", 20.0),
             100.0, 100.0, t.get("leverage", 1.0), t["pnl"], t.get("fees", 0.02),
             (exit_time - timedelta(hours=1)).isoformat(), exit_time.isoformat(),
             t.get("strategy", "confluence"), "confluence"),
        )

    if equity:
        for i, eq in enumerate(equity):
            conn.execute(
                "INSERT INTO equity_snapshots (ts, equity, balance) VALUES (?,?,?)",
                ((base + timedelta(days=i)).isoformat(), eq, eq),
            )

    conn.commit()
    conn.close()
    return str(db)


def gewinner_serie(n=120, tage=40):
    """Solide Serie: 60% Trefferquote, Payoff ~1.7, gleichmässig verteilt."""
    base = datetime(2026, 1, 1)
    trades = []
    for i in range(n):
        gewinn = (i % 5) < 3          # 60% Gewinne
        trades.append({
            "pnl": 1.7 if gewinn else -1.0,
            "exit_time": base + timedelta(hours=i * (tage * 24 / n)),
        })
    return trades


def steigende_equity(tage=40, start=100.0, pro_tag=0.4):
    """Gleichmaessig steigende Kurve mit etwas Rauschen."""
    return [start + i * pro_tag + (0.25 if i % 3 == 0 else -0.15) for i in range(tage)]


class TestPerformanceBerechnung:
    def test_leere_db(self, tmp_path):
        db = make_db(tmp_path, [])
        r = compute_performance(db)
        assert r.total_trades == 0
        assert r.profit_factor == 0.0

    def test_fehlende_db(self, tmp_path):
        r = compute_performance(str(tmp_path / "gibtsnicht.db"))
        assert r.total_trades == 0
        assert load_trades(str(tmp_path / "gibtsnicht.db")) == []

    def test_profit_factor_gegen_payoff(self, tmp_path):
        """Payoff 1.5 sieht gut aus, echter PF 0.75 ist ein Verlustsystem."""
        db = make_db(tmp_path, [{"pnl": 3.0}, {"pnl": -2.0}, {"pnl": -2.0}])
        r = compute_performance(db)
        assert r.payoff_ratio == pytest.approx(1.5)
        assert r.profit_factor == pytest.approx(0.75)

    def test_kennzahlen(self, tmp_path):
        db = make_db(tmp_path, [{"pnl": 2.0, "fees": 0.1}, {"pnl": -1.0, "fees": 0.1}])
        r = compute_performance(db)
        assert r.total_trades == 2
        assert r.win_rate == pytest.approx(0.5)
        assert r.net_pnl == pytest.approx(1.0)
        assert r.total_fees == pytest.approx(0.2)
        assert r.expectancy == pytest.approx(0.5)

    def test_konzentration_wird_erkannt(self, tmp_path):
        """Ein Glückstrade trägt fast den ganzen Gewinn."""
        db = make_db(tmp_path, [{"pnl": 90.0}] + [{"pnl": 1.0} for _ in range(10)])
        r = compute_performance(db)
        assert r.largest_win_share == pytest.approx(0.9)

    def test_spot_gegenprobe(self, tmp_path):
        db = make_db(tmp_path, [
            {"pnl": 1.0},
            {"pnl": 1.0, "side": "short"},
            {"pnl": 1.0, "leverage": 8.0},
        ])
        r = compute_performance(db)
        assert r.short_trades == 1
        assert r.leveraged_trades == 1

    def test_wochen_konsistenz(self, tmp_path):
        base = datetime(2026, 1, 5)   # Montag
        db = make_db(tmp_path, [
            {"pnl": 5.0, "exit_time": base},
            {"pnl": -5.0, "exit_time": base + timedelta(days=7)},
            {"pnl": 5.0, "exit_time": base + timedelta(days=14)},
        ])
        r = compute_performance(db)
        assert r.positive_week_share == pytest.approx(2 / 3)

    def test_sharpe_und_drawdown_aus_kurve(self, tmp_path):
        db = make_db(tmp_path, [{"pnl": 1.0}],
                     equity=[100.0, 120.0, 90.0, 110.0])
        r = compute_performance(db)
        assert r.max_drawdown == pytest.approx(0.25)
        assert len(load_daily_equity(db)) == 4

    def test_strategie_aufschluesselung(self, tmp_path):
        db = make_db(tmp_path, [
            {"pnl": 2.0, "strategy": "confluence"},
            {"pnl": -1.0, "strategy": "confluence"},
            {"pnl": 3.0, "strategy": "momentum"},
        ])
        r = compute_performance(db)
        assert r.per_strategy["confluence"]["trades"] == 2
        assert r.per_strategy["momentum"]["pnl"] == pytest.approx(3.0)


class TestGateLehntAb:
    def test_ohne_datenbank(self, tmp_path):
        res = evaluate_gate(str(tmp_path / "nix.db"))
        assert res.passed is False
        assert res.blockers

    def test_ohne_trades(self, tmp_path):
        res = evaluate_gate(make_db(tmp_path, []))
        assert res.passed is False
        assert "Testphase" in res.blockers[0]

    def test_zu_wenige_trades(self, tmp_path):
        db = make_db(tmp_path, gewinner_serie(n=20), equity=steigende_equity())
        res = evaluate_gate(db)
        assert res.passed is False
        assert any(c.key == "trades" and not c.passed for c in res.criteria)

    def test_verlustsystem_wird_abgelehnt(self, tmp_path):
        """Der Kernfall: der Bot verbrennt Geld."""
        base = datetime(2026, 1, 1)
        trades = [{"pnl": 1.0 if i % 5 < 3 else -3.0,
                   "exit_time": base + timedelta(hours=i * 8)} for i in range(120)]
        db = make_db(tmp_path, trades, equity=steigende_equity())
        res = evaluate_gate(db)
        assert res.passed is False
        assert any(c.key == "net_pnl" and not c.passed for c in res.criteria)

    def test_glueckstrade_wird_erkannt(self, tmp_path):
        """Ein einziger Riesengewinn trägt alles — darf nicht durchgehen."""
        base = datetime(2026, 1, 1)
        trades = [{"pnl": 500.0, "exit_time": base}]
        trades += [{"pnl": -1.0, "exit_time": base + timedelta(hours=i * 8)}
                   for i in range(1, 120)]
        db = make_db(tmp_path, trades, equity=steigende_equity())
        res = evaluate_gate(db)
        assert res.passed is False
        assert any(c.key == "concentration" and not c.passed for c in res.criteria)

    def test_shorts_in_der_db_blocken(self, tmp_path):
        """Testdaten mit Shorts beschreiben eine live nicht ausführbare Strategie."""
        trades = gewinner_serie()
        trades[0]["side"] = "short"
        db = make_db(tmp_path, trades, equity=steigende_equity())
        res = evaluate_gate(db)
        assert res.passed is False
        assert any(c.key == "spot_konform" and not c.passed for c in res.criteria)

    def test_hebel_in_der_db_blockt(self, tmp_path):
        trades = gewinner_serie()
        trades[0]["leverage"] = 8.0
        db = make_db(tmp_path, trades, equity=steigende_equity())
        res = evaluate_gate(db)
        assert any(c.key == "spot_konform" and not c.passed for c in res.criteria)

    def test_zu_kurze_laufzeit(self, tmp_path):
        db = make_db(tmp_path, gewinner_serie(n=120, tage=3), equity=steigende_equity(3))
        res = evaluate_gate(db)
        assert any(c.key == "days" and not c.passed for c in res.criteria)


class TestGateLaesstDurch:
    def test_solide_serie_besteht(self, tmp_path):
        db = make_db(tmp_path, gewinner_serie(n=120, tage=40),
                     equity=steigende_equity(40))
        res = evaluate_gate(db)
        nicht_bestanden = [c.key for c in res.criteria if not c.passed]
        assert res.passed, f"Nicht bestanden: {nicht_bestanden}"
        assert res.passed_count == res.total_count

    def test_kriterien_ueberschreibbar(self, tmp_path):
        db = make_db(tmp_path, gewinner_serie(n=20, tage=40),
                     equity=steigende_equity(40))
        assert evaluate_gate(db).passed is False
        assert evaluate_gate(db, criteria={"min_trades": 10}).passed is True


class TestAusgabe:
    def test_report_ist_lesbar(self, tmp_path):
        db = make_db(tmp_path, gewinner_serie(), equity=steigende_equity())
        text = format_report(evaluate_gate(db))
        assert "PROFITABILITAETS-GATE" in text
        assert "Profit Factor" in text

    def test_report_ohne_daten(self, tmp_path):
        text = format_report(evaluate_gate(str(tmp_path / "nix.db")))
        assert "nicht moeglich" in text

    def test_json_export(self, tmp_path):
        db = make_db(tmp_path, gewinner_serie(), equity=steigende_equity())
        d = evaluate_gate(db).to_dict()
        assert "passed" in d and "criteria" in d
        assert len(d["criteria"]) == len(DEFAULT_CRITERIA) + 1   # + Spot-Gegenprobe
