"""
Tests für core/portfolio.py — PNL-Rechnung, SQLite-Persistenz und Metriken.

Deckt gezielt die Defekte ab, die vor P0 bestanden:
Trades wurden beim Start nie geladen, Trade-IDs kollidierten nach Neustart,
und der Profit Factor war in Wahrheit die Payoff-Ratio.
"""

from datetime import datetime, timedelta

import pytest

from core.portfolio import Portfolio


def _roundtrip(portfolio, symbol, entry, exit_price, size=20.0, leverage=1.0,
               side="long", fees=0.0, strategy="test"):
    """Öffnet und schließt eine Position, gibt den Trade zurück."""
    portfolio.open_position(
        symbol=symbol, side=side, size=size, price=entry,
        leverage=leverage, strategy=strategy,
    )
    return portfolio.close_position(
        symbol=symbol, exit_price=exit_price, fees=fees, strategy=strategy,
    )


class TestPnL:
    def test_long_gewinn_bei_leverage_1(self, portfolio):
        # +10% Preisbewegung auf 20 EUR Notional = +2 EUR
        trade = _roundtrip(portfolio, "BTC_EUR", 100.0, 110.0, size=20.0)
        assert trade.pnl == pytest.approx(2.0)

    def test_long_verlust(self, portfolio):
        trade = _roundtrip(portfolio, "BTC_EUR", 100.0, 95.0, size=20.0)
        assert trade.pnl == pytest.approx(-1.0)

    def test_leverage_skaliert_pnl(self, portfolio):
        # Balance reicht: margin = size/leverage = 20/10 = 2 EUR
        trade = _roundtrip(portfolio, "BTC_EUR", 100.0, 110.0, size=20.0, leverage=10.0)
        assert trade.pnl == pytest.approx(20.0)

    def test_gebuehren_werden_abgezogen(self, portfolio):
        trade = _roundtrip(portfolio, "BTC_EUR", 100.0, 110.0, size=20.0, fees=0.5)
        assert trade.pnl == pytest.approx(1.5)

    def test_balance_nach_gewinn(self, portfolio):
        _roundtrip(portfolio, "BTC_EUR", 100.0, 110.0, size=20.0)
        # Margin (20) zurück + 2 Gewinn
        assert portfolio.balance == pytest.approx(102.0)


class TestPersistenz:
    def test_trades_ueberleben_neustart(self, tmp_path):
        db = str(tmp_path / "p.db")
        p1 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        _roundtrip(p1, "BTC_EUR", 100.0, 110.0)
        _roundtrip(p1, "ETH_EUR", 100.0, 90.0)
        assert len(p1.trades) == 2

        # Vor P0 war self.trades nach dem Neustart leer, obwohl die Zeilen
        # in der DB lagen — get_avg_win_loss() lieferte danach 0.
        p2 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        assert len(p2.trades) == 2
        assert {t.symbol for t in p2.trades} == {"BTC_EUR", "ETH_EUR"}
        assert p2.get_avg_win_loss() == p1.get_avg_win_loss()

    def test_trade_ids_kollidieren_nicht_nach_neustart(self, tmp_path):
        db = str(tmp_path / "p.db")
        p1 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        first = _roundtrip(p1, "BTC_EUR", 100.0, 110.0)

        p2 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        second = _roundtrip(p2, "ETH_EUR", 100.0, 110.0)

        assert first.id != second.id
        assert len({t.id for t in p2.trades}) == len(p2.trades)

    def test_balance_und_counts_ueberleben_neustart(self, tmp_path):
        db = str(tmp_path / "p.db")
        p1 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        _roundtrip(p1, "BTC_EUR", 100.0, 110.0)
        balance, wins = p1.balance, p1.win_count

        p2 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        assert p2.balance == pytest.approx(balance)
        assert p2.win_count == wins

    def test_offene_position_ueberlebt_neustart(self, tmp_path):
        db = str(tmp_path / "p.db")
        p1 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        p1.open_position("BTC_EUR", "long", 20.0, 100.0, 1.0, "test")

        p2 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        assert "BTC_EUR" in p2.positions
        assert p2.positions["BTC_EUR"].entry_price == pytest.approx(100.0)

    def test_equity_snapshots_werden_persistiert(self, tmp_path):
        db = str(tmp_path / "p.db")
        p1 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        p1.update_position_prices({})
        p1.update_position_prices({})

        p2 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        assert len(p2.get_equity_snapshots()) >= 2

    def test_snapshot_drosselung(self, tmp_path):
        p = Portfolio(start_capital=100.0, db_path=str(tmp_path / "p.db"),
                      snapshot_interval_seconds=3600)
        for _ in range(10):
            p.update_position_prices({})
        # Nur der erste Aufruf schreibt, der Rest liegt innerhalb des Intervalls
        assert len(p.get_equity_snapshots()) == 1

    def test_reset_leert_snapshots(self, portfolio):
        portfolio.update_position_prices({})
        portfolio.reset()
        assert portfolio.get_equity_snapshots() == []
        assert portfolio.trades == []


class TestMetriken:
    def test_profit_factor_ist_nicht_payoff_ratio(self, portfolio):
        """
        Der entscheidende Test: 1 Gewinn von 3, Payoff 1.5.
        Payoff-Ratio = 1.5 (sieht profitabel aus), echter PF = 0.75 (Verlust).
        """
        _roundtrip(portfolio, "A_EUR", 100.0, 115.0, size=20.0)   # +3.0
        _roundtrip(portfolio, "B_EUR", 100.0, 90.0, size=20.0)    # -2.0
        _roundtrip(portfolio, "C_EUR", 100.0, 90.0, size=20.0)    # -2.0

        avg_win, avg_loss = portfolio.get_avg_win_loss()
        payoff = abs(avg_win / avg_loss)

        assert payoff == pytest.approx(1.5)
        assert portfolio.get_profit_factor() == pytest.approx(0.75)
        assert portfolio.get_profit_factor() < 1.0

    def test_profit_factor_ohne_verluste(self, portfolio):
        _roundtrip(portfolio, "A_EUR", 100.0, 110.0)
        assert portfolio.get_profit_factor() == float("inf")

    def test_profit_factor_ohne_trades(self, portfolio):
        assert portfolio.get_profit_factor() == 0.0

    def test_expectancy_und_net_pnl(self, portfolio):
        _roundtrip(portfolio, "A_EUR", 100.0, 110.0, size=20.0)   # +2.0
        _roundtrip(portfolio, "B_EUR", 100.0, 95.0, size=20.0)    # -1.0

        assert portfolio.get_net_pnl() == pytest.approx(1.0)
        assert portfolio.get_expectancy() == pytest.approx(0.5)

    def test_total_fees(self, portfolio):
        _roundtrip(portfolio, "A_EUR", 100.0, 110.0, fees=0.3)
        _roundtrip(portfolio, "B_EUR", 100.0, 110.0, fees=0.2)
        assert portfolio.get_total_fees() == pytest.approx(0.5)

    def test_win_rate(self, portfolio):
        _roundtrip(portfolio, "A_EUR", 100.0, 110.0)
        _roundtrip(portfolio, "B_EUR", 100.0, 90.0)
        assert portfolio.get_win_rate() == pytest.approx(0.5)

    def test_sharpe_braucht_mehrere_tage(self, portfolio):
        """Mit weniger als drei Tagespunkten gibt es keinen sinnvollen Sharpe."""
        portfolio.update_position_prices({})
        assert portfolio.get_sharpe_ratio() == 0.0

    def test_sharpe_auf_tagesbasis(self, tmp_path):
        """
        Sharpe muss aus Tages-Returns kommen und mit sqrt(365) annualisiert
        werden. Vorher wurden Sekunden-Snapshots mit sqrt(24*365) hochskaliert.
        """
        import sqlite3
        db = str(tmp_path / "p.db")
        p = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)

        # Fünf Tage mit konstantem täglichem Zuwachs
        conn = sqlite3.connect(db)
        base = datetime(2026, 1, 1, 12, 0, 0)
        for i, eq in enumerate([100.0, 101.0, 102.0, 103.5, 104.0]):
            conn.execute(
                "INSERT INTO equity_snapshots (ts, equity, balance) VALUES (?, ?, ?)",
                ((base + timedelta(days=i)).isoformat(), eq, eq),
            )
        conn.commit()
        conn.close()

        curve = p.get_daily_equity_curve()
        assert len(curve) == 5

        sharpe = p.get_sharpe_ratio()
        # Positiver Trend -> positiver Sharpe, aber in plausibler Größenordnung
        assert sharpe > 0
        assert sharpe < 200

    def test_max_drawdown_aus_snapshots(self, tmp_path):
        import sqlite3
        db = str(tmp_path / "p.db")
        p = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)

        conn = sqlite3.connect(db)
        base = datetime(2026, 1, 1)
        for i, eq in enumerate([100.0, 120.0, 90.0, 110.0]):
            conn.execute(
                "INSERT INTO equity_snapshots (ts, equity, balance) VALUES (?, ?, ?)",
                ((base + timedelta(hours=i)).isoformat(), eq, eq),
            )
        conn.commit()
        conn.close()

        # Peak 120 -> Tief 90 = 25%
        assert p.get_max_drawdown() == pytest.approx(0.25)

    def test_max_drawdown_ueberlebt_neustart(self, tmp_path):
        """Vor P0 lag die Equity-Kurve nur im RAM und war nach Neustart weg."""
        db = str(tmp_path / "p.db")
        p1 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        p1.update_position_prices({})
        p1.balance = 80.0
        p1.update_position_prices({})

        p2 = Portfolio(start_capital=100.0, db_path=db, snapshot_interval_seconds=0)
        assert p2.get_max_drawdown() > 0
