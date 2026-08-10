"""
Tests für den Exit-Pfad.

Der gefährlichste Bug im Repo war, dass _close_position ausschließlich
portfolio.close_position() aufrief und nie eine Order losschickte. Live
hätte der Bot real gekauft, aber Stop-Loss und Take-Profit nur lokal in die
SQLite gebucht — die echte Position wäre unbegrenzt offen geblieben.

Zweiter Punkt: schlägt die Order fehl, darf die Position NICHT lokal
verschwinden. Ein lokal geschlossener, real aber offener Trade ist der
schlechteste denkbare Zustand.
"""

from datetime import datetime, timedelta

import pytest

from core.order_engine import ExecutionResult, Order, OrderType
from strategies.confluence_exit import ConfluenceExitManager


class FakeOrderEngine:
    """Minimale OrderEngine, die Erfolg/Misserfolg steuerbar macht."""

    def __init__(self, succeed=True, execution_price=None, fees=0.12):
        self.succeed = succeed
        self.execution_price = execution_price
        self.fees = fees
        self.calls = []
        self.on_fill = None

    async def execute_market_order(self, symbol, side, size, current_price,
                                   leverage=1.0, strategy="", market_type="crypto"):
        self.calls.append({
            "symbol": symbol, "side": side, "size": size,
            "price": current_price, "leverage": leverage, "strategy": strategy,
        })
        order = Order(id="fake-1", symbol=symbol, side=side,
                      order_type=OrderType.MARKET, size=size)
        if not self.succeed:
            return ExecutionResult(
                order=order,
                success=False,
                message="Insufficient funds",
                execution_price=0.0,
                total_fees=0.0,
                slippage_cost=0.0,
                latency_ms=0,
            )
        return ExecutionResult(
            order=order,
            success=True,
            message="OK",
            execution_price=self.execution_price or current_price,
            total_fees=self.fees,
            slippage_cost=0.0,
            latency_ms=10,
        )

    async def cancel_all_orders(self, symbol=None):
        return 0

    async def close(self):
        return None


@pytest.fixture
def bot(portfolio, monkeypatch):
    """
    TradingBot mit injizierten Komponenten, ohne _init_components().

    Der echte Konstruktor baut Feeds, öffnet trades.db im CWD und kann im
    Live-Zweig RuntimeError werfen — so ist der Bot nicht testbar.
    """
    import logging
    import main as main_module
    from core.market_constraints import MarketConstraints

    b = object.__new__(main_module.TradingBot)
    b.logger = logging.getLogger("test-bot")
    b.config = {}
    b.constraints = MarketConstraints()
    b.portfolio = portfolio
    b.order_engine = FakeOrderEngine()
    b.confluence_exit = ConfluenceExitManager()
    b._exit_failures = {}
    b._last_confluence_breakdowns = {}
    b.running = True
    b.factor_attribution = {}

    class _Reporter:
        telegram = None

        def print_trade_executed(self, trade):
            pass

        async def send_trade_alert(self, trade):
            pass

        async def send_message(self, text):
            pass

    b.reporter = _Reporter()
    return b


def open_long(portfolio, symbol="BTC_EUR", size=20.0, price=100.0):
    return portfolio.open_position(
        symbol=symbol, side="long", size=size, price=price,
        leverage=1.0, strategy="confluence", market_type="confluence",
        stop_loss=95.0, take_profit=110.0,
    )


class TestExitGehtUeberDieEngine:
    async def test_close_position_ruft_die_engine(self, bot):
        open_long(bot.portfolio)
        await bot._close_position("BTC_EUR", 110.0, "Take-Profit")

        assert len(bot.order_engine.calls) == 1
        call = bot.order_engine.calls[0]
        assert call["symbol"] == "BTC_EUR"
        assert call["side"] == "sell"          # Long wird verkauft
        assert call["size"] == 20.0

    async def test_short_wird_zurueckgekauft(self, margin_portfolio, bot):
        bot.portfolio = margin_portfolio
        margin_portfolio.open_position(
            symbol="BTC_EUR", side="short", size=20.0, price=100.0,
            leverage=1.0, strategy="confluence", market_type="confluence",
        )
        await bot._close_position("BTC_EUR", 90.0, "Take-Profit")
        assert bot.order_engine.calls[0]["side"] == "buy"

    async def test_exit_preis_kommt_aus_der_ausfuehrung(self, bot):
        """Nicht der Signalpreis zählt, sondern der echte Fill."""
        open_long(bot.portfolio, price=100.0)
        bot.order_engine.execution_price = 109.5   # Slippage gegenüber 110

        await bot._close_position("BTC_EUR", 110.0, "Take-Profit")

        trade = bot.portfolio.trades[-1]
        assert trade.exit_price == pytest.approx(109.5)

    async def test_gebuehren_kommen_aus_der_ausfuehrung(self, bot):
        """Vorher stand hier size * 0.0006 hartcodiert."""
        open_long(bot.portfolio)
        bot.order_engine.fees = 0.42

        await bot._close_position("BTC_EUR", 110.0, "Take-Profit")

        assert bot.portfolio.trades[-1].fees == pytest.approx(0.42)

    async def test_position_ist_danach_weg(self, bot):
        open_long(bot.portfolio)
        await bot._close_position("BTC_EUR", 110.0, "Take-Profit")
        assert "BTC_EUR" not in bot.portfolio.positions

    async def test_unbekanntes_symbol_macht_nichts(self, bot):
        await bot._close_position("DOGE_EUR", 1.0, "Test")
        assert bot.order_engine.calls == []


class TestFehlgeschlagenerExit:
    async def test_position_bleibt_bei_fehler_offen(self, bot):
        """Der wichtigste Test: kein lokaler Close ohne echten Close."""
        open_long(bot.portfolio)
        bot.order_engine.succeed = False

        await bot._close_position("BTC_EUR", 110.0, "Take-Profit")

        assert "BTC_EUR" in bot.portfolio.positions
        assert bot.portfolio.trades == []

    async def test_exception_laesst_position_offen(self, bot):
        open_long(bot.portfolio)

        async def boom(*args, **kwargs):
            raise ConnectionError("API weg")

        bot.order_engine.execute_market_order = boom
        await bot._close_position("BTC_EUR", 110.0, "Take-Profit")

        assert "BTC_EUR" in bot.portfolio.positions

    async def test_fehlversuche_werden_gezaehlt(self, bot):
        open_long(bot.portfolio)
        bot.order_engine.succeed = False

        await bot._close_position("BTC_EUR", 110.0, "SL")
        await bot._close_position("BTC_EUR", 110.0, "SL")

        assert bot._exit_failures["BTC_EUR"] == 2
        assert bot.running is True

    async def test_kill_switch_nach_zu_vielen_fehlversuchen(self, bot):
        open_long(bot.portfolio)
        bot.order_engine.succeed = False

        for _ in range(bot.MAX_EXIT_FAILURES):
            await bot._close_position("BTC_EUR", 110.0, "SL")

        assert bot.running is False
        assert "BTC_EUR" in bot.portfolio.positions

    async def test_erfolg_setzt_zaehler_zurueck(self, bot):
        open_long(bot.portfolio)
        bot.order_engine.succeed = False
        await bot._close_position("BTC_EUR", 110.0, "SL")
        assert bot._exit_failures["BTC_EUR"] == 1

        bot.order_engine.succeed = True
        await bot._close_position("BTC_EUR", 110.0, "SL")
        assert "BTC_EUR" not in bot._exit_failures


class TestExitSignalRouting:
    async def test_short_signal_schliesst_offenen_long(self, bot):
        open_long(bot.portfolio)

        class Sig:
            symbol = "BTC_EUR"
            confidence = 0.7

        await bot._handle_exit_signal(Sig(), 105.0)

        assert bot.order_engine.calls[0]["side"] == "sell"
        assert "BTC_EUR" not in bot.portfolio.positions

    async def test_short_signal_ohne_position_macht_nichts(self, bot):
        class Sig:
            symbol = "BTC_EUR"
            confidence = 0.7

        await bot._handle_exit_signal(Sig(), 105.0)
        assert bot.order_engine.calls == []


class TestConfluenceExitManager:
    def test_stop_loss_long(self):
        m = ConfluenceExitManager()
        m.register("BTC_EUR", 100.0)
        should, reason = m.check_exit("BTC_EUR", 100.0, 94.0, "long", stop_loss=95.0)
        assert should and "Stop-Loss" in reason

    def test_take_profit_long(self):
        m = ConfluenceExitManager()
        m.register("BTC_EUR", 100.0)
        should, reason = m.check_exit("BTC_EUR", 100.0, 111.0, "long", take_profit=110.0)
        assert should and "Take-Profit" in reason

    def test_kein_exit_im_korridor(self):
        m = ConfluenceExitManager()
        m.register("BTC_EUR", 100.0)
        should, _ = m.check_exit("BTC_EUR", 100.0, 101.0, "long",
                                 stop_loss=95.0, take_profit=110.0)
        assert should is False

    def test_trailing_stop_funktioniert(self):
        """
        Der Kern der Sache: vorher war highest_prices nie befüllt und der
        Trailing-Zweig damit toter Code.
        """
        m = ConfluenceExitManager(trailing_pct_fallback=0.02,
                                  trailing_arm_profit_pct=0.005)
        m.register("BTC_EUR", 100.0)

        # Auf 108 hoch — Trailing wird scharf
        should, _ = m.check_exit("BTC_EUR", 100.0, 108.0, "long", take_profit=200.0)
        assert should is False
        assert m.states["BTC_EUR"].highest == 108.0
        assert m.states["BTC_EUR"].trailing_armed is True

        # Rückfall auf 105: 108 - 2% von 105 (2.1) = 105.9 -> Exit
        should, reason = m.check_exit("BTC_EUR", 100.0, 105.0, "long", take_profit=200.0)
        assert should and "Trailing-Stop" in reason

    def test_trailing_erst_ab_gewinnschwelle(self):
        m = ConfluenceExitManager(trailing_pct_fallback=0.02,
                                  trailing_arm_profit_pct=0.05)
        m.register("BTC_EUR", 100.0)
        # Nur +1% -> Trailing noch nicht scharf
        m.check_exit("BTC_EUR", 100.0, 101.0, "long", take_profit=200.0)
        should, _ = m.check_exit("BTC_EUR", 100.0, 98.0, "long", take_profit=200.0)
        assert should is False

    def test_atr_hat_vorrang_vor_prozent(self):
        m = ConfluenceExitManager(trailing_atr_multiplier=1.0,
                                  trailing_pct_fallback=0.5,
                                  trailing_arm_profit_pct=0.0)
        m.register("BTC_EUR", 100.0)
        m.check_exit("BTC_EUR", 100.0, 110.0, "long", atr=2.0, take_profit=200.0)
        # Trigger = 110 - 2.0 = 108
        should, _ = m.check_exit("BTC_EUR", 100.0, 108.5, "long", atr=2.0, take_profit=200.0)
        assert should is False
        should, _ = m.check_exit("BTC_EUR", 100.0, 107.5, "long", atr=2.0, take_profit=200.0)
        assert should is True

    def test_zeit_stop(self):
        m = ConfluenceExitManager(max_hold_hours=1.0)
        m.register("BTC_EUR", 100.0, datetime.now() - timedelta(hours=2))
        should, reason = m.check_exit("BTC_EUR", 100.0, 100.5, "long",
                                      entry_time=datetime.now() - timedelta(hours=2))
        assert should and "Zeit-Stop" in reason

    def test_forget_entfernt_tracking(self):
        m = ConfluenceExitManager()
        m.register("BTC_EUR", 100.0)
        m.forget("BTC_EUR")
        assert "BTC_EUR" not in m.states

    def test_tracking_wird_nachgezogen(self):
        """Positionen aus der DB nach Neustart haben noch kein Tracking."""
        m = ConfluenceExitManager()
        should, _ = m.check_exit("BTC_EUR", 100.0, 94.0, "long", stop_loss=95.0)
        assert should is True
        assert "BTC_EUR" in m.states
