"""
Tests für core/spot_reconciliation.py und den Fusion-Feed.

Kern der Spot-Reconciliation: auf einem Spot-Venue IST der Base-Asset-Bestand
die Position. Damit lässt sich der Abgleich wirklich durchführen — die alte
Reconciliation konnte nur warnen.

Der wichtigste Fall ist die verwaiste lokale Position: der Bot glaubt BTC zu
halten, das Konto sagt nein. Ein Exit würde garantiert fehlschlagen, deshalb
muss der Start blockieren.
"""

import pytest

from core.spot_reconciliation import SpotReconciler, run_spot_reconciliation
from data.bitpanda_fusion_feed import BitpandaFusionFeed
from data.indicators import atr, calculate_indicators, ohlcv_to_df


class FakeEngine:
    """Engine-Ersatz, der nur fetch_balance beantwortet."""

    def __init__(self, balances=None, raise_error=None):
        self.balances = balances if balances is not None else {"EUR": 100.0}
        self.raise_error = raise_error

    async def fetch_balance(self):
        if self.raise_error:
            raise self.raise_error
        return self.balances


def open_long(portfolio, symbol="BTC_EUR", size=20.0, price=50000.0):
    return portfolio.open_position(
        symbol=symbol, side="long", size=size, price=price,
        leverage=1.0, strategy="confluence", market_type="confluence",
    )


class TestQuoteBalance:
    async def test_kontostand_wird_uebernommen(self, portfolio):
        """Das Konto ist die Wahrheit — sonst rechnet der Bot mit Geld, das er nicht hat."""
        engine = FakeEngine({"EUR": 87.5})
        report = await run_spot_reconciliation(portfolio, engine)

        assert report.balance_synced
        assert portfolio.balance == pytest.approx(87.5)
        assert report.quote_difference == pytest.approx(-12.5)

    async def test_kleine_differenz_wird_ignoriert(self, portfolio):
        engine = FakeEngine({"EUR": 100.5})
        await run_spot_reconciliation(portfolio, engine)
        assert portfolio.balance == pytest.approx(100.0)

    async def test_fehler_beim_abruf(self, portfolio):
        engine = FakeEngine(raise_error=ConnectionError("API weg"))
        report = await run_spot_reconciliation(portfolio, engine)
        assert report.success is False
        assert report.errors

    async def test_unerwartetes_format(self, portfolio):
        class Broken:
            async def fetch_balance(self):
                return "kaputt"

        report = await run_spot_reconciliation(portfolio, Broken())
        assert report.success is False


class TestKapitaltopf:
    """
    Der Bot laeuft auf einem Konto, auf dem auch privates Geld liegt.
    Mit quote_currency=EURCV darf ausschliesslich der EURCV-Bestand sein
    Kapital sein — das EUR-Guthaben muss unsichtbar bleiben.
    """

    async def test_nimmt_nur_das_topf_asset(self, portfolio):
        engine = FakeEngine({"EUR": 5000.0, "EURCV": 100.0})
        report = await run_spot_reconciliation(
            portfolio, engine, quote_currency="EURCV"
        )

        # Entscheidend: NICHT die 5000 EUR
        assert portfolio.balance == pytest.approx(100.0)
        assert report.exchange_quote_balance == pytest.approx(100.0)

    async def test_eur_taucht_als_fremdbestand_auf(self, portfolio):
        """Unsichtbar als Kapital, aber sichtbar im Report — nicht stillschweigend."""
        engine = FakeEngine({"EUR": 5000.0, "EURCV": 100.0})
        report = await run_spot_reconciliation(
            portfolio, engine, quote_currency="EURCV"
        )

        fremd = [d for d in report.drifts if d.kind == "unknown_holding"]
        assert any(d.symbol.startswith("EUR_") for d in fremd)
        assert report.success is True     # blockiert den Start nicht

    async def test_gewinne_wachsen_den_topf(self, portfolio):
        """
        Aus 100 werden 400: der Bot handelt danach mit 400. Es gibt keine
        Obergrenze — die Trennung ist der Schutz, nicht ein Deckel.
        """
        engine = FakeEngine({"EUR": 5000.0, "EURCV": 400.0})
        await run_spot_reconciliation(portfolio, engine, quote_currency="EURCV")
        assert portfolio.balance == pytest.approx(400.0)

    async def test_leerer_topf_gibt_null(self, portfolio):
        """Wer kein EURCV eingezahlt hat, handelt mit 0 — nicht mit dem EUR-Bestand."""
        engine = FakeEngine({"EUR": 5000.0})
        await run_spot_reconciliation(portfolio, engine, quote_currency="EURCV")
        assert portfolio.balance == pytest.approx(0.0)


class TestPositionsabgleich:
    async def test_saubere_uebereinstimmung(self, portfolio):
        open_long(portfolio, size=20.0, price=50000.0)     # 0.0004 BTC
        engine = FakeEngine({"EUR": 80.0, "BTC": 0.0004})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"BTC_EUR": 50000.0}
        )
        assert report.success is True
        assert report.drifts == []

    async def test_verwaiste_position_blockiert(self, portfolio):
        """Der wichtigste Fall: lokale Position ohne Deckung auf dem Konto."""
        open_long(portfolio, size=20.0, price=50000.0)
        engine = FakeEngine({"EUR": 80.0})     # kein BTC

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"BTC_EUR": 50000.0}
        )
        assert report.success is False
        assert report.has_blocking_drift
        assert any(d.kind == "orphaned_local" for d in report.drifts)

    async def test_zu_wenig_bestand_blockiert(self, portfolio):
        open_long(portfolio, size=20.0, price=50000.0)     # erwartet 0.0004
        engine = FakeEngine({"EUR": 80.0, "BTC": 0.0001})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"BTC_EUR": 50000.0}
        )
        assert report.success is False

    async def test_toleranz_bei_kleiner_abweichung(self, portfolio):
        """Rundung, Teilausführungen und Gebühren in Base dürfen nicht blockieren."""
        open_long(portfolio, size=20.0, price=50000.0)
        engine = FakeEngine({"EUR": 80.0, "BTC": 0.000399})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"BTC_EUR": 50000.0}
        )
        assert report.success is True

    async def test_mehr_bestand_warnt_nur(self, portfolio):
        """Manuell zugekauft — unschoen, aber nicht gefaehrlich."""
        open_long(portfolio, size=20.0, price=50000.0)
        engine = FakeEngine({"EUR": 80.0, "BTC": 0.01})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"BTC_EUR": 50000.0}
        )
        assert report.success is True
        assert any(d.kind == "size_mismatch" for d in report.drifts)
        assert report.warnings

    async def test_unbekannter_bestand_warnt_nur(self, portfolio):
        """Der Bot fasst ihn nicht an, muss ihn aber melden."""
        engine = FakeEngine({"EUR": 100.0, "ETH": 0.5})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"ETH_EUR": 3000.0}
        )
        assert report.success is True
        assert any(d.kind == "unknown_holding" for d in report.drifts)

    async def test_staub_wird_ignoriert(self, portfolio):
        """Restbetraege unter 1 EUR Gegenwert sind kein Bestand."""
        engine = FakeEngine({"EUR": 100.0, "ETH": 0.0001})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"ETH_EUR": 3000.0}
        )
        assert report.drifts == []

    async def test_ohne_positionen_und_bestaende(self, portfolio):
        report = await run_spot_reconciliation(portfolio, FakeEngine({"EUR": 100.0}))
        assert report.success is True
        assert report.actions_taken

    async def test_mehrere_positionen(self, portfolio):
        open_long(portfolio, "BTC_EUR", size=20.0, price=50000.0)
        open_long(portfolio, "ETH_EUR", size=20.0, price=2000.0)
        engine = FakeEngine({"EUR": 60.0, "BTC": 0.0004, "ETH": 0.01})

        report = await run_spot_reconciliation(
            portfolio, engine, prices={"BTC_EUR": 50000.0, "ETH_EUR": 2000.0}
        )
        assert report.success is True


class TestFusionFeed:
    def test_symbol_mapping(self):
        feed = BitpandaFusionFeed(api_key="k", config={"pairs": ["BTC_EUR", "ETH_EUR"]})
        assert feed.symbols.to_venue("BTC_EUR") == "BTC-EUR"
        assert feed.symbols.venue_symbols() == ["BTC-EUR", "ETH-EUR"]

    def test_default_pairs(self):
        feed = BitpandaFusionFeed(api_key="k")
        assert feed.pairs == ["BTC_EUR", "ETH_EUR", "SOL_EUR"]

    def test_interface_ist_vollstaendig(self):
        """
        main.py und die Strategien sprechen alle Feeds über dieselbe
        Oberflaeche an — fehlt eine Methode, knallt es erst zur Laufzeit.
        """
        feed = BitpandaFusionFeed(api_key="k")
        for name in ("start", "stop", "get_price", "get_prices", "get_market_data",
                     "get_candles", "get_latest_candle", "get_rsi",
                     "get_volume_spike", "is_connected"):
            assert callable(getattr(feed, name)), f"{name} fehlt"

    def test_leere_zustaende(self):
        feed = BitpandaFusionFeed(api_key="k")
        assert feed.get_price("BTC_EUR") is None
        assert feed.get_prices() == {}
        assert feed.get_candles("BTC_EUR", "5m") is None
        assert feed.get_rsi("BTC_EUR") is None
        assert feed.get_volume_spike("BTC_EUR") is False
        assert feed.is_connected() is False

    def test_candles_und_indikatoren(self, sample_candles):
        feed = BitpandaFusionFeed(api_key="k")
        feed.candle_history["BTC_EUR"] = {"5m": sample_candles}

        df = feed.get_candles("BTC_EUR", "5m", n=30)
        assert df is not None and len(df) == 30

        candle = feed.get_latest_candle("BTC_EUR", "5m")
        assert candle is not None
        assert candle.close == pytest.approx(float(sample_candles["close"].iloc[-1]))
        assert candle.rsi is not None

        assert feed.get_rsi("BTC_EUR", "5m") is not None


class TestIndikatoren:
    def test_ohlcv_konvertierung(self):
        df = ohlcv_to_df([[1700000000000, 1.0, 2.0, 0.5, 1.5, 100.0]])
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 1

    def test_indikatoren_werden_ergaenzt(self, sample_candles):
        roh = sample_candles[["timestamp", "open", "high", "low", "close", "volume"]]
        out = calculate_indicators(roh)
        for col in ("rsi", "bb_upper", "bb_middle", "bb_lower",
                    "ema_9", "ema_21", "vwap", "volume_delta"):
            assert col in out.columns

    def test_zu_wenige_kerzen(self, sample_candles):
        kurz = sample_candles.head(5)[["timestamp", "open", "high", "low", "close", "volume"]]
        assert "rsi" not in calculate_indicators(kurz).columns

    def test_atr(self, sample_candles):
        assert atr(sample_candles, 14) > 0
        assert atr(sample_candles.head(3), 14) == 0.0
