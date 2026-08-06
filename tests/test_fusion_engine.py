"""
Tests für die Bitpanda-Fusion-Anbindung.

Zwei Dinge stehen im Vordergrund:

1. **Präzision und Mindestgrößen.** Fehlende Rundung auf Tick-/Lot-Size und
   ungeprüfte Mindestordergrößen sind die häufigste Ursache dafür, dass eine
   Erstintegration beim ersten echten Versuch abgelehnt wird. Bei 100 EUR
   Kapital und 20 % Positionsgröße liegt eine Order bei ~20 EUR — nah genug
   an typischen Minima, um das ernst zu nehmen.

2. **Keine stillen Fehlschläge.** Eine abgelehnte Order muss ein
   `success=False`-Result liefern, damit `_close_position` die Position offen
   lässt statt sie lokal wegzubuchen.
"""

import pytest

from core.bitpanda_fusion_engine import BitpandaFusionOrderEngine
from core.fusion_client import (
    FusionAPIError,
    FusionClient,
    FusionEndpoints,
    PairInfo,
    _precision_from,
)
from core.market_constraints import MarketConstraints
from core.order_engine import OrderStatus
from data.symbols import SymbolRegistry, base_asset, from_venue, quote_asset, to_venue


class FakeClient:
    """FusionClient-Ersatz ohne Netzwerk."""

    def __init__(self, pairs=None, order_response=None, raise_on_create=None):
        self.pairs = pairs if pairs is not None else {
            "BTC-EUR": PairInfo(
                symbol="BTC-EUR", base="BTC", quote="EUR",
                min_amount=0.0001, min_notional=10.0,
                amount_precision=6, price_precision=2,
            )
        }
        self.order_response = order_response or {
            "order_id": "abc-123", "status": "filled",
            "average_price": "50000.00", "filled_amount": "0.0004",
        }
        self.raise_on_create = raise_on_create
        self.created = []
        self.cancelled = []
        self.balances = {"EUR": 100.0, "BTC": 0.002}
        self._id_counter = 0

    async def get_pairs(self, force=False):
        return self.pairs

    async def create_order(self, **kwargs):
        if self.raise_on_create:
            raise self.raise_on_create
        self.created.append(kwargs)
        # Echte Venues vergeben je Order eine eigene ID
        self._id_counter += 1
        response = dict(self.order_response)
        response["order_id"] = f"{response.get('order_id', 'ord')}-{self._id_counter}"
        return response

    async def get_order(self, order_id):
        return self.order_response

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    async def list_orders(self, pair=None, status="open"):
        return []

    async def get_balances(self):
        return self.balances

    async def close(self):
        return None


def engine(client=None, shadow=False, **config):
    cfg = {"shadow_mode": shadow}
    cfg.update(config)
    return BitpandaFusionOrderEngine(
        api_key="test-key", config=cfg, client=client or FakeClient()
    )


class TestSymbolMapping:
    def test_fusion_nutzt_bindestrich(self):
        """Fusion erwartet BTC-EUR, nicht BTC/EUR — das war der Fallstrick."""
        assert to_venue("BTC_EUR", "fusion") == "BTC-EUR"
        assert from_venue("BTC-EUR", "fusion") == "BTC_EUR"

    def test_ccxt_venues_nutzen_slash(self):
        assert to_venue("BTC_EUR", "kraken") == "BTC/EUR"
        assert from_venue("BTC/EUR", "kraken") == "BTC_EUR"

    def test_roundtrip(self):
        for venue in ("fusion", "kraken", "onetrading", "binance"):
            for symbol in ("BTC_EUR", "ETH_EUR", "SOL_EUR"):
                assert from_venue(to_venue(symbol, venue), venue) == symbol

    def test_base_und_quote(self):
        assert base_asset("BTC_EUR") == "BTC"
        assert quote_asset("BTC_EUR") == "EUR"

    def test_ungueltiges_symbol(self):
        with pytest.raises(ValueError):
            base_asset("BTCEUR")

    def test_registry(self):
        reg = SymbolRegistry(["BTC_EUR", "ETH_EUR"], "fusion")
        assert reg.to_venue("BTC_EUR") == "BTC-EUR"
        assert reg.from_venue("ETH-EUR") == "ETH_EUR"
        assert "BTC_EUR" in reg
        assert len(reg) == 2

    def test_registry_kennt_auch_unbekannte(self):
        """Vorher warf ein Symbol ausserhalb der Liste einen KeyError."""
        reg = SymbolRegistry(["BTC_EUR"], "fusion")
        assert reg.to_venue("XRP_EUR") == "XRP-EUR"


class TestQuoteAlias:
    """
    Kapitaltrennung: der Bot handelt *-EURCV, sieht also nur den
    EURCV-Bestand als Geld. Intern bleibt alles BTC_EUR, weil das Symbol
    in asset_selector/universe_manager/Strategien fest verdrahtet ist.
    """

    def test_alias_tauscht_nur_die_quote(self):
        assert to_venue("BTC_EUR", "fusion", quote_alias="EURCV") == "BTC-EURCV"
        assert to_venue("SOL_EUR", "fusion", quote_alias="EURCV") == "SOL-EURCV"

    def test_rueckmapping_auf_kanonische_quote(self):
        """Ohne das liessen sich Venue-Antworten nicht auf Positionen mappen."""
        assert from_venue("BTC-EURCV", "fusion", quote_alias="EURCV") == "BTC_EUR"

    def test_roundtrip_mit_alias(self):
        for symbol in ("BTC_EUR", "ETH_EUR", "XRP_EUR"):
            venue = to_venue(symbol, "fusion", quote_alias="EURCV")
            assert from_venue(venue, "fusion", quote_alias="EURCV") == symbol

    def test_eur_alias_ist_ein_no_op(self):
        """Default darf das bisherige Verhalten nicht veraendern."""
        assert to_venue("BTC_EUR", "fusion", quote_alias="EUR") == "BTC-EUR"
        assert to_venue("BTC_EUR", "fusion", quote_alias=None) == "BTC-EUR"

    def test_registry_mit_alias(self):
        reg = SymbolRegistry(["BTC_EUR", "ETH_EUR"], "fusion", quote_alias="EURCV")
        assert reg.to_venue("BTC_EUR") == "BTC-EURCV"
        assert reg.from_venue("ETH-EURCV") == "ETH_EUR"

    def test_engine_ordert_auf_dem_alias_paar(self):
        """Der eigentliche Zweck: die echte Order geht auf BTC-EURCV raus."""
        client = FakeClient(pairs={
            "BTC-EURCV": PairInfo(
                symbol="BTC-EURCV", base="BTC", quote="EURCV",
                min_amount=0.0001, min_notional=10.0,
                amount_precision=6, price_precision=2,
            )
        })
        eng = engine(client, quote_asset="EURCV")
        assert eng.quote_asset == "EURCV"

        import asyncio
        asyncio.run(eng.execute_market_order("BTC_EUR", "buy", 50.0, 50000.0))
        assert client.created[0]["pair"] == "BTC-EURCV"

    def test_preflight_meldet_fehlende_alias_paare(self):
        """
        BTC-EUR zu haben heisst nicht, BTC-EURCV zu haben. Ohne diese
        Pruefung wuerde der Bot auf ein nicht existierendes Paar ordern.
        """
        import asyncio

        client = FakeClient(pairs={"BTC-EURCV": PairInfo(
            symbol="BTC-EURCV", base="BTC", quote="EURCV",
            min_amount=0.0001, min_notional=10.0,
            amount_precision=6, price_precision=2,
        )})
        eng = engine(client, quote_asset="EURCV",
                     pairs=["BTC_EUR", "SOL_EUR"])

        report = {"ok": True, "checks": {}, "errors": []}
        asyncio.run(eng._check_configured_pairs(report))

        assert report["ok"] is False
        assert "SOL-EURCV" in report["errors"][0]
        assert "BTC-EURCV" not in report["errors"][0]

    def test_preflight_gruen_wenn_alle_paare_da(self):
        import asyncio

        client = FakeClient(pairs={
            f"{b}-EURCV": PairInfo(
                symbol=f"{b}-EURCV", base=b, quote="EURCV",
                min_amount=0.0001, min_notional=10.0,
                amount_precision=6, price_precision=2,
            ) for b in ("BTC", "SOL")
        })
        eng = engine(client, quote_asset="EURCV", pairs=["BTC_EUR", "SOL_EUR"])

        report = {"ok": True, "checks": {}, "errors": []}
        asyncio.run(eng._check_configured_pairs(report))

        assert report["ok"] is True
        assert report["checks"]["pairs_configured"]["ok"] is True


class TestPraezision:
    def test_menge_wird_gerundet(self):
        info = PairInfo(symbol="BTC-EUR", base="BTC", quote="EUR", amount_precision=6)
        assert info.round_amount(0.123456789) == pytest.approx(0.123457)

    def test_preis_wird_gerundet(self):
        info = PairInfo(symbol="BTC-EUR", base="BTC", quote="EUR", price_precision=2)
        assert info.round_price(50000.987654) == pytest.approx(50000.99)

    def test_mindestmenge_wird_geprueft(self):
        info = PairInfo(symbol="BTC-EUR", base="BTC", quote="EUR", min_amount=0.001)
        assert info.validate(0.0001, 100.0) is not None
        assert info.validate(0.01, 100.0) is None

    def test_mindestvolumen_wird_geprueft(self):
        info = PairInfo(symbol="BTC-EUR", base="BTC", quote="EUR", min_notional=25.0)
        assert info.validate(1.0, 20.0) is not None
        assert info.validate(1.0, 30.0) is None

    def test_maximalmenge(self):
        info = PairInfo(symbol="BTC-EUR", base="BTC", quote="EUR", max_amount=1.0)
        assert info.validate(2.0, 100.0) is not None

    def test_precision_aus_increment(self):
        """Die API kann Präzision als Zahl oder als Schrittweite liefern."""
        assert _precision_from(0.00000001, 2) == 8
        assert _precision_from(0.01, 8) == 2
        assert _precision_from(6, 2) == 6
        assert _precision_from(None, 4) == 4
        assert _precision_from("quatsch", 3) == 3


class TestMarketOrder:
    async def test_kauf_schickt_quote_volumen(self):
        """
        Beim Kauf ist `amount` (EUR) genauer als der Umweg über die
        Base-Menge — der Rundungsfehler entfällt.
        """
        client = FakeClient()
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.success
        call = client.created[0]
        assert call["pair"] == "BTC-EUR"
        assert call["amount"] == pytest.approx(20.0)
        assert "quantity" not in call

    async def test_verkauf_schickt_base_menge(self):
        """Verkauft werden kann nur, was als Base-Asset im Bestand liegt."""
        client = FakeClient()
        await engine(client).execute_market_order("BTC_EUR", "sell", 20.0, 50000.0)
        call = client.created[0]
        assert call["quantity"] == pytest.approx(0.0004)
        assert "amount" not in call

    async def test_client_order_id_wird_mitgeschickt(self):
        client = FakeClient()
        await engine(client).execute_market_order("BTC_EUR", "buy", 20.0, 50000.0)
        assert client.created[0]["client_order_id"]

    async def test_ids_sind_eindeutig(self):
        client = FakeClient()
        eng = engine(client)
        await eng.execute_market_order("BTC_EUR", "buy", 20.0, 50000.0)
        await eng.execute_market_order("BTC_EUR", "buy", 20.0, 50000.0)
        ids = {c["client_order_id"] for c in client.created}
        assert len(ids) == 2

    async def test_fill_preis_und_gebuehren_aus_antwort(self):
        client = FakeClient(order_response={
            "order_id": "x", "status": "filled",
            "average_price": "49950.5", "filled_amount": "0.0004",
            "fee": {"amount": "0.012"},
        })
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.execution_price == pytest.approx(49950.5)
        assert result.total_fees == pytest.approx(0.012)

    async def test_gebuehren_geschaetzt_wenn_nicht_geliefert(self):
        client = FakeClient(order_response={
            "order_id": "x", "status": "filled", "average_price": "50000",
        })
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.total_fees == pytest.approx(20.0 * 0.0006)


class TestFehlschlaege:
    async def test_api_fehler_gibt_success_false(self):
        """
        Entscheidend: nur so laesst _close_position die Position offen,
        statt sie lokal wegzubuchen.
        """
        client = FakeClient(raise_on_create=FusionAPIError(400, "Insufficient funds", "/v1/orders"))
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.success is False
        assert result.order.status == OrderStatus.REJECTED

    async def test_zu_kleine_order_wird_abgelehnt(self):
        client = FakeClient(pairs={
            "BTC-EUR": PairInfo(symbol="BTC-EUR", base="BTC", quote="EUR",
                                min_notional=50.0)
        })
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.success is False
        assert "Minimum" in result.message
        assert client.created == []

    async def test_hebel_wird_geblockt(self):
        result = await engine().execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0, leverage=5.0
        )
        assert result.success is False
        assert "Leverage" in result.message

    async def test_ungueltiger_preis(self):
        result = await engine().execute_market_order("BTC_EUR", "buy", 20.0, 0.0)
        assert result.success is False

    async def test_abgelehnter_status(self):
        client = FakeClient(order_response={"order_id": "x", "status": "rejected"})
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.success is False

    async def test_unbekanntes_paar_blockt_nicht(self):
        """Ohne Handelsregeln wird gewarnt, aber nicht blockiert."""
        client = FakeClient(pairs={})
        result = await engine(client).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.success is True


class TestShadowMode:
    async def test_shadow_ist_default(self):
        eng = BitpandaFusionOrderEngine(api_key="k", client=FakeClient())
        assert eng.shadow_mode is True

    async def test_shadow_schickt_keine_order(self):
        client = FakeClient()
        result = await engine(client, shadow=True).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.success is True
        assert client.created == []
        assert "Shadow" in result.message

    async def test_shadow_simuliert_slippage(self):
        result = await engine(shadow=True).execute_market_order(
            "BTC_EUR", "buy", 20.0, 50000.0
        )
        assert result.execution_price > 50000.0

    async def test_shadow_verkauf_unter_referenz(self):
        result = await engine(shadow=True).execute_market_order(
            "BTC_EUR", "sell", 20.0, 50000.0
        )
        assert result.execution_price < 50000.0


class TestLimitUndPending:
    async def test_limit_order_wird_getrackt(self):
        eng = engine()
        order = await eng.execute_limit_order("BTC_EUR", "buy", 20.0, 49000.0)
        assert order.status == OrderStatus.PENDING
        assert eng.get_pending_orders()

    async def test_fill_wird_erkannt(self):
        client = FakeClient(order_response={
            "order_id": "abc-123", "status": "filled",
            "average_price": "49000", "filled_amount": "0.0004",
        })
        eng = engine(client)
        await eng.execute_limit_order("BTC_EUR", "buy", 20.0, 49000.0)

        filled = await eng.check_pending_orders({})
        assert len(filled) == 1
        assert filled[0].status == OrderStatus.FILLED
        assert eng.get_pending_orders() == []

    async def test_stornierte_order_verlaesst_tracking(self):
        client = FakeClient(order_response={"order_id": "abc-123", "status": "canceled"})
        eng = engine(client)
        await eng.execute_limit_order("BTC_EUR", "buy", 20.0, 49000.0)

        assert await eng.check_pending_orders({}) == []
        assert eng.get_pending_orders() == []

    async def test_on_fill_callback(self):
        seen = []
        client = FakeClient(order_response={
            "order_id": "abc-123", "status": "filled", "average_price": "49000",
        })
        eng = engine(client)
        eng.on_fill = lambda result: seen.append(result)

        await eng.execute_limit_order("BTC_EUR", "buy", 20.0, 49000.0)
        await eng.check_pending_orders({})
        assert len(seen) == 1

    async def test_cancel_all(self):
        eng = engine()
        await eng.execute_limit_order("BTC_EUR", "buy", 20.0, 49000.0)
        await eng.execute_limit_order("ETH_EUR", "buy", 20.0, 3000.0)
        assert await eng.cancel_all_orders() == 2
        assert eng.get_pending_orders() == []

    async def test_cancel_nach_symbol(self):
        eng = engine()
        await eng.execute_limit_order("BTC_EUR", "buy", 20.0, 49000.0)
        await eng.execute_limit_order("ETH_EUR", "buy", 20.0, 3000.0)
        assert await eng.cancel_all_orders("BTC_EUR") == 1
        assert len(eng.get_pending_orders()) == 1


class TestClientKonfiguration:
    def test_endpoints_ueberschreibbar(self):
        """
        Die URL-Pfade sind aus dem CLI abgeleitet, nicht gegen die Doku
        verifiziert — sie müssen ohne Codeänderung korrigierbar sein.
        """
        ep = FusionEndpoints.from_config({"endpoints": {"orders": "/v2/orders"}})
        assert ep.orders == "/v2/orders"
        assert ep.pairs == "/v1/pairs"

    def test_unbekannte_endpoint_keys_ignoriert(self):
        ep = FusionEndpoints.from_config({"endpoints": {"quatsch": "/x"}})
        assert not hasattr(ep, "quatsch")

    def test_auth_header_konfigurierbar(self):
        """
        Der gehostete Public-MCP (mcp.public.bitpanda.com) erwartet
        x-api-key ohne Schema — die Konfigurierbarkeit ist kein toter Code.
        """
        client = FusionClient("k", config={"auth_header": "x-api-key",
                                           "auth_scheme": ""})
        assert client._headers()["x-api-key"] == "k"
        assert "Authorization" not in client._headers()

    def test_default_auth_header(self):
        """Laut docs.bitpanda.com: Authorization: Bearer <BITPANDA_API_KEY>."""
        assert FusionClient("k")._headers()["Authorization"] == "Bearer k"

    def test_key_ist_pflicht(self):
        with pytest.raises(ValueError):
            FusionClient("")

    def test_fehlerklassifikation(self):
        assert FusionAPIError(429, "", "").is_rate_limit
        assert FusionAPIError(429, "", "").is_retryable
        assert FusionAPIError(503, "", "").is_retryable
        assert FusionAPIError(401, "", "").is_auth_error
        assert not FusionAPIError(400, "", "").is_retryable

    def test_antwort_normalisierung(self):
        assert FusionClient._as_list({"data": [1, 2]}) == [1, 2]
        assert FusionClient._as_list([1, 2]) == [1, 2]
        assert FusionClient._as_list(None) == []
        assert FusionClient._as_list({"x": 1}) == [{"x": 1}]

    def test_candle_parsing_liste(self):
        row = FusionClient._parse_candle([1700000000000, 1, 2, 0.5, 1.5, 100])
        assert row == [1700000000000, 1.0, 2.0, 0.5, 1.5, 100.0]

    def test_candle_parsing_objekt(self):
        row = FusionClient._parse_candle({
            "time": "2026-01-01T00:00:00Z", "open": "1", "high": "2",
            "low": "0.5", "close": "1.5", "volume": "100",
        })
        assert row is not None and row[4] == pytest.approx(1.5)

    def test_candle_parsing_sekunden(self):
        row = FusionClient._parse_candle({"t": 1700000000, "o": 1, "h": 2,
                                          "l": 1, "c": 1, "v": 1})
        assert row[0] == 1700000000000

    def test_pair_parsing_leitet_base_quote_ab(self):
        info = FusionClient._parse_pair({"pair": "BTC-EUR"})
        assert info.base == "BTC" and info.quote == "EUR"


class TestSchutzStops:
    """
    Der Bot prueft SL/TP im 1-Sekunden-Loop. Stirbt der Prozess, ist eine
    offene Position ohne venue-seitigen Stop voellig ungeschuetzt — nach dem
    bereits behobenen Exit-Bug der gefaehrlichste Zustand beim Spot-Handel
    mit echtem Kapital.
    """

    async def test_stop_wird_platziert(self):
        client = FakeClient()
        eng = engine(client)

        order_id = await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0)

        assert order_id is not None
        call = client.created[0]
        assert call["order_type"] == "stop_market"
        assert call["stop_price"] == pytest.approx(49000.0)
        assert call["side"] == "sell"
        assert eng.protective_stops["BTC_EUR"] == order_id

    async def test_stop_wird_gerundet(self):
        client = FakeClient()
        await engine(client).place_protective_stop("BTC_EUR", "sell", 0.123456789, 49000.987)

        call = client.created[0]
        assert call["quantity"] == pytest.approx(0.123457)      # amount_precision 6
        assert call["stop_price"] == pytest.approx(49000.99)    # price_precision 2

    async def test_shadow_platziert_nichts(self):
        client = FakeClient()
        eng = engine(client, shadow=True)

        assert await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0) is None
        assert client.created == []

    async def test_fehlschlag_ist_nicht_fatal(self):
        """Die Position bleibt offen und lokal abgesichert — aber es muss auffallen."""
        client = FakeClient(raise_on_create=FusionAPIError(400, "nope", "/v1/orders"))
        eng = engine(client)

        assert await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0) is None
        assert "BTC_EUR" not in eng.protective_stops

    async def test_ungueltige_werte(self):
        eng = engine()
        assert await eng.place_protective_stop("BTC_EUR", "sell", 0.0, 49000.0) is None
        assert await eng.place_protective_stop("BTC_EUR", "sell", 0.001, 0.0) is None

    async def test_stornierung_entfernt_tracking(self):
        client = FakeClient()
        eng = engine(client)
        await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0)

        assert await eng.cancel_protective_stop("BTC_EUR") is True
        assert "BTC_EUR" not in eng.protective_stops
        assert len(client.cancelled) == 1

    async def test_stornierung_ohne_stop(self):
        assert await engine().cancel_protective_stop("BTC_EUR") is True

    async def test_ausgeloester_stop_wird_erkannt(self):
        client = FakeClient(order_response={
            "order_id": "stop", "status": "filled", "average_price": "49000",
        })
        eng = engine(client)
        await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0)

        ausgeloest = await eng.check_protective_stops()

        assert ausgeloest == ["BTC_EUR"]
        assert "BTC_EUR" not in eng.protective_stops

    async def test_offener_stop_bleibt(self):
        client = FakeClient(order_response={"order_id": "stop", "status": "open"})
        eng = engine(client)
        await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0)

        assert await eng.check_protective_stops() == []
        assert "BTC_EUR" in eng.protective_stops

    async def test_stornierter_stop_verlaesst_tracking(self):
        client = FakeClient(order_response={"order_id": "stop", "status": "canceled"})
        eng = engine(client)
        await eng.place_protective_stop("BTC_EUR", "sell", 0.0004, 49000.0)

        assert await eng.check_protective_stops() == []
        assert "BTC_EUR" not in eng.protective_stops


class TestOrdertypen:
    def test_stop_typen_brauchen_stop_price(self):
        from core.fusion_client import STOP_ORDER_TYPES
        assert "stop_market" in STOP_ORDER_TYPES
        assert "stop_limit" in STOP_ORDER_TYPES
        assert "take_profit_limit" in STOP_ORDER_TYPES

    async def test_stop_ohne_preis_wirft(self):
        from core.fusion_client import FusionClient
        client = FusionClient("k")
        with pytest.raises(ValueError, match="stop_price"):
            await client.create_order(pair="BTC-EUR", side="sell",
                                      order_type="stop_market", quantity=0.001)

    async def test_quantity_und_amount_exklusiv(self):
        from core.fusion_client import FusionClient
        client = FusionClient("k")
        with pytest.raises(ValueError):
            await client.create_order(pair="BTC-EUR", side="buy",
                                      order_type="market", quantity=1, amount=10)
