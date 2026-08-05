"""
HTTP-Client für die Bitpanda Fusion REST-API.

Quellenlage: `docs.fusion.bitpanda.com` blockt automatisierte Abrufe (HTTP 403).
Die hier abgebildete Oberfläche ist aus dem offiziellen CLI
`bitpanda-labs/bitpanda-fusion-cli` (Go, Apache 2.0) abgeleitet — dessen
Kommandos und Flags belegen, welche Operationen und Parameter existieren.

Gesichert aus dem CLI:
  Base-URL      https://api.fusion.bitpanda.com   (Flag --host / FUSION_HOST)
  Auth          FUSION_API_KEY
  Paar-Format   BTC-EUR
  Ordertypen    limit, market
  Ordergröße    quantity (Base) ODER amount (Quote) — exklusiv
  Order-Status  open, closed, new, partially-filled, filled, canceled,
                filled-and-canceled, done-for-day, rejected
  Candles       Intervalle 1m,5m,10m,15m,30m,1h,4h,1d; limit max 1440
  Pairs         liefert min/max Ordergröße, Tick-Size, Increments

NICHT gesichert: die exakten URL-Pfade und der Auth-Header-Name. Beides ist
deshalb über `FusionEndpoints` bzw. `auth_header` konfigurierbar und wird von
`preflight()` gegen die echte API geprüft. Solange dieser Preflight nicht
sauber durchläuft, darf die Engine nicht scharf geschaltet werden.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://api.fusion.bitpanda.com"

# Vom CLI belegte Order-Status
OPEN_STATES = {"open", "new", "partially-filled"}
FILLED_STATES = {"filled", "closed", "filled-and-canceled"}
DEAD_STATES = {"canceled", "rejected", "done-for-day"}


class FusionAPIError(RuntimeError):
    """Fehler der Fusion-API mit HTTP-Status und Antworttext."""

    def __init__(self, status: int, message: str, path: str = ""):
        self.status = status
        self.path = path
        super().__init__(f"Fusion API {status} bei {path}: {message}")

    @property
    def is_auth_error(self) -> bool:
        return self.status in (401, 403)

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429

    @property
    def is_retryable(self) -> bool:
        return self.status == 429 or self.status >= 500


@dataclass
class FusionEndpoints:
    """
    URL-Pfade der API.

    Die Pfade sind die plausibelste Ableitung aus den CLI-Kommandos, aber
    NICHT gegen die offizielle Doku verifiziert. Sie lassen sich vollständig
    über den `endpoints:`-Block in der Config überschreiben, ohne Code zu ändern.
    """
    tickers: str = "/v1/tickers"
    pairs: str = "/v1/pairs"
    orderbook: str = "/v1/orderbook/{pair}"
    candles: str = "/v1/candles/{pair}"
    assets: str = "/v1/assets"
    account: str = "/v1/account"
    balances: str = "/v1/account/balances"
    orders: str = "/v1/orders"
    order_by_id: str = "/v1/orders/{order_id}"
    trades: str = "/v1/trades"

    @classmethod
    def from_config(cls, config: Dict = None) -> "FusionEndpoints":
        cfg = (config or {}).get("endpoints", {}) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in known})


@dataclass
class PairInfo:
    """
    Handelsregeln eines Paares.

    Ohne Rundung auf `amount_precision` / `price_precision` und ohne Prüfung
    gegen `min_amount` sind Rejects beim ersten Live-Versuch praktisch sicher —
    das ist die häufigste Ursache für gescheiterte Erstintegrationen.
    """
    symbol: str
    base: str
    quote: str
    min_amount: float = 0.0
    max_amount: Optional[float] = None
    min_notional: float = 0.0
    amount_precision: int = 8
    price_precision: int = 2
    raw: Dict = field(default_factory=dict)

    def round_amount(self, amount: float) -> float:
        return round(amount, self.amount_precision)

    def round_price(self, price: float) -> float:
        return round(price, self.price_precision)

    def validate(self, amount: float, notional: float) -> Optional[str]:
        """Gibt einen Fehlertext zurück, wenn die Order nicht platzierbar ist."""
        if self.min_amount and amount < self.min_amount:
            return f"Menge {amount} unter Minimum {self.min_amount} fuer {self.symbol}"
        if self.max_amount and amount > self.max_amount:
            return f"Menge {amount} ueber Maximum {self.max_amount} fuer {self.symbol}"
        if self.min_notional and notional < self.min_notional:
            return f"Volumen {notional:.2f} unter Minimum {self.min_notional} fuer {self.symbol}"
        return None


def _f(value: Any, default: float = 0.0) -> float:
    """Robustes float-Parsing — die API liefert Zahlen teils als Strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _precision_from(value: Any, default: int) -> int:
    """
    Leitet Nachkommastellen aus einer Increment-Angabe ab.

    Die API kann Präzision entweder als Zahl (`8`) oder als Schrittweite
    (`0.00000001`) liefern — beides wird unterstützt.
    """
    if value is None:
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num <= 0:
        return default
    if num >= 1 and float(num).is_integer():
        return int(num)
    text = f"{num:.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else default


class FusionClient:
    """
    Schlanker async REST-Client mit Rate-Limit-Backoff.

    Bewusst getrennt von der OrderEngine: der Client kennt nur HTTP und die
    API-Semantik, die Engine kennt nur die Bot-Semantik. Das macht beide
    einzeln testbar.
    """

    def __init__(self, api_key: str, host: str = DEFAULT_HOST,
                 config: Dict = None, session: aiohttp.ClientSession = None):
        if not api_key:
            raise ValueError("FusionClient benoetigt einen API-Key (FUSION_API_KEY)")

        self.config = config or {}
        self.api_key = api_key
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.endpoints = FusionEndpoints.from_config(self.config)
        # Bitpanda nutzt bei seinen anderen APIs X-Api-Key; falls Fusion
        # Bearer erwartet, hier umstellen statt Code anzufassen.
        self.auth_header = self.config.get("auth_header", "X-Api-Key")
        self.auth_scheme = self.config.get("auth_scheme", "")

        self.timeout = aiohttp.ClientTimeout(total=self.config.get("timeout_seconds", 20))
        self.max_retries = self.config.get("max_retries", 3)

        self._session = session
        self._owns_session = session is None
        self._pairs: Dict[str, PairInfo] = {}

    # ----- HTTP -----------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        value = f"{self.auth_scheme} {self.api_key}".strip() if self.auth_scheme else self.api_key
        return {self.auth_header: value, "Accept": "application/json"}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
            self._owns_session = True
        return self._session

    async def request(self, method: str, path: str, params: Dict = None,
                      json_body: Dict = None, idempotent: bool = True) -> Any:
        """
        Führt einen Request aus, mit Backoff bei 429/5xx.

        `idempotent=False` verhindert jeden Retry — entscheidend für
        Order-Erstellung ohne serverseitige Idempotenz: ein blind
        wiederholtes create_order kann eine Doppelposition erzeugen.
        """
        session = await self._ensure_session()
        url = f"{self.host}{path}"
        attempts = self.max_retries if idempotent else 1
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                async with session.request(
                    method, url, params=params, json=json_body, headers=self._headers()
                ) as response:
                    text = await response.text()

                    if response.status >= 400:
                        error = FusionAPIError(response.status, text[:400], path)
                        if error.is_retryable and attempt < attempts - 1:
                            delay = self._retry_delay(attempt, response.headers)
                            logger.warning(
                                f"Fusion {response.status} bei {path}, "
                                f"Retry {attempt + 1}/{attempts - 1} in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            last_error = error
                            continue
                        raise error

                    if not text:
                        return None
                    try:
                        return await response.json(content_type=None)
                    except Exception:
                        return text

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < attempts - 1:
                    await asyncio.sleep(self._retry_delay(attempt, {}))
                    continue
                raise FusionAPIError(0, str(e), path) from e

        if last_error:
            raise last_error
        return None

    @staticmethod
    def _retry_delay(attempt: int, headers) -> float:
        """Exponentieller Backoff, respektiert Retry-After."""
        retry_after = None
        try:
            retry_after = headers.get("Retry-After")
        except AttributeError:
            pass
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except (TypeError, ValueError):
                pass
        return min(2.0 ** attempt, 30.0)

    async def close(self):
        if self._session is not None and self._owns_session and not self._session.closed:
            await self._session.close()
        self._session = None

    # ----- Marktdaten -----------------------------------------------

    async def get_pairs(self, force: bool = False) -> Dict[str, PairInfo]:
        """Lädt die Handelsregeln aller Paare (gecacht)."""
        if self._pairs and not force:
            return self._pairs

        data = await self.request("GET", self.endpoints.pairs)
        pairs: Dict[str, PairInfo] = {}
        for entry in self._as_list(data):
            info = self._parse_pair(entry)
            if info:
                pairs[info.symbol] = info

        self._pairs = pairs
        logger.info(f"Fusion: {len(pairs)} Handelspaare geladen")
        return pairs

    @staticmethod
    def _parse_pair(entry: Dict) -> Optional[PairInfo]:
        """
        Übersetzt einen Pair-Eintrag. Feldnamen sind nicht dokumentiert,
        deshalb werden die gängigen Varianten akzeptiert.
        """
        if not isinstance(entry, dict):
            return None

        symbol = (entry.get("pair") or entry.get("symbol")
                  or entry.get("instrument_code") or entry.get("name"))
        if not symbol:
            return None

        base = entry.get("base") or entry.get("base_asset") or ""
        quote = entry.get("quote") or entry.get("quote_asset") or ""
        if (not base or not quote) and "-" in str(symbol):
            base, quote = str(symbol).split("-", 1)

        return PairInfo(
            symbol=str(symbol),
            base=str(base),
            quote=str(quote),
            min_amount=_f(entry.get("min_size") or entry.get("min_amount")
                          or entry.get("min_quantity")),
            max_amount=(_f(entry.get("max_size") or entry.get("max_amount")
                           or entry.get("max_quantity")) or None),
            min_notional=_f(entry.get("min_notional") or entry.get("min_amount_quote")),
            amount_precision=_precision_from(
                entry.get("amount_precision") or entry.get("size_increment")
                or entry.get("base_increment"), 8),
            price_precision=_precision_from(
                entry.get("price_precision") or entry.get("tick_size")
                or entry.get("price_increment"), 2),
            raw=entry,
        )

    async def get_tickers(self, pairs: List[str] = None) -> Dict[str, float]:
        """Aktuelle Mid-Preise, gemappt als {venue_symbol: preis}."""
        params = {"pair": ",".join(pairs)} if pairs else None
        data = await self.request("GET", self.endpoints.tickers, params=params)

        prices: Dict[str, float] = {}
        for entry in self._as_list(data):
            if not isinstance(entry, dict):
                continue
            symbol = entry.get("pair") or entry.get("symbol") or entry.get("instrument_code")
            price = (entry.get("mid") or entry.get("price") or entry.get("last")
                     or entry.get("last_price"))
            if symbol and price is not None:
                value = _f(price)
                if value > 0:
                    prices[str(symbol)] = value
        return prices

    async def get_candles(self, pair: str, interval: str = "5m",
                          limit: int = 200) -> List[List[float]]:
        """
        OHLCV im CCXT-Format [[ts_ms, open, high, low, close, volume], ...].

        Fusion liefert laut CLI die Intervalle 1m, 5m, 10m, 15m, 30m, 1h, 4h, 1d;
        `limit` ist auf 1440 begrenzt.
        """
        path = self.endpoints.candles.format(pair=pair)
        params = {"interval": interval, "limit": min(int(limit), 1440)}
        if "{pair}" not in self.endpoints.candles:
            params["pair"] = pair

        data = await self.request("GET", path, params=params)
        candles = []
        for entry in self._as_list(data):
            row = self._parse_candle(entry)
            if row:
                candles.append(row)
        candles.sort(key=lambda r: r[0])
        return candles

    @staticmethod
    def _parse_candle(entry: Any) -> Optional[List[float]]:
        """Akzeptiert Listen- wie Objektform."""
        if isinstance(entry, (list, tuple)) and len(entry) >= 6:
            return [int(_f(entry[0])), _f(entry[1]), _f(entry[2]),
                    _f(entry[3]), _f(entry[4]), _f(entry[5])]

        if not isinstance(entry, dict):
            return None

        ts = entry.get("time") or entry.get("timestamp") or entry.get("t")
        if ts is None:
            return None

        if isinstance(ts, str):
            from datetime import datetime
            try:
                ts_ms = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
            except ValueError:
                return None
        else:
            ts_ms = int(_f(ts))
            if ts_ms < 10_000_000_000:      # Sekunden statt Millisekunden
                ts_ms *= 1000

        return [
            ts_ms,
            _f(entry.get("open") or entry.get("o")),
            _f(entry.get("high") or entry.get("h")),
            _f(entry.get("low") or entry.get("l")),
            _f(entry.get("close") or entry.get("c")),
            _f(entry.get("volume") or entry.get("v")),
        ]

    # ----- Konto ----------------------------------------------------

    async def get_balances(self) -> Dict[str, float]:
        """Verfügbare Bestände je Asset, z.B. {'EUR': 100.0, 'BTC': 0.001}."""
        data = await self.request("GET", self.endpoints.balances)
        balances: Dict[str, float] = {}

        if isinstance(data, dict) and "balances" in data:
            data = data["balances"]

        if isinstance(data, dict):
            for asset, value in data.items():
                if isinstance(value, dict):
                    value = value.get("available") or value.get("total") or 0
                balances[str(asset).upper()] = _f(value)
            return balances

        for entry in self._as_list(data):
            if not isinstance(entry, dict):
                continue
            asset = entry.get("asset") or entry.get("currency") or entry.get("code")
            amount = (entry.get("available") if entry.get("available") is not None
                      else entry.get("balance") or entry.get("total") or entry.get("amount"))
            if asset:
                balances[str(asset).upper()] = _f(amount)
        return balances

    # ----- Orders ---------------------------------------------------

    async def create_order(self, pair: str, side: str, order_type: str,
                           quantity: float = None, amount: float = None,
                           limit_price: float = None,
                           client_order_id: str = None) -> Dict:
        """
        Legt eine Order an.

        `quantity` ist die Base-Menge, `amount` das Quote-Volumen — laut CLI
        schliessen sich beide gegenseitig aus. Der Request wird **nie**
        automatisch wiederholt: ohne serverseitige Idempotenz-Garantie könnte
        ein Retry eine zweite Position eröffnen.
        """
        if (quantity is None) == (amount is None):
            raise ValueError("Genau eines von quantity oder amount angeben")

        body: Dict[str, Any] = {"pair": pair, "side": side, "type": order_type}
        if quantity is not None:
            body["quantity"] = str(quantity)
        else:
            body["amount"] = str(amount)
        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if client_order_id:
            body["client_order_id"] = client_order_id

        return await self.request("POST", self.endpoints.orders,
                                  json_body=body, idempotent=False)

    async def get_order(self, order_id: str) -> Dict:
        return await self.request("GET", self.endpoints.order_by_id.format(order_id=order_id))

    async def cancel_order(self, order_id: str) -> bool:
        await self.request("DELETE", self.endpoints.order_by_id.format(order_id=order_id),
                           idempotent=False)
        return True

    async def list_orders(self, pair: str = None, status: str = "open") -> List[Dict]:
        params = {}
        if pair:
            params["pair"] = pair
        if status:
            params["status"] = status
        data = await self.request("GET", self.endpoints.orders, params=params or None)
        return [e for e in self._as_list(data) if isinstance(e, dict)]

    async def get_fee_tier(self) -> Optional[float]:
        """
        Tatsächliche Gebührenrate des Kontos (als Bruchteil, z.B. 0.0025).

        Fusion staffelt über 7 Stufen nach 30-Tage-Volumen; Level 1 sind
        0,25 %. Maker und Taker sind identisch. Die Rate kann als Prozentzahl
        (0.25) oder als Bruchteil (0.0025) kommen — beides wird erkannt.
        """
        data = await self.request("GET", self.endpoints.account)
        if not isinstance(data, dict):
            return None

        for key in ("fee", "fee_rate", "taker_fee", "trading_fee", "current_fee"):
            value = data.get(key)
            if isinstance(value, dict):
                value = value.get("taker") or value.get("rate") or value.get("value")
            if value is None:
                continue
            rate = _f(value, -1.0)
            if rate < 0:
                continue
            # 0.25 bedeutet 0,25 % — nicht 25 %
            return rate / 100.0 if rate > 0.02 else rate

        return None

    # ----- Diagnose -------------------------------------------------

    async def preflight(self) -> Dict[str, Any]:
        """
        Prüft vor dem Scharfschalten, ob Auth und Pfade stimmen.

        Nötig, weil die exakten URL-Pfade und der Auth-Header-Name aus der
        öffentlich zugänglichen Quelle nicht hervorgehen. Lieber hier sauber
        scheitern als beim ersten echten Order-Versuch.
        """
        report: Dict[str, Any] = {"ok": False, "checks": {}, "errors": []}

        async def probe(name: str, coro):
            try:
                result = await coro
                report["checks"][name] = {"ok": True, "detail": self._describe(result)}
                return result
            except FusionAPIError as e:
                report["checks"][name] = {"ok": False, "detail": str(e)}
                report["errors"].append(f"{name}: {e}")
                if e.is_auth_error:
                    report["errors"].append(
                        f"{name}: Auth fehlgeschlagen - FUSION_API_KEY und "
                        f"auth_header (aktuell '{self.auth_header}') pruefen"
                    )
                return None

        await probe("pairs", self.get_pairs(force=True))
        await probe("tickers", self.get_tickers())
        await probe("balances", self.get_balances())

        report["ok"] = not report["errors"]
        return report

    @staticmethod
    def _describe(result: Any) -> str:
        if isinstance(result, dict):
            return f"{len(result)} Eintraege"
        if isinstance(result, list):
            return f"{len(result)} Eintraege"
        return type(result).__name__

    @staticmethod
    def _as_list(data: Any) -> List:
        """Normalisiert Antworten, die als Liste oder gewrappt kommen können."""
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "results", "items", "candles", "orders",
                        "tickers", "pairs", "trades"):
                inner = data.get(key)
                if isinstance(inner, list):
                    return inner
            return [data]
        return []
