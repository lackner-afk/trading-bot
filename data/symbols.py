"""
Zentrales Symbol-Mapping zwischen Bot-Format und Venue-Formaten.

Vorher lag diese Logik dreifach als naives `replace('_', '/')` im Code
(kraken_feed, onetrading_ccxt_feed, live_order_engine) — jeweils ohne
Rückmapping in der Order-Engine, weshalb sich Exchange-Antworten nicht
sauber auf lokale Positionen zurückführen liessen.

Ein globales `replace` reicht ohnehin nicht mehr: Bitpanda Fusion nutzt
`BTC-EUR` mit Bindestrich, CCXT-Venues `BTC/EUR` mit Slash.

Kanonisches Bot-Format ist durchgehend `BTC_EUR` (Underscore) — so in
settings.yaml, in Portfolio.positions und in allen Strategien.

## Quote-Alias (Kapitaltrennung)

Wer den Bot auf einem Konto laufen lässt, auf dem auch privates Geld liegt,
will nicht, dass er den gesamten EUR-Bestand als Handelskapital ansieht.
Bitpanda Fusion bietet neben EUR auch Stablecoins als Quote-Währung an
(EURCV, EURC, USDC). Handelt der Bot `BTC-EURCV` statt `BTC-EUR`, ist sein
Kapital **physisch** getrennt: er sieht nur den EURCV-Bestand, das EUR-Guthaben
existiert für ihn nicht. Gewinne fließen automatisch wieder in EURCV zurück,
der Topf wächst also von selbst mit.

Das darf aber nicht bedeuten, dass überall `BTC_EURCV` steht — `BTC_EUR` ist
in `asset_selector.py`, `universe_manager.py` und den Strategien fest
verdrahtet. Deshalb wird der Quote-Teil **nur an der Venue-Grenze** getauscht:
intern bleibt alles `BTC_EUR`, nach außen geht `BTC-EURCV`.
"""

from typing import Dict, List, Optional

# Trennzeichen je Venue. Kanonisch ist immer der Underscore.
VENUE_SEPARATORS: Dict[str, str] = {
    "kraken": "/",
    "onetrading": "/",
    "binance": "/",
    "fusion": "-",      # Bitpanda Fusion: BTC-EUR
    "bitpanda_fusion": "-",
}

DEFAULT_SEPARATOR = "/"

# Quote-Währung im kanonischen Bot-Format.
CANONICAL_QUOTE = "EUR"


def to_venue(symbol: str, venue: str, quote_alias: str = None) -> str:
    """
    Bot-Format -> Venue-Format.

    `quote_alias` tauscht den Quote-Teil aus, damit der Bot auf einem
    getrennten Kapitaltopf handeln kann (siehe Modul-Docstring).

    >>> to_venue("BTC_EUR", "fusion")
    'BTC-EUR'
    >>> to_venue("BTC_EUR", "kraken")
    'BTC/EUR'
    >>> to_venue("BTC_EUR", "fusion", quote_alias="EURCV")
    'BTC-EURCV'
    """
    sep = VENUE_SEPARATORS.get(venue, DEFAULT_SEPARATOR)
    if quote_alias and quote_alias.upper() != CANONICAL_QUOTE:
        base, _ = split(symbol)
        return f"{base}{sep}{quote_alias.upper()}"
    return symbol.replace("_", sep)


def from_venue(symbol: str, venue: str, quote_alias: str = None) -> str:
    """
    Venue-Format -> Bot-Format. Gegenstück zu to_venue().

    Der Alias wird auf die kanonische Quote zurückgemappt — sonst liessen
    sich Exchange-Antworten nicht auf lokale Positionen zurückführen.

    >>> from_venue("BTC-EUR", "fusion")
    'BTC_EUR'
    >>> from_venue("BTC-EURCV", "fusion", quote_alias="EURCV")
    'BTC_EUR'
    """
    sep = VENUE_SEPARATORS.get(venue, DEFAULT_SEPARATOR)
    if quote_alias and quote_alias.upper() != CANONICAL_QUOTE:
        parts = symbol.split(sep)
        if len(parts) == 2 and parts[1].upper() == quote_alias.upper():
            return f"{parts[0]}_{CANONICAL_QUOTE}"
    return symbol.replace(sep, "_")


def split(symbol: str) -> tuple:
    """
    Zerlegt ein Bot-Symbol in (Base, Quote).

    Wichtig für den Spot-Modus: dort *ist* der Base-Asset-Bestand die
    Position, es gibt kein separates Positions-Konzept auf dem Venue.

    >>> split("BTC_EUR")
    ('BTC', 'EUR')
    """
    parts = symbol.split("_")
    if len(parts) != 2:
        raise ValueError(f"Ungueltiges Symbol-Format: {symbol!r} (erwartet BASE_QUOTE)")
    return parts[0], parts[1]


def base_asset(symbol: str) -> str:
    """Base-Asset eines Paares, z.B. 'BTC' aus 'BTC_EUR'."""
    return split(symbol)[0]


def quote_asset(symbol: str) -> str:
    """Quote-Asset eines Paares, z.B. 'EUR' aus 'BTC_EUR'."""
    return split(symbol)[1]


class SymbolRegistry:
    """
    Mapping für eine feste Symbolliste, mit vorberechneten Dicts.

    Ersetzt die `_to_ccxt`/`_from_ccxt`-Paare in den Feeds. Ein Symbol
    ausserhalb der Liste führte dort zu einem KeyError — hier wird
    stattdessen sauber umgerechnet.
    """

    def __init__(self, symbols: List[str], venue: str, quote_alias: str = None):
        self.venue = venue
        self.quote_alias = quote_alias
        self.symbols = list(symbols)
        self._to = {s: to_venue(s, venue, quote_alias) for s in self.symbols}
        self._from = {v: k for k, v in self._to.items()}

    def to_venue(self, symbol: str) -> str:
        """Bot-Format -> Venue-Format (auch für unbekannte Symbole)."""
        return self._to.get(symbol) or to_venue(symbol, self.venue, self.quote_alias)

    def from_venue(self, symbol: str) -> str:
        """Venue-Format -> Bot-Format (auch für unbekannte Symbole)."""
        return self._from.get(symbol) or from_venue(symbol, self.venue, self.quote_alias)

    def venue_symbols(self) -> List[str]:
        """Alle bekannten Symbole im Venue-Format."""
        return list(self._to.values())

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._to

    def __len__(self) -> int:
        return len(self._to)
