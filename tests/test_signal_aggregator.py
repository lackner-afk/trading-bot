"""
Tests für strategies/signal_aggregator.py.

Der wichtigste Test im ganzen Repo: der Skalen-Mismatch zwischen
Faktor-Scores ([0,1]) und dem konfigurierten Schwellwert (3.5) hat dafür
gesorgt, dass der Bot über seine gesamte Laufzeit keinen einzigen Trade
ausgeführt hat. Diese Tests halten die Invariante fest, damit das nicht
noch einmal passieren kann.
"""

import pytest

from core.market_constraints import MarketConstraints
from strategies.factors.base import FactorResult
from strategies.signal_aggregator import SignalAggregator


def factor(name, score, direction=None, confidence=1.0):
    return FactorResult(name=name, score=score, confidence=confidence,
                        direction=direction, reason="test")


def bullish_factors(score=0.9):
    """Drei technische Faktoren + Sentiment, alle klar long."""
    return [
        factor("multi_timeframe_trend", score, "long"),
        factor("momentum", score, "long"),
        factor("breakout", score, "long"),
        factor("volatility_filter", score),
        factor("sentiment", score, "long"),
    ]


def bearish_factors(score=0.9):
    return [
        factor("multi_timeframe_trend", score, "short"),
        factor("momentum", score, "short"),
        factor("breakout", score, "short"),
        factor("volatility_filter", score),
        factor("sentiment", score, "short"),
    ]


class TestSkalenInvariante:
    def test_score_bleibt_in_null_bis_eins(self):
        """
        Egal welche Faktoren: total_score kann 1.0 nicht überschreiten.
        Genau diese Invariante machte min_confluence_score=3.5 unerreichbar.
        """
        agg = SignalAggregator({"min_confluence_score": 0.01})
        for score in (0.0, 0.25, 0.5, 0.75, 1.0):
            sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(score))
            if sig is not None:
                assert 0.0 <= sig.confluence_score <= 1.0

    def test_maximaler_score_ist_erreichbar(self):
        """Bei perfekten Faktoren muss der Score die Schwelle überschreiten."""
        agg = SignalAggregator({"min_confluence_score": 0.55})
        sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(1.0))
        assert sig is not None
        assert sig.confluence_score == pytest.approx(1.0)

    def test_unerreichbarer_schwellwert_wird_abgelehnt(self):
        """Die alte 0-10-Konfiguration muss beim Start knallen, nicht stumm blockieren."""
        with pytest.raises(RuntimeError, match="unerreichbar"):
            SignalAggregator({"min_confluence_score": 3.5})

        with pytest.raises(RuntimeError, match="unerreichbar"):
            SignalAggregator({"min_confluence_score": 5.8})

    def test_schwellwert_null_wird_abgelehnt(self):
        with pytest.raises(RuntimeError):
            SignalAggregator({"min_confluence_score": 0.0})

    def test_confidence_entspricht_score(self):
        """
        Der Divisor /9.5 ist raus. Vorher lag confidence bei maximal 0.105
        und scheiterte damit garantiert am 0.55-Gate in main.py.
        """
        agg = SignalAggregator({"min_confluence_score": 0.55})
        sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(0.9))
        assert sig is not None
        assert sig.confidence == pytest.approx(sig.confluence_score)
        assert sig.confidence >= 0.55

    def test_anzeige_skala_bleibt_bei_zehn(self):
        agg = SignalAggregator({"min_confluence_score": 0.55})
        sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(0.8))
        assert sig is not None
        assert sig.confluence_score_10 == pytest.approx(sig.confluence_score * 10)
        assert "/10" in sig.reason

    def test_schwacher_score_wird_abgelehnt(self):
        agg = SignalAggregator({"min_confluence_score": 0.55})
        assert agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(0.2)) is None


class TestGewichtsRenormalisierung:
    def test_fehlende_kategorie_druckt_score_nicht(self):
        """
        Fällt der Sentiment-Feed aus, darf der Gesamtscore nicht stumm um den
        Sentiment-Gewichtsanteil einbrechen — das war eine zweite, unabhängige
        Ursache für ausbleibende Signale.
        """
        agg = SignalAggregator({"min_confluence_score": 0.55})

        nur_technisch = [
            factor("multi_timeframe_trend", 0.9, "long"),
            factor("momentum", 0.9, "long"),
            factor("breakout", 0.9, "long"),
        ]
        sig = agg.aggregate("BTC_EUR", 50_000.0, nur_technisch)
        assert sig is not None
        # Ohne Renormalisierung wäre der Score 0.9 * 0.58 = 0.52 und damit
        # unter der Schwelle gewesen.
        assert sig.confluence_score == pytest.approx(0.9)

    def test_leere_faktorliste(self):
        agg = SignalAggregator({"min_confluence_score": 0.55})
        assert agg.aggregate("BTC_EUR", 50_000.0, []) is None


class TestRichtungsVoting:
    def test_klarer_long(self):
        agg = SignalAggregator({"min_confluence_score": 0.55})
        sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(0.9))
        assert sig is not None
        assert sig.direction == "long"
        assert sig.take_profit > 50_000.0
        assert sig.stop_loss < 50_000.0

    def test_gleichstand_wird_abgelehnt(self):
        """Bei Patt ist die Richtung Rauschen, kein Signal."""
        agg = SignalAggregator({"min_confluence_score": 0.1})
        patt = [
            factor("multi_timeframe_trend", 0.9, "long"),
            factor("momentum", 0.9, "short"),
            factor("breakout", 0.9),
            factor("volatility_filter", 0.9),
        ]
        assert agg.aggregate("BTC_EUR", 50_000.0, patt) is None

    def test_confidence_gewichtet_die_stimme(self):
        """Ein unsicherer Faktor soll eine sichere Gegenstimme nicht kippen."""
        agg = SignalAggregator({"min_confluence_score": 0.1, "min_technical_factors": 1})
        gemischt = [
            factor("multi_timeframe_trend", 0.9, "long", confidence=1.0),
            factor("momentum", 0.9, "short", confidence=0.1),
            factor("breakout", 0.9, "long", confidence=1.0),
            factor("volatility_filter", 0.9),
        ]
        sig = agg.aggregate("BTC_EUR", 50_000.0, gemischt)
        assert sig is not None
        assert sig.direction == "long"

    def test_zu_wenige_technische_faktoren(self):
        agg = SignalAggregator({"min_confluence_score": 0.1, "min_technical_factors": 3})
        wenig = [
            factor("multi_timeframe_trend", 0.9, "long"),
            factor("sentiment", 0.9, "long"),
        ]
        assert agg.aggregate("BTC_EUR", 50_000.0, wenig) is None


class TestSpotModus:
    def test_leverage_ist_immer_eins(self):
        agg = SignalAggregator({"min_confluence_score": 0.55, "base_leverage": 8})
        for score in (0.6, 0.8, 1.0):
            sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(score))
            if sig is not None:
                assert sig.suggested_leverage == 1.0

    def test_leverage_floor_von_zwei_greift_nicht(self):
        """
        _calculate_leverage hatte ein max(2.0, ...) — im Spot-Modus muss der
        Early Return davor greifen, sonst käme nie 1.0 heraus.
        """
        agg = SignalAggregator({"min_confluence_score": 0.55})
        assert agg._calculate_leverage(0.0, "low_vol_chop") == 1.0
        assert agg._calculate_leverage(1.0, "trending") == 1.0

    def test_short_wird_zu_exit_signal(self):
        agg = SignalAggregator({"min_confluence_score": 0.55})
        sig = agg.aggregate("BTC_EUR", 50_000.0, bearish_factors(0.9))
        assert sig is not None
        assert sig.direction == "short"
        assert sig.is_exit_signal is True
        assert "EXIT" in sig.reason

    def test_short_policy_ignore_verwirft(self):
        agg = SignalAggregator(
            {"min_confluence_score": 0.55},
            constraints=MarketConstraints(short_signal_policy="ignore"),
        )
        assert agg.aggregate("BTC_EUR", 50_000.0, bearish_factors(0.9)) is None

    def test_long_ist_nie_exit_signal(self):
        agg = SignalAggregator({"min_confluence_score": 0.55})
        sig = agg.aggregate("BTC_EUR", 50_000.0, bullish_factors(0.9))
        assert sig is not None
        assert sig.is_exit_signal is False

    def test_margin_modus_erlaubt_hebel_und_shorts(self):
        """Gegenprobe: ohne Spot-Beschränkung greift die alte Logik weiter."""
        agg = SignalAggregator(
            {"min_confluence_score": 0.55, "base_leverage": 8},
            constraints=MarketConstraints(
                spot_only=False, allow_short=True, max_leverage=20.0
            ),
        )
        sig = agg.aggregate("BTC_EUR", 50_000.0, bearish_factors(0.9))
        assert sig is not None
        assert sig.direction == "short"
        assert sig.is_exit_signal is False
        assert sig.suggested_leverage > 1.0


class TestConstraints:
    def test_clamp_leverage_spot(self):
        c = MarketConstraints()
        assert c.clamp_leverage(18.0) == 1.0
        assert c.clamp_leverage(1.0) == 1.0

    def test_clamp_leverage_margin(self):
        c = MarketConstraints(spot_only=False, max_leverage=10.0)
        assert c.clamp_leverage(18.0) == 10.0
        assert c.clamp_leverage(5.0) == 5.0
        assert c.clamp_leverage(0.5) == 1.0

    def test_from_config_defaults_auf_spot(self):
        """Fehlt der trading:-Block, gilt der sichere Fall."""
        c = MarketConstraints.from_config({})
        assert c.spot_only is True
        assert c.allow_short is False
        assert c.max_leverage == 1.0

    def test_from_config_liest_block(self):
        c = MarketConstraints.from_config({
            "trading": {"spot_only": False, "allow_short": True, "max_leverage": 5}
        })
        assert c.spot_only is False
        assert c.allow_short is True
        assert c.max_leverage == 5.0


class TestRichtungsKonviktion:
    """
    Richtungslose Filter (Volatilität, Volumen) zahlen voll auf den Score ein,
    sagen aber nichts über die Richtung. Ohne eine eigene Hürde reicht ein
    ruhiger Markt mit "gesunder Volatilität", um die Schwelle zu reissen.
    """

    def test_filter_allein_erzeugen_kein_signal(self):
        agg = SignalAggregator({"min_confluence_score": 0.5, "min_technical_factors": 2})
        nur_filter_tragen = [
            factor("volatility_filter", 1.0),        # richtungslos, hoher Score
            factor("volume_confirmation", 0.9),      # richtungslos, hoher Score
            factor("multi_timeframe_trend", 0.08, "long"),
            factor("momentum", 0.15, "long"),
        ]
        sig = agg.aggregate("BTC_EUR", 50_000.0, nur_filter_tragen)
        assert sig is None

    def test_echte_richtungs_konviktion_geht_durch(self):
        agg = SignalAggregator({"min_confluence_score": 0.5, "min_technical_factors": 2})
        mit_konviktion = [
            factor("volatility_filter", 1.0),
            factor("volume_confirmation", 0.9),
            factor("multi_timeframe_trend", 0.7, "long"),
            factor("momentum", 0.8, "long"),
        ]
        sig = agg.aggregate("BTC_EUR", 50_000.0, mit_konviktion)
        assert sig is not None
        assert sig.direction == "long"

    def test_schwelle_ist_konfigurierbar(self):
        schwach = [
            factor("volatility_filter", 1.0),
            factor("volume_confirmation", 0.9),
            factor("multi_timeframe_trend", 0.2, "long"),
            factor("momentum", 0.2, "long"),
        ]
        streng = SignalAggregator({"min_confluence_score": 0.5, "min_technical_factors": 2,
                                   "min_directional_score": 0.35})
        locker = SignalAggregator({"min_confluence_score": 0.5, "min_technical_factors": 2,
                                   "min_directional_score": 0.1})
        assert streng.aggregate("BTC_EUR", 50_000.0, schwach) is None
        assert locker.aggregate("BTC_EUR", 50_000.0, schwach) is not None
