"""
Tests für den historischen Sentiment-Faktor.

Hintergrund: `SentimentFactor` ist im aktuellen Faktorenset der einzige, der
zuverlässig Richtung UND hohen Score liefert. Wendet ein Backtest den heutigen
Fear-&-Greed-Wert auf 90 Tage Historie an, misst er genau den Faktor falsch,
der die Strategie steuert — äußerlich unauffällig. Diese Tests halten fest,
dass der Wert zum Kerzendatum nachgeschlagen wird.
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from strategies.factors.sentiment import SentimentFactor


def candles_ending(day: date, n: int = 60) -> pd.DataFrame:
    """Minimaler Kerzen-DataFrame, dessen letzte Kerze auf `day` fällt."""
    start = datetime.combine(day, datetime.min.time()) - timedelta(minutes=5 * (n - 1))
    return pd.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": [100.0] * n, "high": [101.0] * n,
        "low": [99.0] * n, "close": [100.0] * n, "volume": [10.0] * n,
    })


@pytest.fixture
def factor(tmp_path):
    """Faktor ohne Netzwerk und ohne gemeinsamen Cache."""
    return SentimentFactor({
        "allow_network": False,
        "cache_path": str(tmp_path / "fng.json"),
    })


HISTORIE = {
    date(2026, 1, 1): 15.0,   # Extreme Fear
    date(2026, 1, 2): 35.0,   # Fear
    date(2026, 1, 3): 50.0,   # Neutral
    date(2026, 1, 4): 65.0,   # Greed
    date(2026, 1, 5): 85.0,   # Extreme Greed
}


class TestHistorischerLookup:
    def test_wert_folgt_dem_kerzendatum(self, factor):
        """Der Kern: verschiedene Tage müssen verschiedene Werte liefern."""
        factor.set_history(HISTORIE)

        werte = []
        for tag in sorted(HISTORIE):
            result = factor.calculate("BTC_EUR", candles_ending(tag), 100.0,
                                      regime="low_vol_chop")
            werte.append(result.metadata["fear_and_greed"])

        assert werte == [15.0, 35.0, 50.0, 65.0, 85.0]

    def test_richtung_kippt_mit_dem_index(self, factor):
        """Fear -> long, Neutral -> keine Richtung, Greed -> short (= Exit auf Spot)."""
        factor.set_history(HISTORIE)

        def richtung(tag):
            return factor.calculate("BTC_EUR", candles_ending(tag), 100.0,
                                    regime="low_vol_chop").direction

        assert richtung(date(2026, 1, 1)) == "long"
        assert richtung(date(2026, 1, 2)) == "long"
        assert richtung(date(2026, 1, 3)) is None
        assert richtung(date(2026, 1, 4)) == "short"
        assert richtung(date(2026, 1, 5)) == "short"

    def test_fehlender_tag_nimmt_naechstaelteren(self, factor):
        """Der Index ist der zuletzt bekannte Stand, nicht interpoliert."""
        factor.set_history({date(2026, 1, 1): 20.0, date(2026, 1, 5): 80.0})

        result = factor.calculate("BTC_EUR", candles_ending(date(2026, 1, 3)), 100.0)
        assert result.metadata["fear_and_greed"] == 20.0

    def test_zu_alter_wert_wird_verworfen(self, factor):
        """Ein Monat alter Stand ist keine Aussage über heute."""
        factor.set_history({date(2026, 1, 1): 20.0})
        factor.max_staleness_days = 7

        assert factor.calculate("BTC_EUR", candles_ending(date(2026, 2, 1)), 100.0) is None

    def test_datum_vor_beginn_der_historie(self, factor):
        factor.set_history({date(2026, 6, 1): 20.0})
        assert factor.calculate("BTC_EUR", candles_ending(date(2026, 1, 1)), 100.0) is None

    def test_ohne_historie_kein_ergebnis(self, factor):
        assert factor.calculate("BTC_EUR", candles_ending(date(2026, 1, 1)), 100.0) is None

    def test_as_of_kann_explizit_gesetzt_werden(self, factor):
        factor.set_history(HISTORIE)
        result = factor.calculate("BTC_EUR", candles_ending(date(2026, 1, 5)), 100.0,
                                  as_of=date(2026, 1, 1))
        assert result.metadata["fear_and_greed"] == 15.0

    def test_metadaten_enthalten_das_datum(self, factor):
        factor.set_history(HISTORIE)
        result = factor.calculate("BTC_EUR", candles_ending(date(2026, 1, 2)), 100.0)
        assert result.metadata["as_of"] == "2026-01-02"

    def test_fallback_auf_heute_ohne_timestamp(self, factor):
        """Fehlt die Spalte, wird auf heute zurückgefallen statt zu knallen."""
        factor.set_history({datetime.now().date(): 30.0})
        df = pd.DataFrame({"close": [100.0] * 30})
        result = factor.calculate("BTC_EUR", df, 100.0)
        assert result is not None and result.direction == "long"


class TestRegimeGewichtung:
    def test_extreme_fear_staerker_in_chop(self, factor):
        factor.set_history({date(2026, 1, 1): 15.0})
        candles = candles_ending(date(2026, 1, 1))

        chop = factor.calculate("BTC_EUR", candles, 100.0, regime="low_vol_chop")
        trending = factor.calculate("BTC_EUR", candles, 100.0, regime="trending")

        assert chop.score > trending.score
        assert chop.direction == trending.direction == "long"

    def test_greed_ohne_richtung_im_trend(self, factor):
        """Im Trend soll Sentiment den Trend nicht bekämpfen."""
        factor.set_history({date(2026, 1, 1): 65.0})
        result = factor.calculate("BTC_EUR", candles_ending(date(2026, 1, 1)), 100.0,
                                  regime="trending")
        assert result.direction is None


class TestPersistenz:
    def test_speichern_und_laden(self, tmp_path):
        pfad = str(tmp_path / "fng.json")
        a = SentimentFactor({"allow_network": False, "cache_path": pfad})
        a.set_history(HISTORIE)
        a._save_to_disk()

        b = SentimentFactor({"allow_network": False, "cache_path": pfad})
        assert b._load_from_disk() is True
        assert b._history == HISTORIE

    def test_fehlender_cache(self, tmp_path):
        f = SentimentFactor({"allow_network": False,
                             "cache_path": str(tmp_path / "gibtsnicht.json")})
        assert f._load_from_disk() is False

    def test_kaputter_cache_wirft_nicht(self, tmp_path):
        pfad = tmp_path / "fng.json"
        pfad.write_text("kein json")
        f = SentimentFactor({"allow_network": False, "cache_path": str(pfad)})
        assert f._load_from_disk() is False

    def test_kein_netzwerk_im_offline_modus(self, factor):
        """
        allow_network=False muss jeden Abruf unterbinden — die autouse-Fixture
        in conftest.py würde einen echten Request sofort auffliegen lassen.
        """
        factor._ensure_history()
        assert factor._history == {}


class TestBacktestTauglichkeit:
    def test_is_historical_braucht_mehr_als_einen_tag(self, factor):
        assert factor.is_historical() is False

        factor.set_history({date(2026, 1, 1): 30.0})
        assert factor.is_historical() is False   # ein Tag reicht nicht

        factor.set_history(HISTORIE)
        assert factor.is_historical() is True

    def test_guard_blockt_ohne_historie(self, tmp_path, monkeypatch):
        """backtest.py darf ohne Historie nicht durchlaufen."""
        import backtest

        monkeypatch.setattr(backtest, "__name__", "backtest")
        cfg = {"sentiment": {"allow_network": False,
                             "cache_path": str(tmp_path / "leer.json")}}
        assert backtest.check_sentiment_history(cfg) is False

    def test_guard_erlaubt_bei_deaktiviertem_faktor(self):
        import backtest
        assert backtest.check_sentiment_history({"factors": {"sentiment": False}}) is True

    def test_guard_erlaubt_mit_historie(self, tmp_path):
        import backtest

        pfad = str(tmp_path / "fng.json")
        seed = SentimentFactor({"allow_network": False, "cache_path": pfad})
        seed.set_history(HISTORIE)
        seed._save_to_disk()

        cfg = {"sentiment": {"allow_network": False, "cache_path": pfad}}
        assert backtest.check_sentiment_history(cfg) is True
