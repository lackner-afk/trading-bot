"""
Tests für core/risk_manager.py.

Der RiskManager ist laut CLAUDE.md sicherheitskritisch: die 2%-Risikogrenze
und das 10%-Tagesdrawdown-Limit dürfen nicht entfernt werden. Diese Tests
halten fest, dass die Grenzen greifen und sich per Config nicht nach oben
umgehen lassen.

Deckt zusätzlich die vor P2 vorhandenen Inkonsistenzen ab: drei
widersprüchliche Positionslimits, ein nie gelesenes max_position_size und
ein REDUCE_SIZE, das der Aufrufer ignorierte.
"""

from datetime import datetime, timedelta

import pytest

from core.market_constraints import MarketConstraints
from core.risk_manager import RiskAction, RiskManager


def rm(config=None, constraints=None):
    return RiskManager(config=config or {}, constraints=constraints)


def check(manager, **kwargs):
    """check_trade mit unauffälligen Defaults."""
    params = dict(
        portfolio_equity=100.0,
        position_size=10.0,
        leverage=1.0,
        current_positions=0,
        consecutive_losses=0,
        daily_drawdown=0.0,
    )
    params.update(kwargs)
    return manager.check_trade(**params)


class TestSpotBeschraenkung:
    def test_hebel_wird_geblockt(self):
        result = check(rm(), leverage=5.0)
        assert result.action == RiskAction.BLOCK
        assert "Spot" in result.reason

    def test_leverage_eins_ist_erlaubt(self):
        assert check(rm(), leverage=1.0).action == RiskAction.ALLOW

    def test_spot_check_kommt_vor_allem_anderen(self):
        """
        Steht bewusst als Check 0: auf einem Spot-Venue ist ein gehebelter
        Trade nicht ausführbar, unabhängig von jeder Risikorechnung.
        """
        result = check(rm(), leverage=5.0, daily_drawdown=0.5, consecutive_losses=10)
        assert result.action == RiskAction.BLOCK
        assert "Spot" in result.reason

    def test_margin_modus_erlaubt_hebel(self):
        manager = rm(constraints=MarketConstraints(spot_only=False, max_leverage=20))
        assert check(manager, leverage=5.0).action == RiskAction.ALLOW


class TestHarteGrenzen:
    def test_risiko_pro_trade_nicht_erhoehbar(self):
        """Die 2%-Grenze darf sich per Config nicht nach oben umgehen lassen."""
        manager = rm({"max_risk_per_trade": 0.50})
        assert manager.max_risk_per_trade == RiskManager.MAX_RISK_PER_TRADE

    def test_positionsgroesse_nicht_erhoehbar(self):
        manager = rm({"max_position_size": 0.90})
        assert manager.max_position_size == RiskManager.MAX_POSITION_SIZE

    def test_positionsgroesse_ist_reduzierbar(self):
        """Vorher wurde der Config-Wert gar nicht gelesen."""
        assert rm({"max_position_size": 0.10}).max_position_size == 0.10

    def test_drawdown_grenze_nicht_erhoehbar(self):
        assert rm({"max_daily_drawdown": 0.50}).max_daily_drawdown == \
            RiskManager.MAX_DAILY_DRAWDOWN

    def test_leverage_grenze_nicht_erhoehbar(self):
        assert rm({"max_leverage": 500}).max_leverage == RiskManager.MAX_LEVERAGE

    def test_positionsanzahl_nicht_erhoehbar(self):
        assert rm({"max_concurrent_positions": 99}).max_concurrent_positions == \
            RiskManager.MAX_CONCURRENT_POSITIONS


class TestRiskActions:
    def test_tagesdrawdown_schliesst_alles(self):
        result = check(rm(), daily_drawdown=0.11)
        assert result.action == RiskAction.CLOSE_ALL

    def test_cooldown_nach_verlustserie(self):
        manager = rm({"cooldown_seconds": 300})
        result = check(manager, consecutive_losses=3)
        assert result.action == RiskAction.COOLDOWN
        assert manager.cooldown_until is not None

    def test_cooldown_blockt_folgetrades(self):
        manager = rm()
        manager.cooldown_until = datetime.now() + timedelta(seconds=60)
        assert check(manager).action == RiskAction.COOLDOWN

    def test_positionslimit_nutzt_config_wert(self):
        """
        Vorher stand hier die Klassenkonstante (5), während die Config 2
        sagte — drei widersprüchliche Obergrenzen im Zusammenspiel mit
        main.py.
        """
        manager = rm({"max_concurrent_positions": 2})
        assert check(manager, current_positions=1).action == RiskAction.ALLOW
        assert check(manager, current_positions=2).action == RiskAction.BLOCK

    def test_zu_grosse_position_wird_reduziert(self):
        result = check(rm(), portfolio_equity=100.0, position_size=50.0)
        assert result.action == RiskAction.REDUCE_SIZE
        assert result.suggested_size == pytest.approx(20.0)

    def test_risiko_pro_trade_reduziert(self):
        """2% von 100 EUR bei 5% SL-Abstand -> max 40 EUR Position."""
        result = check(rm(), portfolio_equity=100.0, position_size=19.0,
                       sl_distance_pct=0.50)
        assert result.action == RiskAction.REDUCE_SIZE
        assert result.suggested_size == pytest.approx(4.0)

    def test_macro_multiplikator_reduziert(self):
        result = check(rm(), macro_risk_multiplier=0.5)
        assert result.action == RiskAction.REDUCE_SIZE
        assert result.suggested_size == pytest.approx(5.0)

    def test_drawdown_reduziert_groesse(self):
        result = check(rm(), daily_drawdown=0.05)
        assert result.action == RiskAction.REDUCE_SIZE
        assert result.suggested_size < 10.0

    def test_normaler_trade_wird_erlaubt(self):
        assert check(rm()).action == RiskAction.ALLOW


class TestPositionSizing:
    def test_size_from_risk_trifft_zielrisiko(self):
        """2% von 100 EUR bei 2% SL-Abstand = 100 EUR, gedeckelt auf 20%."""
        manager = rm()
        assert manager.size_from_risk(100.0, 0.02) == pytest.approx(20.0)

    def test_size_from_risk_respektiert_positionslimit(self):
        manager = rm({"max_position_size": 0.10})
        assert manager.size_from_risk(100.0, 0.001) == pytest.approx(10.0)

    def test_size_from_risk_ohne_stop(self):
        assert rm().size_from_risk(100.0, 0.0) == pytest.approx(5.0)

    def test_kelly_bei_positivem_erwartungswert(self):
        size = rm().calculate_position_size(100.0, win_rate=0.6,
                                            avg_win=2.0, avg_loss=1.0)
        assert 0 < size <= 20.0

    def test_kelly_bei_negativem_erwartungswert(self):
        """Verlustsystem -> keine Position."""
        size = rm().calculate_position_size(100.0, win_rate=0.2,
                                            avg_win=1.0, avg_loss=1.0)
        assert size == 0.0

    def test_kelly_fallback_ohne_historie(self):
        size = rm().calculate_position_size(100.0, win_rate=0.0,
                                            avg_win=0.0, avg_loss=0.0)
        assert size == pytest.approx(5.0)
