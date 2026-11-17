"""RiskManager: harte Grenzen lassen sich per Config nicht lockern."""
import pytest

from core.risk_manager import RiskAction, RiskManager


def test_config_kann_harte_grenzen_nicht_lockern():
    rm = RiskManager({'max_risk_per_trade': 0.5, 'max_daily_drawdown': 0.9, 'max_leverage': 500})
    assert rm.max_risk_per_trade == 0.02
    assert rm.max_daily_drawdown == 0.10
    assert rm.max_leverage == 50


def test_size_from_risk():
    rm = RiskManager()
    # 2 % Risiko bei 20 % Stop-Abstand → 10 % des Kapitals
    assert rm.size_from_risk(1000, 0.20) == pytest.approx(100)
    # enger Stop: gedeckelt auf 20 % des Kapitals
    assert rm.size_from_risk(1000, 0.01) == pytest.approx(200)


def test_tageslimit_meldet_close_all():
    rm = RiskManager()
    check = rm.check_trade(1000, 100, 1, 0, 0, daily_drawdown=0.10)
    assert check.action == RiskAction.CLOSE_ALL


def test_zu_grosse_position_wird_gekuerzt():
    rm = RiskManager()
    check = rm.check_trade(100, 30, 1, 0, 0, daily_drawdown=0.0, sl_distance_pct=0.01)
    assert check.action == RiskAction.REDUCE_SIZE
    assert check.suggested_size == pytest.approx(20)


def test_halb_kelly():
    rm = RiskManager()
    # p=0,6, b=1 → Kelly 0,2 → Halb-Kelly 0,1
    assert rm.calculate_position_size(1000, 0.6, 10, 10) == pytest.approx(100)
    # negativer Erwartungswert → keine Position
    assert rm.calculate_position_size(1000, 0.3, 10, 10) == 0


def test_mindestorder_ausnahme_nur_fuer_genau_diese_groesse():
    rm = RiskManager()
    assert rm.check_trade(100, 30, 1, 0, 0, 0.0, exchange_min_order=30).action == RiskAction.ALLOW
    # größer als die Mindestorder: wieder gekürzt
    assert rm.check_trade(100, 40, 1, 0, 0, 0.0, exchange_min_order=30).action == RiskAction.REDUCE_SIZE
    # mehr als das halbe Kapital: keine Ausnahme
    assert rm.check_trade(50, 30, 1, 0, 0, 0.0, exchange_min_order=30).action == RiskAction.REDUCE_SIZE
    # das Tageslimit schlägt die Ausnahme
    assert rm.check_trade(100, 30, 1, 0, 0, 0.10, exchange_min_order=30).action == RiskAction.CLOSE_ALL
