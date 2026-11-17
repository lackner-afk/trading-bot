"""Der Backtester darf nicht stillschweigend auf Zufallskursen rechnen."""
import pytest

import data.backtester as backtester_module
from data.backtester import Backtester
from tests.conftest import run


def test_ohne_marktdaten_bricht_der_backtest_ab(monkeypatch):
    monkeypatch.setattr(backtester_module, 'ccxt_async', None)
    bt = Backtester(config={'data_exchange': 'binance'})
    with pytest.raises(RuntimeError, match='Zufallskursen'):
        run(bt.load_data(['BTC/USDT'], days=5))


def test_zufallsdaten_nur_auf_wunsch(monkeypatch):
    monkeypatch.setattr(backtester_module, 'ccxt_async', None)
    bt = Backtester(config={'data_exchange': 'binance', 'allow_simulated_data': True})
    assert 'BTC/USDT' in run(bt.load_data(['BTC/USDT'], days=5))
