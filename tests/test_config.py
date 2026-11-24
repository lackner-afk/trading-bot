"""Welche Coins die beiden Paper-Bots handeln."""
import yaml

import main
from tests.conftest import ROOT


def _load(overlay=None):
    config = yaml.safe_load((ROOT / 'config' / 'settings.yaml').read_text())
    if overlay:
        config = main._deep_merge(config, yaml.safe_load((ROOT / 'config' / overlay).read_text()))
    return config


def test_panda9_handelt_vier_coins_auf_zwei_plaetzen(make_bot):
    bot = make_bot()
    assert bot._pairs == ['BTC_EUR', 'ETH_EUR', 'SOL_EUR', 'XRP_EUR']
    assert bot.risk_manager.max_concurrent_positions == 2


def test_trend1_bleibt_bei_drei_coins():
    config = _load('daily_trend.yaml')
    assert config['strategies']['momentum']['pairs'] == ['BTC_EUR', 'ETH_EUR', 'SOL_EUR']
    assert config['risk']['max_concurrent_positions'] == 3
    assert config['general']['mode'] == 'paper'


def test_alle_drei_bots_starten_mit_1000_euro():
    for overlay in (None, 'daily_trend.yaml', 'ema_4h.yaml'):
        config = _load(overlay)
        assert config['general']['start_capital'] == 1000
        assert config['general']['mode'] == 'paper'


def test_nur_swing1_darf_shorten():
    assert _load()['exchange_rules']['spot_only'] is True
    assert _load('daily_trend.yaml')['exchange_rules']['spot_only'] is True
    swing = _load('ema_4h.yaml')
    assert swing['exchange_rules']['spot_only'] is False
    assert swing['risk']['max_leverage'] == 1
    assert swing['strategies']['daily_trend']['rule'] == 'ema'
    assert swing['strategies']['momentum']['pairs'] == ['BTC_EUR', 'ETH_EUR', 'SOL_EUR', 'XRP_EUR']
