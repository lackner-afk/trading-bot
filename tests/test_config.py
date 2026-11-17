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
