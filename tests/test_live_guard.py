"""Echte Orders bleiben gesperrt, solange der Live-Pfad keine Verkäufe sendet."""
import pytest


def test_live_ohne_shadow_startet_nicht(make_bot, monkeypatch):
    monkeypatch.setenv('ONETRADING_API_KEY', 'test')
    monkeypatch.setenv('ONETRADING_API_SECRET', 'test')
    # Nur im Test: Die Sperre muss greifen, BEVOR eine Engine gebaut wird.
    with pytest.raises(RuntimeError, match='Echte Orders sind gesperrt'):
        make_bot({'general': {'mode': 'live', 'shadow_mode': False}})


def test_settings_yaml_bleibt_paper():
    import yaml
    from tests.conftest import ROOT
    config = yaml.safe_load((ROOT / 'config' / 'settings.yaml').read_text())
    assert config['general']['mode'] == 'paper'
    assert config['general']['live_explicit_confirmation'] is False
    assert config['strategies']['scalper']['enabled'] is False
