"""
Tests für den Dauerbetrieb.

Die Testphase läuft 30+ Tage unbeaufsichtigt auf einem VPS. Drei Mängel
hätten das gefährdet: unbegrenzt wachsende Logs, ein `systemctl stop`, das
ins SIGKILL läuft, und eine Min-Notional-Falle, die den Bot still und
dauerhaft aufhören lässt zu handeln.
"""

import asyncio
import logging
from pathlib import Path

import pytest

import main as main_module


class TestLogging:
    def test_rotation_und_stdout(self, tmp_path, monkeypatch):
        """
        Rotation gegen volllaufende Platten, StreamHandler damit journalctl
        und deploy/status.sh überhaupt etwas sehen — vorher gab es nur einen
        FileHandler, das Monitoring lief ins Leere.
        """
        from logging.handlers import RotatingFileHandler

        monkeypatch.chdir(tmp_path)
        bot = object.__new__(main_module.TradingBot)
        bot._setup_logging('INFO')

        handlers = logging.getLogger().handlers
        assert any(isinstance(h, RotatingFileHandler) for h in handlers)
        assert any(type(h) is logging.StreamHandler for h in handlers)

    def test_log_level_aus_config(self, tmp_path, monkeypatch):
        """general.log_level wurde vorher nirgends ausgewertet."""
        monkeypatch.chdir(tmp_path)
        bot = object.__new__(main_module.TradingBot)

        bot._setup_logging('WARNING')
        assert logging.getLogger().level == logging.WARNING

        bot._setup_logging('DEBUG')
        assert logging.getLogger().level == logging.DEBUG

    def test_unbekanntes_level_faellt_auf_info(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bot = object.__new__(main_module.TradingBot)
        bot._setup_logging('quatsch')
        assert logging.getLogger().level == logging.INFO

    def test_handler_werden_nicht_dupliziert(self, tmp_path, monkeypatch):
        """Zweimaliger Aufruf (Konstruktor ruft es zweimal) darf nicht stapeln."""
        monkeypatch.chdir(tmp_path)
        bot = object.__new__(main_module.TradingBot)
        bot._setup_logging('INFO')
        bot._setup_logging('INFO')
        assert len(logging.getLogger().handlers) == 2


class TestGeordnetesHerunterfahren:
    """
    Der Signal-Handler setzte vorher nur running=False. Die Reporting-Loops
    schliefen aber bis zu 3600 s und prüften das Flag erst danach — systemctl
    stop lief in den 90-s-Timeout und dann in SIGKILL, wodurch stop() mit
    cancel_all_orders() nie durchlief.
    """

    async def test_sleep_wird_vom_stop_unterbrochen(self):
        bot = object.__new__(main_module.TradingBot)
        bot.running = True
        bot._stop_event = asyncio.Event()

        async def stopper():
            await asyncio.sleep(0.01)
            bot.request_stop()

        asyncio.create_task(stopper())

        loop = asyncio.get_event_loop()
        start = loop.time()
        weiter = await bot._sleep(30)      # würde ohne Fix 30 s dauern
        dauer = loop.time() - start

        assert weiter is False
        assert dauer < 1.0

    async def test_sleep_wartet_normal_aus(self):
        bot = object.__new__(main_module.TradingBot)
        bot.running = True
        bot._stop_event = asyncio.Event()

        assert await bot._sleep(0.01) is True

    async def test_sleep_meldet_stop_ueber_running(self):
        bot = object.__new__(main_module.TradingBot)
        bot.running = False
        bot._stop_event = asyncio.Event()

        assert await bot._sleep(0.01) is False

    async def test_sleep_ohne_event(self):
        """Vor start() gibt es noch kein Event — darf nicht knallen."""
        bot = object.__new__(main_module.TradingBot)
        bot.running = True
        bot._stop_event = None

        assert await bot._sleep(0.01) is True

    def test_request_stop_ohne_event(self):
        bot = object.__new__(main_module.TradingBot)
        bot.running = True
        bot._stop_event = None

        bot.request_stop()
        assert bot.running is False

    async def test_request_stop_setzt_event(self):
        bot = object.__new__(main_module.TradingBot)
        bot.running = True
        bot._stop_event = asyncio.Event()

        bot.request_stop()
        assert bot.running is False
        assert bot._stop_event.is_set()


class TestMinNotionalFalle:
    """
    Die Positionsgröße ist ein fester Anteil des Equity. Fällt das Equity
    unter min_order_notional / max_position_size, liegt JEDE künftige Order
    unter dem Minimum — der Bot handelt nie wieder, ohne dass etwas kaputt
    wäre. Vorher passierte das mit einer INFO-Zeile pro Signal.
    """

    def test_schwelle_ist_berechenbar(self):
        from core.market_constraints import MarketConstraints

        constraints = MarketConstraints(min_order_notional_eur=10.0)
        max_position_size = 0.15
        schwelle = constraints.min_order_notional_eur / max_position_size

        assert schwelle == pytest.approx(66.67, abs=0.01)

    def test_aktuelle_config_haelt_abstand(self):
        """Bei 100 EUR Startkapital muss die Order über dem Minimum liegen."""
        import yaml

        cfg = yaml.safe_load(Path('config/settings.yaml').read_text())
        kapital = cfg['general']['start_capital']
        anteil = cfg['risk']['max_position_size']
        minimum = cfg['trading']['min_order_notional_eur']

        assert kapital * anteil >= minimum, (
            f"Order waere {kapital * anteil:.2f} EUR, Minimum {minimum} EUR"
        )

    def test_puffer_bis_zum_stillstand(self):
        """
        Dokumentiert, wie viel Drawdown die Konfiguration verträgt, bevor der
        Bot dauerhaft aufhört zu handeln.
        """
        import yaml

        cfg = yaml.safe_load(Path('config/settings.yaml').read_text())
        kapital = cfg['general']['start_capital']
        schwelle = (cfg['trading']['min_order_notional_eur']
                    / cfg['risk']['max_position_size'])
        verkraftbarer_drawdown = 1 - schwelle / kapital

        # Das 10%-Tagesdrawdown-Limit muss deutlich vorher greifen
        assert verkraftbarer_drawdown > cfg['risk']['max_daily_drawdown']


class TestDeployKonfiguration:
    @staticmethod
    def _code_lines(pfad: str) -> list:
        """Nur echte Zeilen — Kommentare zaehlen nicht als Konfiguration."""
        return [line.strip() for line in Path(pfad).read_text().splitlines()
                if line.strip() and not line.strip().startswith('#')]

    def test_kein_hartcodierter_pfad(self):
        """deploy.sh lief vorher nur von einem bestimmten Rechner."""
        lines = self._code_lines('deploy/deploy.sh')
        assert not any('/Users/' in line for line in lines)
        assert any(line.startswith('PROJECT_ROOT=') for line in lines)

    def test_startlimit_im_unit_block(self):
        """
        StartLimitIntervalSec/Burst gehoeren in [Unit]. In [Service] werden
        sie von systemd ignoriert und das Restart-Rate-Limit greift nicht.
        """
        lines = self._code_lines('deploy/setup-server.sh')
        unit = lines.index('[Unit]')
        service = lines.index('[Service]')
        limit = next(i for i, line in enumerate(lines)
                     if line.startswith('StartLimitIntervalSec'))

        assert unit < limit < service

    def test_timeout_stop_im_service_block(self):
        lines = self._code_lines('deploy/setup-server.sh')
        service = lines.index('[Service]')
        timeout = next(i for i, line in enumerate(lines)
                       if line.startswith('TimeoutStopSec'))
        assert timeout > service



class TestChecklisteVenue:
    def test_prueft_fusion_key(self, tmp_path, monkeypatch):
        """
        Vorher wurde nur auf ONETRADING_* geprueft — auch wenn Fusion das
        konfigurierte Ziel-Venue ist.
        """
        import tools.paper_to_live_checklist as checklist

        monkeypatch.setattr(checklist, 'PROJECT_ROOT', tmp_path)
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config' / 'settings.yaml').write_text("live:\n  venue: fusion\n")

        (tmp_path / 'config' / 'secrets.env').write_text("ONETRADING_API_KEY=x\nONETRADING_API_SECRET=y\n")
        ok, msg = checklist.check_secrets()
        assert ok is False and 'BITPANDA_API_KEY' in msg

        (tmp_path / 'config' / 'secrets.env').write_text("BITPANDA_API_KEY=echter-key\n")
        ok, msg = checklist.check_secrets()
        assert ok is True

    def test_alter_fusion_key_bleibt_fallback(self, tmp_path, monkeypatch):
        """
        Bitpanda vergibt nur einen Key mit Scopes. Bestehende Installationen
        mit FUSION_API_KEY duerfen davon nicht brechen — main.py liest
        beide Namen in derselben Reihenfolge.
        """
        import tools.paper_to_live_checklist as checklist

        monkeypatch.setattr(checklist, 'PROJECT_ROOT', tmp_path)
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config' / 'settings.yaml').write_text("live:\n  venue: fusion\n")
        (tmp_path / 'config' / 'secrets.env').write_text(
            "BITPANDA_API_KEY=\nFUSION_API_KEY=alter-key\n"
        )

        ok, msg = checklist.check_secrets()
        assert ok is True and 'FUSION_API_KEY' in msg

    def test_leerer_key_zaehlt_nicht(self, tmp_path, monkeypatch):
        import tools.paper_to_live_checklist as checklist

        monkeypatch.setattr(checklist, 'PROJECT_ROOT', tmp_path)
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config' / 'settings.yaml').write_text("live:\n  venue: fusion\n")
        (tmp_path / 'config' / 'secrets.env').write_text("BITPANDA_API_KEY=\n")

        ok, msg = checklist.check_secrets()
        assert ok is False and 'leer' in msg

    def test_onetrading_venue_prueft_alte_keys(self, tmp_path, monkeypatch):
        import tools.paper_to_live_checklist as checklist

        monkeypatch.setattr(checklist, 'PROJECT_ROOT', tmp_path)
        (tmp_path / 'config').mkdir()
        (tmp_path / 'config' / 'settings.yaml').write_text("live:\n  venue: onetrading\n")
        (tmp_path / 'config' / 'secrets.env').write_text(
            "ONETRADING_API_KEY=x\nONETRADING_API_SECRET=y\n"
        )

        ok, _ = checklist.check_secrets()
        assert ok is True
