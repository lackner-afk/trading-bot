#!/usr/bin/env python3
"""
Trading-Bot Haupt-Orchestrator
Koordiniert Datenfeeds, Strategien und Execution

Unterstützt zwei Modi:
- Paper (Default): Simulierte Orders + Kraken- oder Fusion-Feed (general.data_feed)
- Live: Echte Orders auf One Trading via LiveOrderEngine + Reconciliation
"""

import asyncio
import signal
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import yaml
from dotenv import load_dotenv

# Lokale Imports
from core.portfolio import Portfolio
from core.risk_manager import RiskManager, RiskAction
from core.order_engine import OrderEngine
from core.live_order_engine import LiveOrderEngine
from core.reconciliation import run_startup_reconciliation
from data.kraken_feed import KrakenFeed
from data.fusion_feed import FusionFeed
from data.onetrading_ccxt_feed import OneTradingCCXTFeed
from strategies.crypto_scalper import CryptoScalper, SignalType
from strategies.momentum import MomentumStrategy
from strategies.ml_predictor import MLPredictor
from strategies.confluence_strategy import ConfluenceStrategy  # New 2026 multi-factor system
from strategies.daily_trend import DailyTrendStrategy
from notifications.reporter import Reporter


class TradingBot:
    """
    Haupt-Bot-Klasse

    Orchestriert alle Komponenten im async Event-Loop.

    Unterstützt Paper- und Live-Modus (gesteuert über config['general']['mode']).
    Im Live-Modus werden echte Orders auf One Trading ausgeführt + Reconciliation
    beim Start durchgeführt.
    """

    def __init__(self, config_path: str = 'config/settings.yaml'):
        # Logging einrichten
        self._setup_logging()

        # Konfiguration laden
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger('TradingBot')

        # Komponenten initialisieren
        self._init_components()

        # State
        self.running = False
        self.start_time = None

        # Preise älter als das gelten als unbrauchbar — dann wird nicht gehandelt.
        self._max_price_age = float(
            self.config.get('general', {}).get('max_price_age_seconds', 60)
        )
        self._prices_stale = False          # für Zustandswechsel-Logging
        self._bg_tasks: set = set()         # harte Referenzen auf Hintergrund-Tasks

    def _setup_logging(self):
        """Konfiguriert Logging — schreibt in bot.log"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler('bot.log', encoding='utf-8'),
            ]
        )

    def _load_config(self, config_path: str) -> Dict:
        """Lädt Konfiguration aus YAML"""
        path = Path(config_path)
        if not path.exists():
            self.logger.warning(f"Config nicht gefunden: {config_path} - verwende Defaults")
            return self._default_config()

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Environment-Variablen laden
        load_dotenv('config/secrets.env')

        return config

    def _default_config(self) -> Dict:
        """Standard-Konfiguration — IMMER Paper-Modus als sichere Default"""
        return {
            'general': {
                'mode': 'paper',
                'live_explicit_confirmation': False,
                'start_capital': 10000,
                'base_currency': 'USDT'
            },
            'strategies': {
                'momentum': {'enabled': True, 'leverage': 10},
                'scalper': {'enabled': False, 'leverage': 20},
                'ml': {'enabled': True}
            },
            'risk': {
                'max_risk_per_trade': 0.02,
                'max_daily_drawdown': 0.10,
                'max_leverage': 50
            }
        }

    def _init_components(self):
        """Initialisiert alle Bot-Komponenten (Paper oder Live je nach Config)"""
        general = self.config.get('general', {})
        risk_config = self.config.get('risk', {})
        strategy_config = self.config.get('strategies', {})
        fees_config = self.config.get('fees', {})

        mode = general.get('mode', 'paper')
        self.is_live = mode == 'live'

        # Core (Portfolio + Risk immer gleich)
        self.portfolio = Portfolio(
            start_capital=general.get('start_capital', 10000)
        )
        self.risk_manager = RiskManager(config=risk_config)

        # Trading-Pairs aus Momentum oder Scalper Config
        momentum_config = strategy_config.get('momentum', {})
        scalper_config = strategy_config.get('scalper', {})
        pairs = momentum_config.get('pairs', scalper_config.get('pairs', []))
        daily_trend_config = strategy_config.get('daily_trend', {})
        if daily_trend_config.get('enabled'):
            # Die Trendfolge braucht ihre eigenen Paare im Feed (Tageskerzen).
            pairs = list(daily_trend_config.get('pairs', pairs))

        if self.is_live:
            # === LIVE MODE ===
            import os
            api_key = os.getenv('ONETRADING_API_KEY')
            api_secret = os.getenv('ONETRADING_API_SECRET')

            if not api_key or not api_secret:
                raise RuntimeError(
                    "LIVE MODE AKTIVIERT, aber ONETRADING_API_KEY / ONETRADING_API_SECRET fehlen in secrets.env!"
                )

            self.logger = logging.getLogger('TradingBot')  # re-fetch after possible config load
            self.logger.critical("=== LIVE MODE INITIALISIERT ===")
            self.logger.critical("Verwende OneTradingCCXTFeed + LiveOrderEngine")

            # Shadow Mode gehört zu general:, nicht zu fees: — dort hätte ihn
            # niemand vermutet, und der alte Default False bedeutete: wer live
            # schaltet, handelt sofort mit echtem Geld. Jetzt umgekehrt: ohne
            # ausdrückliches shadow_mode: false wird nichts echt platziert.
            shadow = general.get('shadow_mode', True)
            if shadow:
                self.logger.critical(
                    "SHADOW MODE: Orders werden simuliert, NICHT an die Börse geschickt."
                )
            else:
                self.logger.critical(
                    "!!! SHADOW MODE AUS — es werden ECHTE Orders mit ECHTEM GELD platziert !!!"
                )

            # Echte Execution Engine
            self.order_engine = LiveOrderEngine(
                api_key=api_key,
                api_secret=api_secret,
                config={**fees_config, 'shadow_mode': shadow}
            )

            # Echter One Trading Feed (mit Keys für Balance etc.)
            self.crypto_feed = OneTradingCCXTFeed(
                api_key=api_key,
                api_secret=api_secret,
                config={'pairs': pairs}
            )
        else:
            # === PAPER MODE (Standard) ===
            self.order_engine = OrderEngine(config=fees_config)
            feed_name = str(general.get('data_feed', 'kraken')).lower()
            if feed_name == 'fusion':
                # Bitpanda Fusion liefert auch Marktdaten nur mit API-Key.
                import os
                fusion_key = os.getenv('FUSION_API_KEY') or os.getenv('BITPANDA_API_KEY')
                if not fusion_key:
                    raise RuntimeError(
                        "general.data_feed ist 'fusion', aber FUSION_API_KEY fehlt in secrets.env."
                    )
                self.crypto_feed = FusionFeed(api_key=fusion_key, config={'pairs': pairs})
            else:
                # Kraken als Default für Paper (gute EUR-Paare, kein Key nötig)
                self.crypto_feed = KrakenFeed(config={'pairs': pairs})

        # Strategien
        self.momentum = MomentumStrategy(config=momentum_config)
        self.scalper = CryptoScalper(config=scalper_config)
        self.ml_predictor = MLPredictor(config=strategy_config.get('ml', {}))

        # New Multi-Factor Confluence Strategy (2026 overhaul)
        self.use_confluence_strategy = self.config.get('strategies', {}).get('confluence', {}).get('enabled', False)
        if self.use_confluence_strategy:
            self.confluence_strategy = ConfluenceStrategy.create_default(
                self.config.get('strategies', {}).get('confluence', {})
            )
            self.logger.info("ConfluenceStrategy (neues Multi-Factor System) aktiviert")
        else:
            self.confluence_strategy = None

        # Trendfolge auf Tageskerzen (Long/Flat, Spot) — siehe docs/TREND_TAGESBASIS.md
        self.daily_trend: Optional[DailyTrendStrategy] = None
        if daily_trend_config.get('enabled'):
            self.daily_trend = DailyTrendStrategy(config=daily_trend_config)
            self.logger.info(
                f"DailyTrendStrategy aktiviert: {', '.join(self.daily_trend.pairs)} | "
                f"SMA{self.daily_trend.params.ma_days} "
                f"+{self.daily_trend.params.entry_buffer_pct:.0%}/-{self.daily_trend.params.exit_buffer_pct:.0%} | "
                f"Notstopp {self.daily_trend.params.max_loss_pct:.0%}"
            )

        # Phase 6: Regime tracking for change alerting
        self._last_regime_name: Optional[str] = None
        self._last_regime_confidence: float = 0.0

        # Phase 6: Macro event window tracking (for alerting)
        self._was_in_macro_event: bool = False
        self._last_macro_event_name: Optional[str] = None

        # Phase 6: Lightweight Factor Performance Attribution (wins/losses per factor)
        # Updated when confluence trades are closed
        self.factor_attribution: Dict[str, Dict[str, float]] = {}  # factor_name -> {"wins": , "losses": , "pnl": }

        # Phase 6: Cache last confluence factor breakdown per symbol (for later attribution on close)
        self._last_confluence_breakdowns: Dict[str, dict] = {}

        # Verlust-Cooldown pro Symbol (Confluence): symbol -> gesperrt bis
        self._confluence_loss_block: Dict[str, datetime] = {}

        # Reporter
        self.reporter = Reporter(config=self.config.get('notifications', {}))

        # Telegram-Config
        self._telegram_config = self.config.get('notifications', {}).get('telegram', {})

        # Callback setzen (funktioniert für beide Engines)
        self.order_engine.on_fill = self._on_order_fill

    async def _on_order_fill(self, result):
        """Callback wenn Order gefüllt wird"""
        self.logger.info(f"Order gefüllt: {result.order.symbol} @ {result.execution_price}")

    async def start(self):
        """Startet den Bot"""
        self.running = True
        self.start_time = datetime.now()

        mode = self.config.get('general', {}).get('mode', 'paper')
        live_confirmed = self.config.get('general', {}).get('live_explicit_confirmation', False)

        # ============================================================
        # ⚠️  EXTREM LAUTE LIVE-MODE WARNUNG (Phase 0 Sicherheitsmaßnahme)
        # ============================================================
        if mode == 'live':
            self.logger.critical("=" * 70)
            self.logger.critical("!!! LIVE-MODUS AKTIVIERT !!!")
            self.logger.critical("!!! ECHTES GELD WIRD VERWENDET !!!")
            self.logger.critical("=" * 70)
            self.logger.critical(f"Mode: {mode}")
            self.logger.critical(f"live_explicit_confirmation: {live_confirmed}")
            self.logger.critical("Starte in 10 Sekunden... (Ctrl+C zum Abbrechen)")
            self.logger.critical("=" * 70)

            # Harte Verzögerung + mehrfache Warnung
            import time
            for i in range(10, 0, -1):
                self.logger.critical(f"  LIVE START IN {i} SEKUNDEN...")
                time.sleep(1)

            if not live_confirmed:
                self.logger.critical("ABBRUCH: live_explicit_confirmation ist nicht true!")
                self.logger.critical("Setze in settings.yaml general.live_explicit_confirmation: true")
                raise RuntimeError("Live mode blocked: missing explicit confirmation flag")

            self.logger.critical("!!! LETZTE WARNUNG: ECHTE ORDERS WERDEN JETZT PLATZIERT !!!")
        else:
            self.logger.info("=" * 60)
            self.logger.info("Paper-Trading-Bot startet (Paper-Modus)")
            self.logger.info("=" * 60)

        # Startup-Banner
        self.reporter.print_startup_banner(self.config)

        # Komponenten starten
        await self.crypto_feed.start()
        await self.reporter.start()

        # === Reconciliation im Live-Modus (Phase 3/4) ===
        if self.is_live:
            self.logger.critical("Starte Reconciliation mit One Trading (Exchange als Source of Truth)...")
            try:
                report = await run_startup_reconciliation(self.portfolio, self.order_engine)
                if not report.success:
                    self.logger.critical("RECONCILIATION FEHLGESCHLAGEN — Live-Start wird aus Sicherheitsgründen abgebrochen!")
                    raise RuntimeError("Reconciliation failed. Bot refuses to start in live mode.")
                self.logger.critical("Reconciliation erfolgreich abgeschlossen.")
            except Exception as recon_err:
                self.logger.critical(f"Reconciliation Fehler: {recon_err}")
                raise RuntimeError("Reconciliation error — aborting live start for safety") from recon_err

        # ML-Modelle initial trainieren
        if self.config.get('strategies', {}).get('ml', {}).get('enabled'):
            await self._initial_ml_training()

        # Telegram starten (falls konfiguriert und Token gesetzt)
        import os
        tg = self._telegram_config
        if tg.get('enabled'):
            token = os.environ.get('TELEGRAM_BOT_TOKEN', tg.get('token', ''))
            chat_id = os.environ.get('TELEGRAM_CHAT_ID', tg.get('chat_id', ''))
            if token and chat_id:
                await self.reporter.setup_telegram(token, chat_id)

        # Haupt-Loops starten
        tasks = [
            asyncio.create_task(self._supervise(self._main_loop, 'main')),
            asyncio.create_task(self._supervise(self._risk_check_loop, 'risk')),
            asyncio.create_task(self._supervise(self._reporting_loop, 'reporting')),
            asyncio.create_task(self._supervise(self._telegram_hourly_loop, 'telegram')),
        ]

        if self.daily_trend is not None:
            tasks.append(asyncio.create_task(self._supervise(self._daily_trend_loop, 'daily_trend')))

        if self.use_confluence_strategy:
            # Nur das neue Multi-Factor Confluence System (Phase 1-6)
            tasks.append(asyncio.create_task(self._supervise(self._confluence_loop, 'confluence')))
        elif self.daily_trend is None:
            # Alte Strategien (wenn Confluence nicht als Haupt-System aktiviert ist)
            tasks.append(asyncio.create_task(self._supervise(self._momentum_loop, 'momentum')))
            tasks.append(asyncio.create_task(self._supervise(self._scalper_loop, 'scalper')))
            tasks.append(asyncio.create_task(self._supervise(self._ml_loop, 'ml')))

        # Warte auf Beendigung
        try:
            # return_exceptions=True: ein gestorbener Loop darf die übrigen nicht
            # mitreißen. Der Supervisor fängt Fehler ohnehin ab — das hier ist das
            # letzte Netz, damit gather nicht beim ersten Fehler alles abbricht.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for task, result in zip(tasks, results):
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    self.logger.error(f"Loop endete mit Fehler: {result!r}")
        except asyncio.CancelledError:
            self.logger.info("Bot wird beendet...")
        finally:
            # Laufende Tasks sauber abräumen, bevor stop() den Feed schließt.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.stop()

    async def stop(self):
        """Stoppt den Bot graceful"""
        self.running = False
        self.logger.info("Stoppe Bot-Komponenten...")

        await self.crypto_feed.stop()
        await self.reporter.stop()

        # Final Report
        self.reporter.print_daily_report(self.portfolio, self._get_strategy_stats())

        self.logger.info("Bot gestoppt.")

    def _spawn(self, coro):
        """
        Startet einen Hintergrund-Task und hält eine Referenz darauf. Ohne diese
        Referenz darf der GC den Task vor Fertigstellung einsammeln — verlorene
        Telegram-Alerts ohne jede Spur. Exceptions werden geloggt statt verschluckt.
        """
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        task.add_done_callback(self._log_task_exception)
        return task

    def _log_task_exception(self, task):
        """Holt die Exception eines Hintergrund-Tasks ab, damit sie im Log landet."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.error(f"Hintergrund-Task fehlgeschlagen: {exc!r}")

    async def _supervise(self, factory, name: str):
        """
        Hält einen Loop am Leben. Fliegt ihm eine unerwartete Exception um die
        Ohren, wird er nach wachsender Pause neu gestartet, statt über gather()
        den kompletten Bot mitzureißen.
        """
        backoff = 5
        while self.running:
            try:
                await factory()
                return  # regulär beendet (self.running == False)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(
                    f"Loop '{name}' abgestürzt: {e!r} — Neustart in {backoff}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)

    def _check_price_freshness(self, fresh: dict, all_prices: dict):
        """Meldet Zustandswechsel zwischen frischen und veralteten Preisdaten."""
        stale = set(all_prices) - set(fresh)
        if stale and not self._prices_stale:
            self._prices_stale = True
            ages = ", ".join(
                f"{s} {self.crypto_feed.get_price_age(s):.0f}s" for s in sorted(stale)
            )
            self.logger.warning(
                f"[DATEN VERALTET] Handel pausiert für: {ages} "
                f"(Grenze {self._max_price_age:.0f}s)"
            )
            if self.reporter.telegram:
                self._spawn(self.reporter.telegram.send_message(
                    f"⚠️ <b>Datenausfall</b>\nKeine frischen Preise für: {ages}\n"
                    f"Handel und Exit-Prüfung pausieren, bis der Feed liefert."
                ))
        elif not stale and self._prices_stale:
            self._prices_stale = False
            self.logger.info("[DATEN OK] Preise wieder frisch — Handel läuft weiter.")
            if self.reporter.telegram:
                self._spawn(self.reporter.telegram.send_message(
                    "✅ <b>Feed wieder da</b>\nPreise sind aktuell, Handel läuft weiter."
                ))

    async def _main_loop(self):
        """Haupt-Event-Loop"""
        self.logger.info("Haupt-Loop gestartet")

        while self.running:
            try:
                # Preis-Updates verarbeiten. Nur frische Preise fließen weiter —
                # update_position_prices und _check_exit_conditions überspringen
                # Symbole, die hier fehlen, statt auf alten Kursen zu handeln.
                all_prices = self.crypto_feed.get_prices()
                prices = self.crypto_feed.get_prices(max_age_seconds=self._max_price_age)
                self._check_price_freshness(prices, all_prices)
                self.portfolio.update_position_prices(prices)

                # Pending Orders prüfen
                filled = await self.order_engine.check_pending_orders(prices)
                for order in filled:
                    await self._process_filled_order(order)

                # Exit-Conditions prüfen
                await self._check_exit_conditions(prices)

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Fehler im Haupt-Loop: {e}")
                await asyncio.sleep(5)

    def _get_trend(self, candles) -> bool:
        """Ermittelt Trendrichtung anhand 1h EMA9/EMA21. True = Aufwärtstrend, False = Abwärtstrend, None = unklar"""
        if candles is None or len(candles) < 21:
            return None
        import pandas as pd
        ema_9 = candles['ema_9'].iloc[-1]
        ema_21 = candles['ema_21'].iloc[-1]
        if pd.isna(ema_9) or pd.isna(ema_21):
            return None
        return bool(ema_9 > ema_21)

    async def _momentum_loop(self):
        """Momentum-Strategie Loop (alle 30 Sekunden) — 5m-Kerzen mit 1h-Trend-Filter"""
        if not self.config.get('strategies', {}).get('momentum', {}).get('enabled'):
            return

        self.logger.info("Momentum-Loop gestartet")

        while self.running:
            try:
                for symbol in self.momentum.pairs:
                    # 5m-Kerzen für Signal (weniger Rauschen als 1m)
                    candles = self.crypto_feed.get_candles(symbol, '5m')
                    price = self.crypto_feed.get_price(symbol, max_age_seconds=self._max_price_age)

                    if candles is None or price is None:
                        continue

                    signal = self.momentum.analyze(symbol, candles, price)

                    if signal and signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                        # Trend-Filter: 1h-Trend muss Signal bestätigen (robuster als 15m)
                        candles_1h = self.crypto_feed.get_candles(symbol, '1h')
                        trend_up = self._get_trend(candles_1h)

                        if signal.signal_type == SignalType.LONG and trend_up is False:
                            self.logger.info(f"Trend-Filter: {symbol} LONG blockiert (1h Abwärtstrend)")
                            continue
                        if signal.signal_type == SignalType.SHORT and trend_up is True:
                            self.logger.info(f"Trend-Filter: {symbol} SHORT blockiert (1h Aufwärtstrend)")
                            continue

                        await self._execute_signal(signal, strategy_name='momentum')

                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"Fehler im Momentum-Loop: {e}")
                await asyncio.sleep(10)

    async def _scalper_loop(self):
        """Scalper-Strategie Loop (alle 15 Sekunden)"""
        if not self.config.get('strategies', {}).get('scalper', {}).get('enabled'):
            return

        self.logger.info("Scalper-Loop gestartet")

        while self.running:
            try:
                for symbol in self.scalper.pairs:
                    candles = self.crypto_feed.get_candles(symbol, '1m')
                    price = self.crypto_feed.get_price(symbol, max_age_seconds=self._max_price_age)

                    if candles is None or price is None:
                        continue

                    signal = self.scalper.analyze(symbol, candles, price)

                    if signal and signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                        if signal.confidence >= 0.6:
                            await self._execute_signal(signal, strategy_name='scalper')

                await asyncio.sleep(15)

            except Exception as e:
                self.logger.error(f"Fehler im Scalper-Loop: {e}")
                await asyncio.sleep(10)

    async def _ml_loop(self):
        """ML-Predictor Loop (alle 5 Minuten)"""
        if not self.config.get('strategies', {}).get('ml', {}).get('enabled'):
            return

        self.logger.info("ML-Loop gestartet")

        while self.running:
            try:
                for symbol in self.momentum.pairs:
                    # Retrain wenn nötig
                    if self.ml_predictor.should_retrain(symbol):
                        candles = self.crypto_feed.get_candles(symbol, '1m', n=500)
                        if candles is not None:
                            self.ml_predictor.train(candles, symbol)

                    # Prediction machen
                    candles = self.crypto_feed.get_candles(symbol, '1m')
                    if candles is not None:
                        prediction = self.ml_predictor.predict(candles, symbol, sentiment_score=0.0)

                        if prediction and prediction.confidence >= 0.7:
                            self.logger.info(f"ML-Signal: {symbol} {prediction.direction} "
                                           f"(Prob: {prediction.probability:.0%})")

                await asyncio.sleep(300)  # 5 Minuten

            except Exception as e:
                self.logger.error(f"Fehler im ML-Loop: {e}")
                await asyncio.sleep(60)


    @staticmethod
    def _completed_candles(candles, timeframe_minutes: int = 5):
        """
        Schneidet die letzte, noch laufende Kerze ab.

        Die Faktoren (Volumen-Ratio, RSI(2), Breakout) sind auf abgeschlossene
        Kerzen kalibriert — eine halb gefüllte Kerze liefert systematisch
        verzerrte Werte (z.B. Volumen 0.0x) und drückt den Confluence-Score.
        """
        if candles is None or len(candles) == 0:
            return candles
        try:
            last_ts = candles['timestamp'].iloc[-1]
            if hasattr(last_ts, 'to_pydatetime'):
                last_ts = last_ts.to_pydatetime()
            if last_ts.tzinfo is not None:
                last_ts = last_ts.replace(tzinfo=None)
            if datetime.utcnow() < last_ts + timedelta(minutes=timeframe_minutes):
                return candles.iloc[:-1]
        except Exception:
            pass
        return candles

    async def _confluence_loop(self):
        """Neue Multi-Factor Confluence Strategie Loop (Phase 1+ der Überarbeitung)"""
        if not self.use_confluence_strategy or self.confluence_strategy is None:
            return

        self.logger.info("ConfluenceStrategy-Loop gestartet (neues Multi-Factor System)")

        confluence_cfg = self.config.get('strategies', {}).get('confluence', {})
        interval = confluence_cfg.get('interval_seconds', 45)
        # Confidence-Gate aus der Config (confidence == confluence_score auf 0-1-Skala);
        # muss zur kalibrierten min_confluence_score passen, sonst filtert es verdeckt.
        min_conf = confluence_cfg.get('min_signal_confidence', 0.55)

        while self.running:
            try:
                # Phase 6: Macro Event Alerting (prüft auf CPI/FOMC/etc. Fenster)
                self._check_macro_event_alert()

                # Hole aktuelle empfohlene Assets vom UniverseManager
                all_candles = {}
                for symbol in self.momentum.pairs:  # vorerst noch die alten Pairs als Basis
                    candles = self._completed_candles(
                        self.crypto_feed.get_candles(symbol, '5m', n=251))  # 250 fertige Kerzen (200er-EMA)
                    if candles is not None:
                        all_candles[symbol] = candles

                input_count = len(all_candles)
                universe = self.confluence_strategy.get_recommended_assets(all_candles)
                selected_symbols = universe.get("symbols", [])
                selected_count = len(selected_symbols)

                # Permanent health / starvation logging (critical for debugging)
                if selected_count == 0:
                    self.logger.warning(
                        f"[CONFLUENCE HEALTH] Starvation! Input candidates: {input_count}, "
                        f"selected after AssetSelector: {selected_count}. "
                        f"Regime likely too hostile or selector too strict."
                    )
                else:
                    self.logger.info(
                        f"[CONFLUENCE HEALTH] Input candidates: {input_count} → selected: {selected_count}"
                    )

                # Temporary starvation fallback (aggressive test mode)
                if selected_count == 0 and input_count > 0:
                    self.logger.warning("[CONFLUENCE HEALTH] Activating starvation fallback - analyzing raw input list")
                    symbols_to_analyze = list(all_candles.keys())
                else:
                    symbols_to_analyze = selected_symbols

                analyzed = 0
                best_score = 0.0
                best_symbol = None
                regime_name = None

                for symbol in symbols_to_analyze:
                    # Verlust-Cooldown: Symbol nach Verlust-Trade vorübergehend auslassen
                    blocked = self._confluence_loss_block.get(symbol)
                    if blocked and datetime.now() < blocked:
                        continue

                    candles = self._completed_candles(
                        self.crypto_feed.get_candles(symbol, '5m', n=251))  # 250 fertige Kerzen (200er-EMA)
                    price = self.crypto_feed.get_price(symbol, max_age_seconds=self._max_price_age)

                    if candles is None or price is None:
                        continue

                    analyzed += 1

                    # Always get fresh regime for this symbol (even if no signal)
                    regime = self.confluence_strategy.regime_detector.detect(symbol, candles)
                    if regime and regime.name:
                        regime_name = regime.name

                    signal = self.confluence_strategy.analyze_legacy(symbol, candles, price)

                    # Get regime info even for rejected signals
                    cd = getattr(signal, '_confluence_data', None) or {} if signal else {}
                    score = cd.get('confluence_score', 0) if cd else 0
                    if score > best_score:
                        best_score = score
                        best_symbol = symbol

                    if signal:
                        if signal.confidence < min_conf:
                            # Visible rejection reason during test phase
                            self.logger.info(
                                f"[CONFLUENCE REJECT] {symbol} @ {price:.2f} | "
                                f"Conf {signal.confidence:.0%} | Score {score:.2f} | Regime {regime_name or 'unknown'}"
                            )
                    if signal and signal.confidence >= min_conf:
                        # Phase 6: Rich factor attribution logging + console
                        regime_name = None
                        cd = getattr(signal, '_confluence_data', None) or {}
                        regime_obj = cd.get('regime')
                        if regime_obj and hasattr(regime_obj, 'name'):
                            regime_name = regime_obj.name

                        self.logger.info(
                            f"[CONFLUENCE] {signal.signal_type.value.upper()} {symbol} @ {price:.2f} "
                            f"(Conf: {signal.confidence:.0%} | Score: {cd.get('confluence_score', 0):.2f}) | "
                            f"Regime: {regime_name or 'unknown'} | {signal.reason}"
                        )

                        # Detailed per-factor breakdown into the log file (very valuable for debugging)
                        breakdown = cd.get('factor_breakdown', {}) if cd else {}
                        if breakdown:
                            factor_lines = []
                            for fname, fres in sorted(breakdown.items(), key=lambda x: getattr(x[1], 'score', 0), reverse=True):
                                fscore = getattr(fres, 'score', 0)
                                fdir = getattr(fres, 'direction', None) or "-"
                                freason = getattr(fres, 'reason', '')
                                factor_lines.append(f"    • {fname}: score={fscore:.2f} dir={fdir} | {freason}")
                            if factor_lines:
                                self.logger.info("[CONFLUENCE FACTORS]\n" + "\n".join(factor_lines))

                        # Beautiful console table (Phase 6)
                        self.reporter.print_factor_breakdown(signal, regime=regime_name)

                        # Phase 6: Regime Change Alerting (CRITICAL for high-risk regimes)
                        if regime_name:
                            self._check_and_alert_regime_change(regime_name, getattr(regime_obj, 'confidence', 0.0) if regime_obj else 0.0)

                        # Phase 6: Optional Telegram "Why did I take this trade?" message
                        # Default AUS — Trade-Alerts (Open/Close) reichen; per
                        # notifications.telegram.signal_messages: true reaktivierbar.
                        if self.reporter.telegram and self._telegram_config.get('signal_messages', False):
                            breakdown = cd.get('factor_breakdown', {}) if cd else {}
                            top_factor_names = [
                                name.replace("_", " ").title()
                                for name, _ in sorted(breakdown.items(), key=lambda x: getattr(x[1], 'score', 0), reverse=True)[:3]
                            ]
                            self._spawn(
                                self.reporter.send_confluence_signal_decision(
                                    signal, regime=regime_name, top_factors=top_factor_names
                                )
                            )

                        # Phase 6: Remember breakdown for attribution when this trade eventually closes
                        if breakdown:
                            self._last_confluence_breakdowns[signal.symbol] = breakdown

                        await self._execute_confluence_signal(signal)

                # Phase 6 Improvement: Cycle summary for visibility (even when no trade)
                if analyzed > 0:
                    self.logger.info(
                        f"[CONFLUENCE CYCLE] Analyzed {analyzed} symbols | "
                        f"Best: {best_symbol} ({best_score:.2f}) | "
                        f"Regime: {regime_name or 'unknown'} | "
                        f"No high-confluence signal this cycle"
                    )
                else:
                    self.logger.warning("[CONFLUENCE CYCLE] No symbols analyzed this cycle")

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Fehler im Confluence-Loop: {e}")
                await asyncio.sleep(30)

    async def _daily_trend_loop(self):
        """
        Trendfolge auf Tageskerzen: eine Entscheidung je abgeschlossener Tageskerze.

        Der Loop läuft stündlich, handelt aber nur, wenn seit der letzten
        Prüfung eine neue Tageskerze abgeschlossen wurde. So sieht der Bot
        dieselben Kerzen wie der Backtester (tools/backtest_daily_trend.py).
        """
        strat = self.daily_trend
        if strat is None:
            return

        self.logger.info(
            f"DailyTrend-Loop gestartet ({', '.join(strat.pairs)}, Prüfung alle "
            f"{strat.interval_seconds // 60} Min)"
        )

        min_equity = strat.min_equity_for_trade(self.risk_manager.MAX_POSITION_SIZE)
        state = self.portfolio.get_state()
        if state.equity < min_equity:
            self.logger.warning(
                f"[DAILY TREND] Eigenkapital {state.equity:.2f} € liegt unter {min_equity:.0f} €. "
                f"Mit der 20 %-Kappung je Position und {strat.min_order_amount:.0f} € Mindestorder "
                f"kann keine Order platziert werden — der Bot beobachtet nur."
            )

        evaluated: Dict[str, object] = {}   # symbol -> Zeitstempel der zuletzt bewerteten Tageskerze

        while self.running:
            try:
                for symbol in strat.pairs:
                    candles = self._completed_candles(
                        self.crypto_feed.get_candles(symbol, '1d', n=strat.params.warmup_bars + 20),
                        timeframe_minutes=1440,
                    )
                    if candles is None or len(candles) < strat.params.warmup_bars:
                        have = 0 if candles is None else len(candles)
                        self.logger.debug(f"[DAILY TREND] {symbol}: {have}/{strat.params.warmup_bars} Tageskerzen")
                        continue

                    last_ts = candles['timestamp'].iloc[-1]
                    if evaluated.get(symbol) == last_ts:
                        continue

                    price = self.crypto_feed.get_price(symbol, max_age_seconds=self._max_price_age)
                    if price is None:
                        continue   # nächste Runde erneut versuchen

                    position = self.portfolio.positions.get(symbol)
                    if position is not None:
                        if position.market_type == 'daily_trend':
                            should_exit, reason = strat.check_trend_exit(symbol, candles)
                            if should_exit:
                                await self._close_position(symbol, price, reason)
                            else:
                                self.logger.info(f"[DAILY TREND] {symbol}: Trend intakt, Position bleibt")
                    else:
                        signal = strat.analyze(symbol, candles, price)
                        if signal is not None:
                            self.logger.info(f"[DAILY TREND] {symbol}: {signal.reason}")
                            await self._execute_daily_trend_signal(signal)
                        else:
                            sma = float(candles['close'].tail(strat.params.ma_days).mean())
                            self.logger.info(
                                f"[DAILY TREND] {symbol}: kein Einstieg (Schluss "
                                f"{float(candles['close'].iloc[-1]):.2f}, SMA{strat.params.ma_days} {sma:.2f})"
                            )

                    evaluated[symbol] = last_ts

                await asyncio.sleep(strat.interval_seconds)

            except Exception as e:
                self.logger.error(f"Fehler im DailyTrend-Loop: {e!r}")
                await asyncio.sleep(60)

    async def _execute_daily_trend_signal(self, signal):
        """
        Kauft Spot ohne Hebel. Größe: Wunschanteil, dann die harten Grenzen des
        RiskManagers, dann das verfügbare Cash. Unter der Mindestorder der
        Börse wird nicht gehandelt, sondern erklärt, warum.
        """
        strat = self.daily_trend
        if strat is None or signal.symbol in self.portfolio.positions:
            return

        state = self.portfolio.get_state()
        size = strat.target_notional(state.equity)
        sl_distance_pct = strat.params.max_loss_pct

        # check_trade meldet je Aufruf nur eine Reduktion — deshalb wiederholen,
        # bis die Größe alle Regeln erfüllt.
        risk_check = None
        for _ in range(4):
            risk_check = self.risk_manager.check_trade(
                portfolio_equity=state.equity,
                position_size=size,
                leverage=1,
                current_positions=len(state.positions),
                consecutive_losses=self.portfolio.consecutive_losses,
                daily_drawdown=self.portfolio.get_daily_drawdown(),
                sl_distance_pct=sl_distance_pct,
            )
            if (risk_check.action == RiskAction.REDUCE_SIZE
                    and risk_check.suggested_size and risk_check.suggested_size < size):
                size = risk_check.suggested_size
                continue
            break

        if risk_check.action in (RiskAction.BLOCK, RiskAction.COOLDOWN, RiskAction.CLOSE_ALL):
            self.logger.warning(f"[DAILY TREND] {signal.symbol} nicht gekauft: {risk_check.reason}")
            return

        fee_rate = self.config.get('fees', {}).get('crypto_taker', 0.0006)
        size = min(size, state.balance / (1.0 + fee_rate) * 0.999)

        if size < strat.min_order_amount:
            self.logger.warning(
                f"[DAILY TREND] {signal.symbol}: Ordergröße {size:.2f} € unter Mindestorder "
                f"{strat.min_order_amount:.0f} € (Equity {state.equity:.2f} €, Kappung "
                f"{self.risk_manager.MAX_POSITION_SIZE:.0%}). Kein Kauf."
            )
            return

        result = await self.order_engine.execute_market_order(
            symbol=signal.symbol,
            side='buy',
            size=size,
            current_price=signal.price,
            leverage=1,
            strategy='daily_trend',
        )
        if not result.success:
            self.logger.warning(f"[DAILY TREND] Order für {signal.symbol} nicht ausgeführt")
            return

        filled_size = result.order.filled_size or size
        self.portfolio.open_position(
            symbol=signal.symbol,
            side='long',
            size=filled_size,
            price=result.execution_price,
            leverage=1,
            strategy='daily_trend',
            market_type='daily_trend',
            stop_loss=signal.stop_loss,
            take_profit=None,
            fees=result.total_fees,
        )
        self.reporter.print_info(
            f"DAILY TREND: KAUF {signal.symbol} @ {result.execution_price:.2f} € | "
            f"{filled_size:.2f} € | Notstopp {signal.stop_loss:.2f}"
        )
        if self.reporter.telegram and self._telegram_config.get('send_trade_alerts', True):
            msg = (
                f"🟢 <b>TREND-KAUF</b> {signal.symbol}\n"
                f"Einstieg: {result.execution_price:.2f} €\n"
                f"Größe: {filled_size:.2f} € (Spot, kein Hebel)\n"
                f"Notstopp: {signal.stop_loss:.2f} € | Ausstieg, wenn der Tagesschluss unter die "
                f"SMA{strat.params.ma_days} fällt\n"
                f"{signal.reason}"
            )
            self._spawn(self.reporter.telegram.send_message(msg))

    async def _risk_check_loop(self):
        """Risk-Check Loop (alle 5 Minuten)"""
        self.logger.info("Risk-Check-Loop gestartet")

        while self.running:
            try:
                state = self.portfolio.get_state()
                daily_dd = self.portfolio.get_daily_drawdown()
                sharpe = self.portfolio.get_sharpe_ratio()

                metrics = self.risk_manager.get_metrics(state.equity, daily_dd, sharpe)

                if metrics['status'] == 'CRITICAL':
                    self.reporter.print_warning("KRITISCHER DRAWDOWN - Alle Positionen werden geschlossen!")
                    await self._close_all_positions("Risk-Limit erreicht")

                elif metrics['status'] == 'WARNING' and daily_dd > 0:
                    self.reporter.print_warning(f"Drawdown bei {daily_dd:.1%}")

                await asyncio.sleep(300)

            except Exception as e:
                self.logger.error(f"Fehler im Risk-Check: {e}")
                await asyncio.sleep(60)

    async def _reporting_loop(self):
        """Reporting Loop"""
        self.logger.info("Reporting-Loop gestartet")

        # Erster Report nach 5 Minuten
        await asyncio.sleep(300)

        while self.running:
            try:
                state = self.portfolio.get_state()
                positions = self.portfolio.positions
                trades = self.portfolio.get_recent_trades(10)

                # Portfolio-Status ausgeben
                self.reporter.print_portfolio_summary(state)
                self.reporter.print_positions(positions)
                self.reporter.print_recent_trades(trades)

                # Stündlicher Report
                if self.reporter.should_send_hourly_report():
                    metrics = self.risk_manager.get_metrics(
                        state.equity,
                        self.portfolio.get_daily_drawdown(),
                        self.portfolio.get_sharpe_ratio()
                    )
                    self.reporter.print_hourly_report(self.portfolio, metrics)

                # Täglicher Report
                if self.reporter.should_send_daily_report():
                    self.reporter.print_daily_report(
                        self.portfolio,
                        self._get_strategy_stats()
                    )
                    # Phase 6: Factor Attribution Summary (wenn Confluence aktiv war)
                    if self.use_confluence_strategy and self.factor_attribution:
                        attr_text = self.get_factor_attribution_summary()
                        self.reporter.console.print("\n[bold cyan]Phase 6 Attribution[/bold cyan]")
                        self.reporter.console.print(attr_text)

                await asyncio.sleep(3600)  # 1 Stunde

            except Exception as e:
                self.logger.error(f"Fehler im Reporting: {e}")
                await asyncio.sleep(60)

    async def _execute_signal(self, signal, strategy_name: str = 'momentum',
                             regime: str = None, macro_risk_multiplier: float = 1.0):
        """Führt Trading-Signal aus (Momentum oder Scalper)"""
        # Prüfe ob bereits Position für dieses Symbol existiert
        if signal.symbol in self.portfolio.positions:
            return

        state = self.portfolio.get_state()

        # Positions-Obergrenze aus der Config (der RiskManager prüft sie ohnehin;
        # die früher hier hartcodierte 5 widersprach settings.yaml).
        if len(state.positions) >= self.risk_manager.max_concurrent_positions:
            return

        if state.equity < 20:
            return

        # SL-Distanz aus Signal ableiten (ATR-basiert)
        sl_distance_pct = 0.0
        if signal.price > 0 and signal.stop_loss > 0:
            sl_distance_pct = abs(signal.price - signal.stop_loss) / signal.price

        # Risk-Check (Phase 5: regime + macro aware)
        risk_check = self.risk_manager.check_trade(
            portfolio_equity=state.equity,
            position_size=state.equity * 0.1,
            leverage=signal.suggested_leverage,
            current_positions=len(state.positions),
            consecutive_losses=self.portfolio.consecutive_losses,
            daily_drawdown=self.portfolio.get_daily_drawdown(),
            sl_distance_pct=sl_distance_pct if sl_distance_pct > 0 else None,
            regime=regime,
            macro_risk_multiplier=macro_risk_multiplier
        )

        if risk_check.action == RiskAction.BLOCK:
            self.logger.warning(f"Trade blockiert: {risk_check.reason}")
            return

        if risk_check.action == RiskAction.COOLDOWN:
            self.logger.info(f"Cooldown aktiv: {risk_check.reason}")
            return

        # Position Sizing wie im Backtest: Margin ist der Equity-Anteil,
        # das Notional (= size) ergibt sich daraus mal Hebel.
        if sl_distance_pct > 0 and hasattr(signal, 'atr_value') and signal.atr_value > 0:
            margin = self.risk_manager.size_from_risk(state.equity, sl_distance_pct)
        else:
            margin = state.equity * 0.20
        # Skaliert auf Kapital: Min 15% des Equity, Max 25% des Equity
        min_margin = max(10.0, state.equity * 0.15)
        max_margin = state.equity * 0.25
        margin = max(min_margin, min(max_margin, margin))

        if margin > state.balance or state.balance < 20:
            return

        position_size = margin * signal.suggested_leverage

        side = 'buy' if signal.signal_type == SignalType.LONG else 'sell'

        result = await self.order_engine.execute_market_order(
            symbol=signal.symbol,
            side=side,
            size=position_size,
            current_price=signal.price,
            leverage=signal.suggested_leverage,
            strategy=strategy_name
        )

        if result.success:
            # Die tatsächlich gefüllte Größe buchen, nicht die angeforderte —
            # bei einem Partial Fill hielt das Portfolio sonst mehr als die Order.
            filled_size = result.order.filled_size or position_size
            self.portfolio.open_position(
                symbol=signal.symbol,
                side='long' if signal.signal_type == SignalType.LONG else 'short',
                size=filled_size,
                price=result.execution_price,
                leverage=signal.suggested_leverage,
                strategy=strategy_name,
                market_type=strategy_name,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                fees=result.total_fees
            )

            self.reporter.print_info(
                f"{strategy_name.upper()}: {signal.signal_type.value.upper()} {signal.symbol} "
                f"@ ${result.execution_price:.4f} (Conf: {signal.confidence:.0%})"
            )

            # Telegram-Alert bei Eröffnung. Vorher gab es NUR beim Schließen einen
            # Alert — die Eröffnung war bloß Konsolen-Ausgabe.
            if self.reporter.telegram and self._telegram_config.get('send_trade_alerts', True):
                direction = signal.signal_type.value.upper()
                emoji = '🟢' if direction == 'LONG' else '🔴'
                msg = (
                    f"{emoji} <b>TRADE AUF</b> {direction} {signal.symbol}\n"
                    f"Einstieg: {result.execution_price:.4f}\n"
                    f"TP: {signal.take_profit:.4f} | SL: {signal.stop_loss:.4f}\n"
                    f"Größe: {filled_size:.2f} (Hebel {signal.suggested_leverage}x) | "
                    f"Konfidenz {signal.confidence:.0%}"
                )
                self._spawn(self.reporter.telegram.send_message(msg))

    async def _execute_confluence_signal(self, signal):
        """
        Führt ein Signal der neuen ConfluenceStrategy (Multi-Factor System) aus.
        Berücksichtigt Regime + Macro-Risk für besseres Risikomanagement bei häufigerem Traden (Phase 5).
        """
        regime = None
        macro_multiplier = 1.0

        if self.confluence_strategy is not None:
            last_regime = getattr(self.confluence_strategy, '_last_regime', None)
            if last_regime:
                regime = last_regime.name

            macro_filter = next(
                (f for f in self.confluence_strategy.factors if f.name == "macro_news_filter"),
                None
            )
            if macro_filter:
                macro_multiplier = macro_filter.get_risk_multiplier()

        await self._execute_signal(
            signal,
            strategy_name='confluence',
            regime=regime,
            macro_risk_multiplier=macro_multiplier
        )

    def _check_and_alert_regime_change(self, new_regime: str, confidence: float = 0.0):
        """
        Phase 6: Überwacht Regime-Wechsel und sendet Alerts bei kritischen Übergängen.

        Besonders wichtig:
        - Wechsel in high_vol_event oder event_driven → stark reduzierte Positionen + Aufmerksamkeit
        - Wechsel aus low_vol_chop → potenziell gute Trading-Gelegenheiten
        """
        old_regime = self._last_regime_name

        if old_regime == new_regime:
            return  # kein Wechsel

        # Update State
        self._last_regime_name = new_regime
        self._last_regime_confidence = confidence

        if old_regime is None:
            # Erster Durchlauf
            self.logger.info(f"[REGIME] Initiales Regime erkannt: {new_regime} (conf={confidence:.0%})")
            return

        # Log immer
        self.logger.warning(f"[REGIME CHANGE] {old_regime} → {new_regime} (conf={confidence:.0%})")

        # Kritische Regime, bei denen wir laut Alarm schlagen müssen
        critical_regimes = {"high_vol_event", "event_driven"}

        is_critical = new_regime in critical_regimes
        was_critical = old_regime in critical_regimes

        if is_critical or was_critical or new_regime == "low_vol_chop":
            # Baue Alert-Nachricht
            emoji = "🔴" if is_critical else ("🟠" if new_regime == "low_vol_chop" else "🟡")
            direction = "BETRETEN" if is_critical else "VERLASSEN"

            msg = (
                f"{emoji} <b>REGIME WECHSEL</b>\n"
                f"{old_regime} → <b>{new_regime}</b> (Conf {confidence:.0%})\n\n"
            )

            if new_regime == "high_vol_event":
                msg += "⚠️ Hohe Volatilität / Event-Modus! Position Sizes stark reduziert. Sehr vorsichtig traden!"
            elif new_regime == "event_driven":
                msg += "📰 Makro-Event-Fenster (CPI/FOMC/etc.). Risk-Multiplier aktiv. Weniger Exposure!"
            elif new_regime == "low_vol_chop":
                msg += "😴 Sehr ruhiger Markt (Low-Vol Chop). Weniger Signale erwartet. Besser abwarten."
            elif was_critical and new_regime == "trending":
                msg += "✅ Aus kritischem Regime raus in sauberen Trend. Gute Bedingungen möglich."
            else:
                msg += f"Regime-Shift: {old_regime} → {new_regime}. Faktor-Gewichtungen werden automatisch angepasst."

            # Console
            self.reporter.print_warning(f"REGIME CHANGE: {old_regime} → {new_regime}")

            # Telegram nur wenn Event-Alerts aktiviert sind (Default aus —
            # Nici will nur Trade-Open/-Close + periodischen Bericht)
            if self.reporter.telegram and self._telegram_config.get('event_alerts', False):
                self._spawn(self.reporter.telegram.send_message(msg))

    def _check_macro_event_alert(self):
        """
        Phase 6: Prüft ob wir in ein Macro-Event-Fenster (CPI, FOMC, NFP...) reingegangen sind
        oder es verlassen haben und sendet entsprechende Alerts.
        """
        if not self.confluence_strategy:
            return

        macro_filter = next(
            (f for f in self.confluence_strategy.factors if f.name == "macro_news_filter"),
            None
        )
        if not macro_filter:
            return

        active_event = None
        try:
            active_event = macro_filter.calendar.is_in_event_window(
                hours_before=getattr(macro_filter, 'hours_before', 4),
                hours_after=getattr(macro_filter, 'hours_after', 2)
            )
        except Exception:
            return

        currently_in = active_event is not None
        event_name = active_event.name if active_event else None

        # State-Change Detection
        if currently_in and not self._was_in_macro_event:
            # Neu reingegangen
            self._was_in_macro_event = True
            self._last_macro_event_name = event_name
            self.logger.warning(f"[MACRO EVENT] Betreten: {event_name or 'unbekanntes High-Impact Event'}")

            msg = (
                f"📰 <b>MACRO EVENT WINDOW AKTIV</b>\n"
                f"<b>{event_name or 'High-Impact Event'}</b>\n\n"
                f"Der Bot hat das Risiko automatisch reduziert (Risk-Multiplier aktiv).\n"
                f"Erwarte deutlich weniger oder kleinere Positionen in den nächsten Stunden."
            )
            self.reporter.print_warning(f"MACRO EVENT: {event_name}")
            if self.reporter.telegram and self._telegram_config.get('event_alerts', False):
                self._spawn(self.reporter.telegram.send_message(msg))

        elif not currently_in and self._was_in_macro_event:
            # Rausgegangen
            self._was_in_macro_event = False
            last = self._last_macro_event_name or "Macro Event"
            self._last_macro_event_name = None
            self.logger.info(f"[MACRO EVENT] Verlassen: {last}")

            msg = (
                f"✅ <b>MACRO EVENT VORBEI</b>\n"
                f"{last} Fenster geschlossen.\n\n"
                f"Risk-Multiplier zurück auf normal. Volles Exposure wieder möglich."
            )
            self.reporter.print_info(f"Macro Event vorbei: {last}")
            if self.reporter.telegram and self._telegram_config.get('event_alerts', False):
                self._spawn(self.reporter.telegram.send_message(msg))

    def _update_factor_attribution(self, trade, factor_breakdown: Dict):
        """Phase 6: Aktualisiert die per-Factor Win/Loss/PnL Statistik."""
        is_win = trade.pnl > 0
        pnl = trade.pnl

        for fname, fres in factor_breakdown.items():
            if fname not in self.factor_attribution:
                self.factor_attribution[fname] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}

            stats = self.factor_attribution[fname]
            stats["trades"] += 1
            stats["pnl"] += pnl
            if is_win:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        self.logger.info(f"[ATTRIBUTION] Trade {trade.symbol} {'WIN' if is_win else 'LOSS'} {pnl:+.2f}€ → {len(factor_breakdown)} Faktoren aktualisiert")

    def get_factor_attribution_summary(self) -> str:
        """Phase 6: Gibt eine kurze Text-Zusammenfassung der Factor-Performance zurück."""
        if not self.factor_attribution:
            return "Noch keine Attribution-Daten (warte auf geschlossene Confluence-Trades)."

        lines = ["FACTOR ATTRIBUTION (Confluence):"]
        # Sortiere nach PnL absteigend
        sorted_factors = sorted(
            self.factor_attribution.items(),
            key=lambda x: x[1]["pnl"],
            reverse=True
        )
        for name, stats in sorted_factors[:6]:  # Top 6
            wr = stats["wins"] / (stats["wins"] + stats["losses"]) if (stats["wins"] + stats["losses"]) > 0 else 0
            lines.append(
                f"  {name}: {stats['trades']} trades | WR {wr:.0%} | PnL {stats['pnl']:+.2f}€"
            )
        return "\n".join(lines)

    async def _check_exit_conditions(self, prices: Dict[str, float]):
        """Prüft Exit-Bedingungen für alle Positionen"""
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Richtige Strategie für Exit-Check wählen (Phase 5)
            if position.market_type == 'confluence':
                # Dedizierte Confluence-Exits: NUR die ATR-kalibrierten TP/SL aus dem
                # Signal (5x/7x ATR). Kein Trailing-Stop — der Momentum-Fallback mit
                # 0.5%-Trailing hat Gewinner vor dem Ziel gekappt und damit das im
                # Backtest kalibrierte Profil (68% WR) zerstört.
                sl = position.stop_loss
                tp = position.take_profit
                should_exit, reason = False, ''
                if position.side == 'long':
                    if sl and current_price <= sl:
                        should_exit, reason = True, f"Stop-Loss erreicht ({current_price:.2f} <= {sl:.2f})"
                    elif tp and current_price >= tp:
                        should_exit, reason = True, f"Take-Profit erreicht ({current_price:.2f} >= {tp:.2f})"
                else:
                    if sl and current_price >= sl:
                        should_exit, reason = True, f"Stop-Loss erreicht ({current_price:.2f} >= {sl:.2f})"
                    elif tp and current_price <= tp:
                        should_exit, reason = True, f"Take-Profit erreicht ({current_price:.2f} <= {tp:.2f})"

                if should_exit:
                    await self._close_position(symbol, current_price, reason)
                continue

            if position.market_type == 'daily_trend':
                # Trendfolge: der reguläre Ausstieg passiert im DailyTrend-Loop auf
                # Tagesschluss-Basis. Hier nur der Notstopp gegen den Live-Preis.
                sl = position.stop_loss
                if sl and current_price <= sl:
                    await self._close_position(
                        symbol, current_price,
                        f"Notstopp erreicht ({current_price:.2f} <= {sl:.2f})"
                    )
                continue

            if position.market_type == 'momentum':
                strategy = self.momentum
            else:
                strategy = self.scalper

            should_exit, reason = strategy.check_exit_conditions(
                symbol=symbol,
                entry_price=position.entry_price,
                current_price=current_price,
                side=position.side,
                highest_since_entry=strategy.highest_prices.get(symbol),
                stop_loss_price=position.stop_loss if hasattr(position, 'stop_loss') else None,
                take_profit_price=position.take_profit if hasattr(position, 'take_profit') else None
            )

            if should_exit:
                await self._close_position(symbol, current_price, reason)

    async def _close_position(self, symbol: str, price: float, reason: str):
        """Schließt eine Position"""
        position = self.portfolio.positions.get(symbol)
        if not position:
            return

        # Taker-Gebühr aus der Config (Fusion Stufe 1: 0,25 %). Der alte Festwert
        # 0,06 % ließ jeden Exit um den Faktor 4 zu billig aussehen.
        fee_rate = self.config.get('fees', {}).get('crypto_taker', 0.0006)
        fees = position.size * fee_rate

        trade = self.portfolio.close_position(
            symbol=symbol,
            exit_price=price,
            fees=fees,
            strategy=position.market_type
        )

        if trade:
            # Schließungsgrund protokollieren — ohne ihn ist im Nachhinein nicht
            # nachvollziehbar, ob TP, SL oder ein Risk-Eingriff geschlossen hat.
            self.logger.info(
                f"[CLOSE] {symbol} {trade.side} @ {price:.4f} | PnL {trade.pnl:+.2f} | Grund: {reason}"
            )
            self.reporter.print_trade_executed(trade)
            await self.reporter.send_trade_alert(trade)

            # Phase 6: Factor Attribution Update (wenn der Trade aus dem Confluence-System kam)
            if position.market_type == 'confluence':
                breakdown = self._last_confluence_breakdowns.pop(trade.symbol, None)
                if breakdown:
                    self._update_factor_attribution(trade, breakdown)

                # Verlust-Cooldown: Symbol nach Verlust-Exit pausieren, damit das
                # (oft noch aktive) Signal nicht sofort die nächste Verlustkette startet.
                if trade.pnl < 0:
                    cooldown_s = self.config.get('strategies', {}).get('confluence', {}) \
                                            .get('loss_cooldown_seconds', 1800)
                    self._confluence_loss_block[symbol] = datetime.now() + timedelta(seconds=cooldown_s)
                    self.logger.info(f"[CONFLUENCE COOLDOWN] {symbol} nach Verlust pausiert "
                                     f"bis {self._confluence_loss_block[symbol]:%H:%M:%S}")

    async def _telegram_hourly_loop(self):
        """Sendet periodischen Telegram-Report (Intervall konfigurierbar, Default 5h)"""
        interval_s = int(self._telegram_config.get('report_interval_hours', 5) * 3600)
        # Wanduhr statt langem asyncio.sleep: auf macOS zählt asyncio nur WACHE
        # Zeit — bei Maintenance Sleep im Batteriebetrieb sammelt ein 5h-Timer
        # real über 15h nicht genug an und feuert nie. Darum in kurzen Schritten
        # schlafen und gegen datetime.now() prüfen.
        next_report = datetime.now() + timedelta(seconds=interval_s)
        while self.running:
            if datetime.now() < next_report:
                await asyncio.sleep(60)
                continue
            next_report = datetime.now() + timedelta(seconds=interval_s)
            try:
                state = self.portfolio.get_state()
                metrics = self.risk_manager.get_metrics(
                    state.equity,
                    self.portfolio.get_daily_drawdown(),
                    self.portfolio.get_sharpe_ratio()
                )
                uptime_h = (datetime.now() - self.start_time).total_seconds() / 3600
                await self.reporter.send_telegram_hourly_report(
                    self.portfolio, metrics, uptime_h
                )
            except Exception as e:
                self.logger.error(f"Telegram-Loop Fehler: {e}")

    async def _close_all_positions(self, reason: str):
        """Schließt alle Positionen"""
        # Bewusst ohne Altersfilter: das hier ist der Notfallpfad (Risk-Limit,
        # Shutdown). Bei totem Feed gar nicht zu schließen wäre schlechter als
        # zum letzten bekannten Kurs zu schließen — der Preis wird aber vermerkt,
        # damit ein dadurch verzerrter PnL später nachvollziehbar bleibt.
        prices = self.crypto_feed.get_prices()
        for symbol in list(self.portfolio.positions.keys()):
            age = self.crypto_feed.get_price_age(symbol)
            if age > self._max_price_age:
                self.logger.warning(
                    f"[NOTFALL-EXIT] {symbol} wird auf einem {age:.0f}s alten Preis "
                    f"geschlossen ({reason}) — PnL kann abweichen."
                )

        for symbol in list(self.portfolio.positions.keys()):
            price = prices.get(symbol)
            if price:
                await self._close_position(symbol, price, reason)

    async def _process_filled_order(self, order):
        """Verarbeitet gefüllte Order"""
        self.logger.info(f"Order verarbeitet: {order.id}")

    async def _initial_ml_training(self):
        """Initiales ML-Training"""
        self.logger.info("Starte initiales ML-Training...")

        for symbol in self.momentum.pairs:
            candles = self.crypto_feed.get_candles(symbol, '1m', n=500)
            if candles is not None and len(candles) >= 100:
                self.ml_predictor.train(candles, symbol)

    def _get_strategy_stats(self) -> Dict:
        """Sammelt Strategie-Statistiken"""
        stats = {
            'momentum': self.momentum.get_statistics(),
            'scalper': self.scalper.get_statistics(),
            'ml': self.ml_predictor.get_statistics()
        }

        if self.daily_trend is not None:
            stats['daily_trend'] = self.daily_trend.get_statistics()

        if self.confluence_strategy is not None:
            stats['confluence'] = {
                'enabled': True,
                'last_regime': getattr(self.confluence_strategy, '_last_regime', None),
            }

        return stats


def main():
    """Haupteinstiegspunkt"""
    # PID-Lock: verhindert mehrfache Instanzen
    import os
    pid_file = Path('/tmp/trading-bot.pid')
    if pid_file.exists():
        old_pid = int(pid_file.read_text().strip())
        # os.kill(pid, 0) statt /proc-Check — /proc existiert auf macOS nicht,
        # wodurch der Lock nie griff und Doppel-Instanzen möglich waren.
        try:
            os.kill(old_pid, 0)
            print(f"Bot läuft bereits (PID {old_pid}). Beende.")
            sys.exit(0)
        except (ProcessLookupError, PermissionError):
            pass  # verwaistes PID-File — weiter
    pid_file.write_text(str(os.getpid()))

    bot = TradingBot()

    def signal_handler(sig, frame):
        print("\nBeende Bot...")
        bot.running = False
        pid_file.unlink(missing_ok=True)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Bot starten
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nBot beendet.")
    except Exception as e:
        print(f"Fataler Fehler: {e}")
        sys.exit(1)
    finally:
        pid_file.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
