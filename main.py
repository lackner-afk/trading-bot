#!/usr/bin/env python3
"""
Trading-Bot Haupt-Orchestrator
Koordiniert Datenfeeds, Strategien und Execution

Unterstützt zwei Modi:
- Paper (Default): Simulierte Orders + Kraken oder OneTrading Feed
- Live: Echte Orders auf One Trading via LiveOrderEngine + Reconciliation
"""

import asyncio
import signal
import logging
import os
import sys
from datetime import datetime
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
from core.market_constraints import MarketConstraints
from data.kraken_feed import KrakenFeed
from data.onetrading_ccxt_feed import OneTradingCCXTFeed
from strategies.crypto_scalper import CryptoScalper, SignalType
from strategies.momentum import MomentumStrategy
from strategies.ml_predictor import MLPredictor
from strategies.confluence_strategy import ConfluenceStrategy  # New 2026 multi-factor system
from strategies.confluence_exit import ConfluenceExitManager
from notifications.reporter import Reporter


class TradingBot:
    """
    Haupt-Bot-Klasse

    Orchestriert alle Komponenten im async Event-Loop.

    Unterstützt Paper- und Live-Modus (gesteuert über config['general']['mode']).
    Im Live-Modus werden echte Orders auf One Trading ausgeführt + Reconciliation
    beim Start durchgeführt.
    """

    # Nach so vielen fehlgeschlagenen Exit-Versuchen für dasselbe Symbol
    # stoppt der Bot. Wer Positionen nicht mehr schließen kann, darf keine
    # neuen aufmachen.
    MAX_EXIT_FAILURES = 5

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

        # Marktbeschränkungen des Ziel-Venues (Bitpanda Fusion: Spot-only).
        # Gelten auch im Paper-Modus, damit die Testphase genau das misst,
        # was live überhaupt ausführbar wäre.
        self.constraints = MarketConstraints.from_config(self.config)
        self.logger.info(f"Marktbeschränkungen: {self.constraints.describe()}")

        # Core (Portfolio + Risk immer gleich)
        self.portfolio = Portfolio(
            start_capital=general.get('start_capital', 10000),
            constraints=self.constraints
        )
        self.risk_manager = RiskManager(config=risk_config, constraints=self.constraints)

        # Trading-Pairs aus Momentum oder Scalper Config
        momentum_config = strategy_config.get('momentum', {})
        scalper_config = strategy_config.get('scalper', {})
        pairs = momentum_config.get('pairs', scalper_config.get('pairs', []))

        if self.is_live:
            # === LIVE MODE ===
            api_key = os.getenv('ONETRADING_API_KEY')
            api_secret = os.getenv('ONETRADING_API_SECRET')

            if not api_key or not api_secret:
                raise RuntimeError(
                    "LIVE MODE AKTIVIERT, aber ONETRADING_API_KEY / ONETRADING_API_SECRET fehlen in secrets.env!"
                )

            self.logger = logging.getLogger('TradingBot')  # re-fetch after possible config load
            self.logger.critical("=== LIVE MODE INITIALISIERT ===")
            self.logger.critical("Verwende OneTradingCCXTFeed + LiveOrderEngine")

            # Echte Execution Engine.
            # shadow_mode lag bisher nicht in der übergebenen Config: main.py
            # reichte nur den fees:-Block durch, sodass sich Shadow Mode
            # ausschließlich durch ein shadow_mode: true INNERHALB von fees:
            # aktivieren ließ. Die in LIVE_TRADING.md empfohlene Rollout-Stufe 1
            # war damit praktisch nicht erreichbar.
            live_config = dict(fees_config)
            live_config.update(self.config.get('live', {}) or {})

            if live_config.get('shadow_mode'):
                self.logger.critical("SHADOW MODE aktiv - es werden KEINE echten Orders platziert")

            self.order_engine = LiveOrderEngine(
                api_key=api_key,
                api_secret=api_secret,
                config=live_config
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
                self.config.get('strategies', {}).get('confluence', {}),
                constraints=self.constraints
            )
            self.logger.info("ConfluenceStrategy (neues Multi-Factor System) aktiviert")
        else:
            self.confluence_strategy = None

        # Exit-Logik für Confluence-Positionen (eigenes Trailing-Tracking)
        self.confluence_exit = ConfluenceExitManager.from_config(
            self.config.get('strategies', {}).get('confluence', {})
        )
        # Fehlgeschlagene Exit-Versuche je Symbol (Kill-Switch-Zähler)
        self._exit_failures: Dict[str, int] = {}

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
            # Hürde 1: die Umgebungsvariable. War dokumentiert (LIVE_TRADING.md,
            # secrets.env.example) und wurde nur vom Checklisten-Tool gelesen —
            # main.py hat sie ignoriert, die "dritte unabhängige Hürde"
            # existierte faktisch nicht.
            if not os.getenv('LIVE_TRADING_ENABLED'):
                self.logger.critical("ABBRUCH: Umgebungsvariable LIVE_TRADING_ENABLED ist nicht gesetzt!")
                self.logger.critical("  export LIVE_TRADING_ENABLED=1")
                raise RuntimeError("Live mode blocked: LIVE_TRADING_ENABLED not set")

            # Hürde 2: das Confirmation-Flag. Bewusst VOR dem Countdown —
            # 10 Sekunden warten, um dann an einer Config-Prüfung zu scheitern,
            # ist sinnlos.
            if not live_confirmed:
                self.logger.critical("ABBRUCH: live_explicit_confirmation ist nicht true!")
                self.logger.critical("Setze in settings.yaml general.live_explicit_confirmation: true")
                raise RuntimeError("Live mode blocked: missing explicit confirmation flag")

            self.logger.critical("=" * 70)
            self.logger.critical("!!! LIVE-MODUS AKTIVIERT !!!")
            self.logger.critical("!!! ECHTES GELD WIRD VERWENDET !!!")
            self.logger.critical("=" * 70)
            self.logger.critical(f"Mode: {mode}")
            self.logger.critical(f"Shadow-Mode: {self.config.get('live', {}).get('shadow_mode', False)}")
            self.logger.critical(f"Beschraenkungen: {self.constraints.describe()}")
            self.logger.critical("Starte in 10 Sekunden... (Ctrl+C zum Abbrechen)")
            self.logger.critical("=" * 70)

            # Harte Verzögerung + mehrfache Warnung (async, damit die anderen
            # Loops nicht blockiert werden)
            for i in range(10, 0, -1):
                self.logger.critical(f"  LIVE START IN {i} SEKUNDEN...")
                await asyncio.sleep(1)

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
        tg = self._telegram_config
        if tg.get('enabled'):
            token = os.environ.get('TELEGRAM_BOT_TOKEN', tg.get('token', ''))
            chat_id = os.environ.get('TELEGRAM_CHAT_ID', tg.get('chat_id', ''))
            if token and chat_id:
                await self.reporter.setup_telegram(token, chat_id)

        # Haupt-Loops starten
        tasks = [
            asyncio.create_task(self._main_loop()),
            asyncio.create_task(self._risk_check_loop()),
            asyncio.create_task(self._reporting_loop()),
            asyncio.create_task(self._telegram_hourly_loop()),
        ]

        if self.use_confluence_strategy:
            # Nur das neue Multi-Factor Confluence System (Phase 1-6)
            tasks.append(asyncio.create_task(self._confluence_loop()))
        else:
            # Alte Strategien (wenn Confluence nicht als Haupt-System aktiviert ist)
            tasks.append(asyncio.create_task(self._momentum_loop()))
            tasks.append(asyncio.create_task(self._scalper_loop()))
            tasks.append(asyncio.create_task(self._ml_loop()))

        # Warte auf Beendigung
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Bot wird beendet...")
        finally:
            await self.stop()

    async def stop(self):
        """Stoppt den Bot graceful"""
        self.running = False
        self.logger.info("Stoppe Bot-Komponenten...")

        # Offene Orders stornieren, bevor die Verbindung fällt — sonst bleiben
        # sie beim Exchange stehen und werden ohne laufenden Bot gefüllt.
        try:
            cancelled = await self.order_engine.cancel_all_orders()
            if cancelled:
                self.logger.info(f"{cancelled} offene Order(s) storniert")
        except Exception as e:
            self.logger.error(f"Konnte offene Orders nicht stornieren: {e}")

        await self.crypto_feed.stop()
        await self.reporter.stop()

        # Live-Engine hält eine eigene aiohttp-Session (via CCXT)
        close = getattr(self.order_engine, 'close', None)
        if close is not None:
            try:
                await close()
            except Exception as e:
                self.logger.error(f"Fehler beim Schliessen der OrderEngine: {e}")

        # Final Report
        self.reporter.print_daily_report(self.portfolio, self._get_strategy_stats())

        self.logger.info("Bot gestoppt.")

    async def _main_loop(self):
        """Haupt-Event-Loop"""
        self.logger.info("Haupt-Loop gestartet")

        while self.running:
            try:
                # Preis-Updates verarbeiten
                prices = self.crypto_feed.get_prices()
                self.portfolio.update_position_prices(prices)

                # Hoch/Tief seit Entry fortschreiben — ohne diesen Schritt
                # hat der Trailing-Stop keine Datengrundlage.
                for sym, px in prices.items():
                    self.confluence_exit.update_price(sym, px)

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
                    price = self.crypto_feed.get_price(symbol)

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
                    price = self.crypto_feed.get_price(symbol)

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

    async def _confluence_loop(self):
        """Neue Multi-Factor Confluence Strategie Loop (Phase 1+ der Überarbeitung)"""
        if not self.use_confluence_strategy or self.confluence_strategy is None:
            return

        self.logger.info("ConfluenceStrategy-Loop gestartet (neues Multi-Factor System)")

        interval = self.config.get('strategies', {}).get('confluence', {}).get('interval_seconds', 45)

        while self.running:
            try:
                # Phase 6: Macro Event Alerting (prüft auf CPI/FOMC/etc. Fenster)
                self._check_macro_event_alert()

                # Hole aktuelle empfohlene Assets vom UniverseManager
                all_candles = {}
                for symbol in self.momentum.pairs:  # vorerst noch die alten Pairs als Basis
                    candles = self.crypto_feed.get_candles(symbol, '5m', n=80)
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

                # Kein Starvation-Fallback mehr: der frühere Fallback hat bei
                # leerer Auswahl einfach die komplette Rohliste analysiert und
                # damit den AssetSelector wirkungslos gemacht. Wenn der Selector
                # nichts durchlässt, ist das Marktumfeld das Signal.
                symbols_to_analyze = selected_symbols

                analyzed = 0
                best_score = 0.0
                best_symbol = None
                regime_name = None

                for symbol in symbols_to_analyze:
                    candles = self.crypto_feed.get_candles(symbol, '5m', n=80)
                    price = self.crypto_feed.get_price(symbol)

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

                    # Spot-Modus: ein SHORT-Signal ist kein Entry, sondern die
                    # Aufforderung eine offene Long-Position zu schließen.
                    if signal is not None and getattr(signal, 'is_exit_signal', False):
                        await self._handle_exit_signal(signal, price)
                        continue

                    # Kein zweites Confidence-Gate mehr. Vorher stand hier eine
                    # zusätzliche Schwelle von 0.55 gegen eine confidence, die
                    # durch den /9.5-Divisor nie über 0.105 kam — zwei Schwellen
                    # für dieselbe Größe haben den Bot doppelt blockiert.
                    # Die einzige Schwelle ist jetzt min_confluence_score im
                    # Aggregator; kommt ein Signal hier an, ist es akzeptiert.
                    if signal:
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
                        # (only if confluence is the source and telegram is configured)
                        if self.reporter.telegram:
                            breakdown = cd.get('factor_breakdown', {}) if cd else {}
                            top_factor_names = [
                                name.replace("_", " ").title()
                                for name, _ in sorted(breakdown.items(), key=lambda x: getattr(x[1], 'score', 0), reverse=True)[:3]
                            ]
                            asyncio.create_task(
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

        # Positionslimit kommt ausschließlich vom RiskManager. Das frühere
        # hartcodierte >= 5 hier war die dritte von drei widersprüchlichen
        # Obergrenzen (Klassenkonstante 5, Config 2, hier 5).

        if state.equity < 20:
            return

        # Spot-Venue: SHORT-Signale sind keine Entries. Sie werden im
        # Confluence-Loop als Exit geroutet und dürfen hier nicht ankommen.
        if self.constraints.spot_only and signal.signal_type != SignalType.LONG:
            self.logger.warning(
                f"SHORT-Entry auf Spot-Venue verworfen: {signal.symbol}"
            )
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

        # ATR-basierte Position Size (2% Risiko pro Trade)
        if sl_distance_pct > 0 and hasattr(signal, 'atr_value') and signal.atr_value > 0:
            position_size = self.risk_manager.size_from_risk(state.equity, sl_distance_pct)
        else:
            position_size = state.equity * self.risk_manager.max_position_size

        # REDUCE_SIZE wurde bisher ignoriert — damit waren die Macro-Reduktion,
        # die Drawdown-Reduktion und das Beta-Limit wirkungslos.
        if risk_check.action == RiskAction.REDUCE_SIZE and risk_check.suggested_size:
            position_size = min(position_size, risk_check.suggested_size)
            self.logger.info(f"Positionsgröße reduziert: {risk_check.reason}")

        # Obergrenze ist das konfigurierte max_position_size. Der frühere
        # Clamp auf 15–25% des Equity hat die 20%-Grenze nach oben überschritten
        # und die Reduktionen wieder aufgehoben.
        position_size = min(position_size, state.equity * self.risk_manager.max_position_size)

        # Mindest-Ordervolumen des Venues
        if position_size < self.constraints.min_order_notional_eur:
            self.logger.info(
                f"Order zu klein: {position_size:.2f}€ < "
                f"{self.constraints.min_order_notional_eur:.2f}€ Mindestvolumen"
            )
            return

        if position_size > state.balance or state.balance < 20:
            return

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
            position = self.portfolio.open_position(
                symbol=signal.symbol,
                side='long' if signal.signal_type == SignalType.LONG else 'short',
                size=position_size,
                price=result.execution_price,
                leverage=signal.suggested_leverage,
                strategy=strategy_name,
                market_type=strategy_name,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            if position is None:
                # Order ausgeführt, aber lokal nicht buchbar — das darf nicht
                # unbemerkt bleiben, sonst existiert real eine Position, die
                # der Bot nicht kennt und folglich nie schließt.
                self.logger.critical(
                    f"Order fuer {signal.symbol} ausgefuehrt, aber Position konnte "
                    f"lokal nicht gebucht werden! Bestand manuell pruefen."
                )
                return

            # Trailing-Tracking starten
            self.confluence_exit.register(
                signal.symbol, result.execution_price, position.timestamp
            )

            self.reporter.print_info(
                f"{strategy_name.upper()}: {signal.signal_type.value.upper()} {signal.symbol} "
                f"@ ${result.execution_price:.4f} (Conf: {signal.confidence:.0%})"
            )

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

    async def _handle_exit_signal(self, signal, price: float):
        """
        Verarbeitet ein SHORT-Signal im Spot-Modus.

        Auf einem Spot-Venue lässt sich nicht short gehen — die sinnvolle
        Entsprechung ist "verkauf, was du hast": existiert eine offene
        Long-Position auf dem Symbol, wird sie geschlossen. Sonst passiert
        nichts. Bewusst getrennt vom Entry-Pfad (_execute_signal), damit
        Ein- und Ausstieg nicht dieselbe Risikologik durchlaufen.
        """
        symbol = signal.symbol
        position = self.portfolio.positions.get(symbol)

        if position is None:
            self.logger.info(
                f"[CONFLUENCE EXIT] {symbol}: Short-Signal ohne offene Position - ignoriert "
                f"(Conf {signal.confidence:.0%})"
            )
            return

        if position.side != 'long':
            return

        self.logger.info(
            f"[CONFLUENCE EXIT] {symbol} @ {price:.2f}: Short-Signal schließt Long-Position "
            f"(Conf {signal.confidence:.0%})"
        )
        await self._close_position(symbol, price, "Confluence-Flip short")

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

            # Telegram (sofort, nicht rate-limited — das ist wichtig)
            if self.reporter.telegram:
                asyncio.create_task(self.reporter.telegram.send_message(msg))

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
            if self.reporter.telegram:
                asyncio.create_task(self.reporter.telegram.send_message(msg))

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
            if self.reporter.telegram:
                asyncio.create_task(self.reporter.telegram.send_message(msg))

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

            if position.market_type == 'confluence':
                # Eigene Exit-Logik mit funktionierendem Trailing-Stop.
                # Vorher lief das über self.momentum, dessen highest_prices
                # nirgends befüllt wird — der Trailing-Zweig war toter Code.
                should_exit, reason = self.confluence_exit.check_exit(
                    symbol=symbol,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    side=position.side,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    atr=self._get_atr(symbol),
                    entry_time=position.timestamp
                )
            else:
                strategy = self.momentum if position.market_type == 'momentum' else self.scalper
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

    def _get_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        ATR aus dem Feed statt aus dem Momentum-Cache, der bei
        Confluence-Trades nie gefüllt wird (ScalperSignal.atr_value = 0.0).
        """
        try:
            candles = self.crypto_feed.get_candles(symbol, '5m', n=period + 5)
            if candles is None or len(candles) < period:
                return None
            tr = (candles['high'] - candles['low']).rolling(period).mean()
            value = tr.iloc[-1]
            return float(value) if value == value and value > 0 else None
        except Exception:
            return None

    async def _close_position(self, symbol: str, price: float, reason: str):
        """
        Schließt eine Position — über die OrderEngine, nicht nur im Portfolio.

        Vorher buchte diese Methode ausschließlich portfolio.close_position()
        und schickte nie eine Order los. Im Live-Modus hätte der Bot damit real
        gekauft, aber Stop-Loss und Take-Profit nur lokal in die SQLite
        geschrieben — die echte Position wäre unbegrenzt offen geblieben.
        """
        position = self.portfolio.positions.get(symbol)
        if not position:
            return

        close_side = 'sell' if position.side == 'long' else 'buy'

        try:
            result = await self.order_engine.execute_market_order(
                symbol=symbol,
                side=close_side,
                size=position.size,
                current_price=price,
                leverage=position.leverage,
                strategy=position.market_type
            )
        except Exception as e:
            self.logger.critical(
                f"EXIT FEHLGESCHLAGEN (Exception) {symbol}: {e} - Position bleibt offen!"
            )
            self._register_exit_failure(symbol, reason)
            return

        if not result.success:
            # Position NICHT aus dem Portfolio entfernen. Ein lokal geschlossener,
            # real aber offener Trade ist der gefährlichste denkbare Zustand.
            self.logger.critical(
                f"EXIT FEHLGESCHLAGEN {symbol}: {getattr(result, 'message', 'unbekannt')} "
                f"- Position bleibt offen, Retry im naechsten Tick"
            )
            self._register_exit_failure(symbol, reason)
            return

        self._exit_failures.pop(symbol, None)

        # Exit-Preis und Gebühren kommen aus der tatsächlichen Ausführung,
        # nicht aus dem Signalpreis und nicht aus einer hartcodierten Fee.
        trade = self.portfolio.close_position(
            symbol=symbol,
            exit_price=result.execution_price,
            fees=result.total_fees,
            strategy=position.market_type
        )

        self.confluence_exit.forget(symbol)

        if trade:
            self.logger.info(
                f"[EXIT] {symbol} @ {result.execution_price:.4f} | {reason} | "
                f"PNL {trade.pnl:+.2f}€ | Fees {trade.fees:.4f}€"
            )
            self.reporter.print_trade_executed(trade)
            await self.reporter.send_trade_alert(trade)

            # Phase 6: Factor Attribution Update (wenn der Trade aus dem Confluence-System kam)
            if position.market_type == 'confluence':
                breakdown = self._last_confluence_breakdowns.pop(trade.symbol, None)
                if breakdown:
                    self._update_factor_attribution(trade, breakdown)

    def _register_exit_failure(self, symbol: str, reason: str):
        """
        Zählt fehlgeschlagene Exit-Versuche. Nach MAX_EXIT_FAILURES wird der
        Bot gestoppt — wenn Positionen nicht mehr geschlossen werden können,
        ist Weiterhandeln die schlechteste aller Optionen.
        """
        count = self._exit_failures.get(symbol, 0) + 1
        self._exit_failures[symbol] = count

        if count >= self.MAX_EXIT_FAILURES:
            self.logger.critical(
                f"KILL-SWITCH: {count} fehlgeschlagene Exit-Versuche fuer {symbol} "
                f"({reason}). Bot wird gestoppt - Position manuell pruefen!"
            )
            asyncio.create_task(self.reporter.send_message(
                f"🚨 KILL-SWITCH: Exit fuer {symbol} scheitert seit {count} Versuchen. "
                f"Bot gestoppt. Position bitte manuell pruefen!"
            ))
            self.running = False

    async def _telegram_hourly_loop(self):
        """Sendet stündlichen Telegram-Report"""
        await asyncio.sleep(3600)   # erste Sendung nach 1h
        while self.running:
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
            await asyncio.sleep(3600)

    async def _close_all_positions(self, reason: str):
        """Schließt alle Positionen"""
        prices = self.crypto_feed.get_prices()

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

        if self.confluence_strategy is not None:
            stats['confluence'] = {
                'enabled': True,
                'last_regime': getattr(self.confluence_strategy, '_last_regime', None),
            }

        return stats


def main():
    """Haupteinstiegspunkt"""
    # PID-Lock: verhindert mehrfache Instanzen
    pid_file = Path('/tmp/trading-bot.pid')
    if pid_file.exists():
        old_pid = int(pid_file.read_text().strip())
        if Path(f'/proc/{old_pid}').exists():
            print(f"Bot läuft bereits (PID {old_pid}). Beende.")
            sys.exit(0)
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
