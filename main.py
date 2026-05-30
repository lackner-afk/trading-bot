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
from data.kraken_feed import KrakenFeed
from data.onetrading_ccxt_feed import OneTradingCCXTFeed
from strategies.crypto_scalper import CryptoScalper, SignalType
from strategies.momentum import MomentumStrategy
from strategies.ml_predictor import MLPredictor
from strategies.confluence_strategy import ConfluenceStrategy  # New 2026 multi-factor system
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

            # Echte Execution Engine
            self.order_engine = LiveOrderEngine(
                api_key=api_key,
                api_secret=api_secret,
                config=fees_config
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
                self.config.get('strategies', {}).get('confluence', {})
            )
            self.logger.info("ConfluenceStrategy (neues Multi-Factor System) aktiviert")
        else:
            self.confluence_strategy = None

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
            asyncio.create_task(self._main_loop()),
            asyncio.create_task(self._momentum_loop()),
            asyncio.create_task(self._scalper_loop()),
            asyncio.create_task(self._ml_loop()),
            asyncio.create_task(self._risk_check_loop()),
            asyncio.create_task(self._reporting_loop()),
            asyncio.create_task(self._telegram_hourly_loop()),
        ]

        # Neue Confluence Strategy Loop (wenn aktiviert)
        if self.use_confluence_strategy:
            tasks.append(asyncio.create_task(self._confluence_loop()))

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

        await self.crypto_feed.stop()
        await self.reporter.stop()

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

                universe = self.confluence_strategy.get_recommended_assets(all_candles)

                for symbol in universe.get("symbols", []):
                    candles = self.crypto_feed.get_candles(symbol, '5m', n=80)
                    price = self.crypto_feed.get_price(symbol)

                    if candles is None or price is None:
                        continue

                    signal = self.confluence_strategy.analyze_legacy(symbol, candles, price)

                    if signal and signal.confidence >= 0.55:
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

        # Max 5 gleichzeitige Positionen
        if len(state.positions) >= 5:
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

        # ATR-basierte Position Size (2% Risiko pro Trade)
        if sl_distance_pct > 0 and hasattr(signal, 'atr_value') and signal.atr_value > 0:
            position_size = self.risk_manager.size_from_risk(state.equity, sl_distance_pct)
        else:
            position_size = state.equity * 0.20
        # Skaliert auf Kapital: Min 15% des Equity, Max 25% des Equity
        min_size = max(10.0, state.equity * 0.15)
        max_size = state.equity * 0.25
        position_size = max(min_size, min(max_size, position_size))

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
            self.portfolio.open_position(
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

            # Richtige Strategie für Exit-Check wählen (Phase 5)
            if position.market_type == 'momentum':
                strategy = self.momentum
            elif position.market_type == 'confluence':
                # Für Confluence-Positionen nutzen wir die Momentum-Exit-Logik als Fallback.
                # Langfristig sollte hier eine dedizierte Exit-Logik der ConfluenceStrategy kommen.
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

        fees = position.size * 0.0006  # Taker Fee (size ist bereits in USD)

        trade = self.portfolio.close_position(
            symbol=symbol,
            exit_price=price,
            fees=fees,
            strategy=position.market_type
        )

        if trade:
            self.reporter.print_trade_executed(trade)
            await self.reporter.send_trade_alert(trade)

            # Phase 6: Factor Attribution Update (wenn der Trade aus dem Confluence-System kam)
            if position.market_type == 'confluence':
                breakdown = self._last_confluence_breakdowns.pop(trade.symbol, None)
                if breakdown:
                    self._update_factor_attribution(trade, breakdown)

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
    import os
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
