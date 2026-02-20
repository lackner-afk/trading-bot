#!/usr/bin/env python3
"""
Paper-Trading-Bot Haupt-Orchestrator
Koordiniert Datenfeeds, Strategien und Execution
"""

import asyncio
import signal
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml
from dotenv import load_dotenv

# Lokale Imports
from core.portfolio import Portfolio
from core.risk_manager import RiskManager, RiskAction
from core.order_engine import OrderEngine
from data.kraken_feed import KrakenFeed
from strategies.crypto_scalper import CryptoScalper, SignalType
from strategies.momentum import MomentumStrategy
from strategies.ml_predictor import MLPredictor
from notifications.reporter import Reporter


class TradingBot:
    """
    Haupt-Bot-Klasse

    Orchestriert alle Komponenten im async Event-Loop
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
        """Konfiguriert Logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
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
        """Standard-Konfiguration"""
        return {
            'general': {
                'mode': 'paper',
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
        """Initialisiert alle Bot-Komponenten"""
        general = self.config.get('general', {})
        risk_config = self.config.get('risk', {})
        strategy_config = self.config.get('strategies', {})

        # Core
        self.portfolio = Portfolio(
            start_capital=general.get('start_capital', 10000)
        )
        self.risk_manager = RiskManager(config=risk_config)
        self.order_engine = OrderEngine(config=self.config.get('fees', {}))

        # Trading-Pairs aus Momentum oder Scalper Config
        momentum_config = strategy_config.get('momentum', {})
        scalper_config = strategy_config.get('scalper', {})
        pairs = momentum_config.get('pairs', scalper_config.get('pairs', []))

        # Kraken Daten-Feed (echte EUR-Pairs, kein API-Key nötig)
        self.crypto_feed = KrakenFeed(config={'pairs': pairs})

        # Strategien
        self.momentum = MomentumStrategy(config=momentum_config)
        self.scalper = CryptoScalper(config=scalper_config)
        self.ml_predictor = MLPredictor(config=strategy_config.get('ml', {}))

        # Reporter
        self.reporter = Reporter(config=self.config.get('notifications', {}))

        # Telegram-Config (Initialisierung in start(), da await nötig)
        self._telegram_config = self.config.get('notifications', {}).get('telegram', {})

        # Callbacks setzen
        self.order_engine.on_fill = self._on_order_fill

    async def _on_order_fill(self, result):
        """Callback wenn Order gefüllt wird"""
        self.logger.info(f"Order gefüllt: {result.order.symbol} @ {result.execution_price}")

    async def start(self):
        """Startet den Bot"""
        self.running = True
        self.start_time = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("Paper-Trading-Bot startet...")
        self.logger.info("=" * 60)

        # Startup-Banner
        self.reporter.print_startup_banner(self.config)

        # Komponenten starten
        await self.crypto_feed.start()
        await self.reporter.start()

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

    async def _momentum_loop(self):
        """Momentum-Strategie Loop (alle 30 Sekunden)"""
        if not self.config.get('strategies', {}).get('momentum', {}).get('enabled'):
            return

        self.logger.info("Momentum-Loop gestartet")

        while self.running:
            try:
                for symbol in self.momentum.pairs:
                    candles = self.crypto_feed.get_candles(symbol, '1m')
                    price = self.crypto_feed.get_price(symbol)

                    if candles is None or price is None:
                        continue

                    signal = self.momentum.analyze(symbol, candles, price)

                    if signal and signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
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

                await asyncio.sleep(3600)  # 1 Stunde

            except Exception as e:
                self.logger.error(f"Fehler im Reporting: {e}")
                await asyncio.sleep(60)

    async def _execute_signal(self, signal, strategy_name: str = 'momentum'):
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

        # Risk-Check
        risk_check = self.risk_manager.check_trade(
            portfolio_equity=state.equity,
            position_size=state.equity * 0.1,
            leverage=signal.suggested_leverage,
            current_positions=len(state.positions),
            consecutive_losses=self.portfolio.consecutive_losses,
            daily_drawdown=self.portfolio.get_daily_drawdown(),
            sl_distance_pct=sl_distance_pct if sl_distance_pct > 0 else None
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
            position_size = min(abs(state.equity) * 0.1, abs(state.balance) * 0.3)
        # Skaliert auf Kapital: Min 5% des Equity, Max 10% des Equity
        min_size = max(5.0, state.equity * 0.05)
        max_size = state.equity * 0.10
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
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            self.reporter.print_info(
                f"{strategy_name.upper()}: {signal.signal_type.value.upper()} {signal.symbol} "
                f"@ ${result.execution_price:.4f} (Conf: {signal.confidence:.0%})"
            )

    async def _check_exit_conditions(self, prices: Dict[str, float]):
        """Prüft Exit-Bedingungen für alle Positionen"""
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Richtige Strategie für Exit-Check wählen
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
        return {
            'momentum': self.momentum.get_statistics(),
            'scalper': self.scalper.get_statistics(),
            'ml': self.ml_predictor.get_statistics()
        }


def main():
    """Haupteinstiegspunkt"""
    # Signal-Handler für graceful shutdown
    bot = TradingBot()

    def signal_handler(sig, frame):
        print("\nBeende Bot...")
        bot.running = False

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


if __name__ == '__main__':
    main()
