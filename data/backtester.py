"""
Backtester für historische Simulation
Lädt Daten von Binance via CCXT und simuliert Strategien
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    ccxt_async = None

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator


@dataclass
class BacktestResult:
    """Ergebnis eines Backtests"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    buy_hold_return: float
    alpha: float  # Outperformance vs Buy&Hold
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)


@dataclass
class BacktestTrade:
    """Ein Trade im Backtest"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    leverage: float
    pnl: float
    pnl_pct: float
    fees: float


class Backtester:
    """
    Historische Simulation für Trading-Strategien
    Nutzt CoinGecko für Preisdaten

    Realistische Limits:
    - Max 3 gleichzeitige Positionen
    - Max 10% des Kapitals pro Position
    - Min 1000 USD Kapital für Trading
    - Cooldown nach Trade auf gleichem Symbol
    """

    def __init__(self, initial_capital: float = 10000.0, config: Dict = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        self.logger = logging.getLogger('Backtester')

        # Fee-Struktur
        self.maker_fee = self.config.get('maker_fee', 0.0004)
        self.taker_fee = self.config.get('taker_fee', 0.0006)

        # Risk-Limits
        self.max_positions = self.config.get('max_positions', 3)
        self.max_position_pct = self.config.get('max_position_pct', 0.10)  # 10% max pro Position
        self.min_capital = self.config.get('min_capital', 1000)  # Min $1000 zum Traden
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.008)  # 0.8% Stop-Loss
        self.take_profit_pct = self.config.get('take_profit_pct', 0.015)  # 1.5% Take-Profit
        self.cooldown_periods = self.config.get('cooldown_periods', 5)  # 5 Perioden Pause nach Trade

        # Daten-Cache
        self.price_data: Dict[str, pd.DataFrame] = {}

    async def load_data(self, symbols: List[str], days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Lädt historische OHLCV-Daten von Binance via CCXT

        Args:
            symbols: Liste von Trading-Pairs (z.B. ['BTC/USDT', 'ETH/USDT'])
            days: Anzahl Tage historischer Daten

        Returns:
            Dict mit DataFrames pro Symbol
        """
        self.logger.info(f"Lade historische Daten für {len(symbols)} Coins über {days} Tage via Binance")

        if ccxt_async is None:
            self.logger.error("ccxt nicht installiert - pip install ccxt")
            self.price_data = self._generate_simulated_data(symbols, days)
            return self.price_data

        exchange = ccxt_async.binance({'enableRateLimit': True})

        try:
            for symbol in symbols:
                try:
                    df = await self._fetch_binance_ohlcv(exchange, symbol, days)
                    if df is not None and len(df) > 0:
                        df = self._calculate_indicators(df)
                        self.price_data[symbol] = df
                        start_price = df['close'].iloc[0]
                        end_price = df['close'].iloc[-1]
                        change = (end_price - start_price) / start_price * 100
                        self.logger.info(f"{symbol}: {len(df)} Kerzen geladen | "
                                        f"${start_price:.2f} → ${end_price:.2f} ({change:+.1f}%)")
                except Exception as e:
                    self.logger.error(f"Fehler beim Laden von {symbol}: {e}")
        finally:
            await exchange.close()

        # Fallback auf simulierte Daten wenn API nicht verfügbar
        if len(self.price_data) == 0:
            self.logger.warning("Keine API-Daten verfügbar - generiere simulierte Daten")
            self.price_data = self._generate_simulated_data(symbols, days)

        return self.price_data

    async def _fetch_binance_ohlcv(self, exchange, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Holt OHLCV-Daten von Binance via CCXT (stündliche Kerzen)"""
        # Zeitraum berechnen
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        timeframe = '1h'
        all_candles = []

        # Binance liefert max 1000 Kerzen pro Request - paginieren
        while True:
            try:
                candles = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Nächste Seite: ab letztem Timestamp + 1ms
                last_ts = candles[-1][0]
                if last_ts == since:
                    break
                since = last_ts + 1

                # Rate-Limit beachten
                await asyncio.sleep(0.1)

                # Fertig wenn weniger als 1000 zurückkommen
                if len(candles) < 1000:
                    break

            except Exception as e:
                self.logger.warning(f"Binance API-Fehler für {symbol}: {e}")
                break

        if not all_candles:
            return None

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

        return df

    def _generate_simulated_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """
        Generiert realistische simulierte historische Daten

        Verwendet:
        - Mean-Reversion für kurzfristige Bewegungen
        - Trend-Following für längerfristige Richtung
        - Volatility-Clustering (hohe Vol folgt auf hohe Vol)
        """
        data = {}
        base_prices = {
            'BTC/USDT': 95000,
            'ETH/USDT': 3200,
            'SOL/USDT': 180,
            'DOGE/USDT': 0.32,
        }

        # Volatilitäten pro Asset (realistisch)
        volatilities = {
            'BTC/USDT': 0.015,   # ~1.5% stündlich
            'ETH/USDT': 0.018,
            'SOL/USDT': 0.025,
            'DOGE/USDT': 0.03,
        }

        hours = days * 24
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]

        np.random.seed(42)  # Reproduzierbare Ergebnisse

        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            base_vol = volatilities.get(symbol, 0.02)

            closes = [base_price]
            opens = [base_price]
            highs = [base_price * 1.005]
            lows = [base_price * 0.995]
            volumes = [np.random.uniform(1000000, 5000000)]

            # State für Volatility-Clustering
            current_vol = base_vol
            trend = 0  # Akkumulierter Trend

            for i in range(1, hours):
                # Volatility-Clustering: Vol ändert sich langsam
                vol_change = np.random.normal(0, 0.1)
                current_vol = max(base_vol * 0.5, min(base_vol * 2, current_vol * (1 + vol_change)))

                # Trend mit Mean-Reversion
                trend = trend * 0.95 + np.random.normal(0, 0.001)  # Langsamer Drift

                # Preisänderung: Trend + Random
                price_change = trend + np.random.normal(0, current_vol)

                # Mean-Reversion zum Basis-Preis (sehr schwach)
                mean_reversion = (base_price - closes[-1]) / base_price * 0.001

                new_close = closes[-1] * (1 + price_change + mean_reversion)
                new_open = closes[-1]  # Open = vorheriger Close

                # Realistische High/Low basierend auf Volatilität
                intrabar_vol = abs(np.random.normal(0, current_vol * 0.5))
                if new_close > new_open:
                    new_high = max(new_close, new_open) * (1 + intrabar_vol)
                    new_low = min(new_close, new_open) * (1 - intrabar_vol * 0.5)
                else:
                    new_high = max(new_close, new_open) * (1 + intrabar_vol * 0.5)
                    new_low = min(new_close, new_open) * (1 - intrabar_vol)

                # Volume korreliert mit Volatilität
                new_volume = np.random.uniform(1000000, 5000000) * (1 + abs(price_change) * 10)

                opens.append(new_open)
                closes.append(new_close)
                highs.append(new_high)
                lows.append(new_low)
                volumes.append(new_volume)

            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })

            df = self._calculate_indicators(df)
            data[symbol] = df

            # Log Preisentwicklung
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            change = (end_price - start_price) / start_price * 100
            self.logger.info(f"{symbol}: ${start_price:.2f} → ${end_price:.2f} ({change:+.1f}%)")

        return data

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechnet technische Indikatoren für Backtest"""
        if len(df) < 20:
            return df

        # RSI (14)
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()

        # Bollinger Bands (20, 2)
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()

        # EMA (9, 21)
        ema9 = EMAIndicator(close=df['close'], window=9)
        ema21 = EMAIndicator(close=df['close'], window=21)
        df['ema_9'] = ema9.ema_indicator()
        df['ema_21'] = ema21.ema_indicator()

        # Average Volume
        df['avg_volume'] = df['volume'].rolling(window=20).mean()

        return df

    def run_backtest(self, strategy_func: Callable, symbols: List[str] = None,
                    leverage: float = 1.0, strategy_name: str = "Strategy") -> BacktestResult:
        """
        Führt Backtest mit gegebener Strategie-Funktion durch

        Args:
            strategy_func: Funktion die (df, index) -> signal ('long', 'short', 'close', None)
            symbols: Zu testende Symbole
            leverage: Hebel für Positionen
            strategy_name: Name für Ergebnis

        Returns:
            BacktestResult mit allen Metriken
        """
        if symbols is None:
            symbols = list(self.price_data.keys())

        if not symbols:
            self.logger.error("Keine Daten für Backtest verfügbar")
            return None

        # Leverage begrenzen auf max 20x für realistische Ergebnisse
        leverage = min(leverage, 20.0)

        # Initialisierung
        capital = self.initial_capital
        equity_curve = [(datetime.now() - timedelta(days=90), capital)]
        trades = []
        positions = {}  # symbol -> {'side', 'entry_price', 'size', 'entry_time', 'margin'}
        cooldowns = {}  # symbol -> periods_remaining

        # Kombiniere alle Daten nach Timestamp
        all_data = {}
        for symbol in symbols:
            if symbol in self.price_data:
                df = self.price_data[symbol].copy()
                for _, row in df.iterrows():
                    ts = row['timestamp']
                    if ts not in all_data:
                        all_data[ts] = {}
                    all_data[ts][symbol] = row

        # Sortieren nach Zeit
        sorted_times = sorted(all_data.keys())

        # Durchlaufe alle Zeitpunkte
        for i, ts in enumerate(sorted_times):
            data_point = all_data[ts]

            # Cooldowns reduzieren
            for sym in list(cooldowns.keys()):
                cooldowns[sym] -= 1
                if cooldowns[sym] <= 0:
                    del cooldowns[sym]

            for symbol, row in data_point.items():
                current_price = row['close']

                # Position verwalten
                if symbol in positions:
                    pos = positions[symbol]

                    # PnL berechnen (realistisch: auf Margin, nicht auf Notional)
                    if pos['side'] == 'long':
                        price_change_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    else:
                        price_change_pct = (pos['entry_price'] - current_price) / pos['entry_price']

                    # PnL = Margin * Leverage * Price Change
                    pnl = pos['margin'] * leverage * price_change_pct

                    # Stop-Loss / Take-Profit
                    if price_change_pct <= -self.stop_loss_pct or price_change_pct >= self.take_profit_pct:
                        # Position schließen
                        fees = pos['margin'] * leverage * current_price / pos['entry_price'] * self.taker_fee
                        net_pnl = pnl - fees

                        # Margin zurückgeben + PnL
                        capital += pos['margin'] + net_pnl

                        trade = BacktestTrade(
                            symbol=symbol,
                            side=pos['side'],
                            entry_price=pos['entry_price'],
                            exit_price=current_price,
                            entry_time=pos['entry_time'],
                            exit_time=ts,
                            size=pos['margin'] * leverage,
                            leverage=leverage,
                            pnl=net_pnl,
                            pnl_pct=price_change_pct * leverage,
                            fees=fees
                        )
                        trades.append(trade)
                        del positions[symbol]

                        # Cooldown setzen
                        cooldowns[symbol] = self.cooldown_periods

                # Strategie-Signal holen (nur wenn keine Position und kein Cooldown)
                if symbol not in positions and symbol not in cooldowns:
                    # Prüfe ob noch Kapital und Position-Limits erlauben
                    if capital < self.min_capital:
                        continue  # Zu wenig Kapital

                    if len(positions) >= self.max_positions:
                        continue  # Zu viele offene Positionen

                    df = self.price_data[symbol]
                    idx = df[df['timestamp'] == ts].index
                    if len(idx) > 0:
                        signal = strategy_func(df, idx[0])

                        if signal in ['long', 'short']:
                            # Margin berechnen (max 10% des Kapitals)
                            margin = min(capital * self.max_position_pct, capital * 0.5)

                            if margin < 100:  # Min $100 Margin
                                continue

                            # Entry-Fees
                            fees = margin * leverage * self.taker_fee
                            capital -= (margin + fees)  # Margin + Fees abziehen

                            positions[symbol] = {
                                'side': signal,
                                'entry_price': current_price,
                                'margin': margin,
                                'entry_time': ts
                            }

            # Equity-Curve aktualisieren
            current_equity = capital
            for symbol, pos in positions.items():
                if symbol in data_point:
                    price = data_point[symbol]['close']
                    if pos['side'] == 'long':
                        price_change_pct = (price - pos['entry_price']) / pos['entry_price']
                    else:
                        price_change_pct = (pos['entry_price'] - price) / pos['entry_price']

                    unrealized = pos['margin'] * leverage * price_change_pct
                    current_equity += pos['margin'] + unrealized  # Margin + unrealized PnL

            equity_curve.append((ts, current_equity))

        # Offene Positionen schließen
        for symbol, pos in list(positions.items()):
            if symbol in self.price_data:
                last_price = self.price_data[symbol]['close'].iloc[-1]
                if pos['side'] == 'long':
                    price_change_pct = (last_price - pos['entry_price']) / pos['entry_price']
                else:
                    price_change_pct = (pos['entry_price'] - last_price) / pos['entry_price']

                pnl = pos['margin'] * leverage * price_change_pct
                fees = pos['margin'] * leverage * self.taker_fee
                capital += pos['margin'] + pnl - fees

        # Metriken berechnen
        return self._calculate_metrics(strategy_name, trades, equity_curve, capital, symbols)

    def _calculate_metrics(self, strategy_name: str, trades: List[BacktestTrade],
                          equity_curve: List[tuple], final_capital: float,
                          symbols: List[str]) -> BacktestResult:
        """Berechnet alle Backtest-Metriken"""

        # Basis-Metriken
        total_return = final_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        # Win/Loss
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Profit Factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max Drawdown
        equities = [e for _, e in equity_curve]
        max_dd = 0
        peak = equities[0]
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe Ratio (annualisiert)
        if len(equity_curve) > 1:
            returns = [(equity_curve[i][1] - equity_curve[i-1][1]) / equity_curve[i-1][1]
                      for i in range(1, len(equity_curve)) if equity_curve[i-1][1] > 0]
            if returns:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Buy & Hold Vergleich
        buy_hold_return = 0
        for symbol in symbols:
            if symbol in self.price_data:
                df = self.price_data[symbol]
                start_price = df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                buy_hold_return += (end_price - start_price) / start_price

        buy_hold_return /= len(symbols) if symbols else 1

        # Alpha (Outperformance)
        alpha = total_return_pct - buy_hold_return

        # Datum-Range
        start_date = equity_curve[0][0] if equity_curve else datetime.now()
        end_date = equity_curve[-1][0] if equity_curve else datetime.now()

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            buy_hold_return=buy_hold_return,
            alpha=alpha,
            trades=[{
                'symbol': t.symbol,
                'side': t.side,
                'entry': t.entry_price,
                'exit': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct
            } for t in trades],
            equity_curve=equity_curve
        )

    def grid_search(self, strategy_class: Type, param_grid: Dict,
                   symbols: List[str] = None) -> List[BacktestResult]:
        """
        Grid-Search für Strategie-Parameter-Optimierung

        Args:
            strategy_class: Strategie-Klasse mit run() Methode
            param_grid: Dict mit Parameter-Listen z.B. {'rsi_period': [10, 14, 20]}
            symbols: Zu testende Symbole

        Returns:
            Liste von BacktestResults sortiert nach Return
        """
        from itertools import product

        results = []

        # Alle Parameter-Kombinationen
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))
            self.logger.info(f"Teste Parameter: {params}")

            try:
                strategy = strategy_class(**params)

                def strategy_func(df, idx):
                    return strategy.generate_signal(df, idx)

                result = self.run_backtest(
                    strategy_func,
                    symbols=symbols,
                    strategy_name=f"{strategy_class.__name__}_{params}"
                )

                if result:
                    results.append(result)

            except Exception as e:
                self.logger.error(f"Fehler bei {params}: {e}")

        # Sortieren nach Return
        results.sort(key=lambda r: r.total_return_pct, reverse=True)

        return results

    def print_results(self, result: BacktestResult):
        """Gibt Backtest-Ergebnisse formatiert aus"""
        print("\n" + "="*60)
        print(f"BACKTEST ERGEBNIS: {result.strategy_name}")
        print("="*60)
        print(f"Zeitraum: {result.start_date.strftime('%Y-%m-%d')} bis {result.end_date.strftime('%Y-%m-%d')}")
        print(f"Startkapital: ${result.initial_capital:,.2f}")
        print(f"Endkapital: ${result.final_capital:,.2f}")
        print("-"*60)
        print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2%})")
        print(f"Buy & Hold Return: {result.buy_hold_return:.2%}")
        print(f"Alpha: {result.alpha:.2%}")
        print("-"*60)
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print("-"*60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning: {result.winning_trades} | Losing: {result.losing_trades}")
        print(f"Avg Win: ${result.avg_win:,.2f} | Avg Loss: ${result.avg_loss:,.2f}")
        print("="*60 + "\n")
