"""
PNL-Reporting und Notifications
Console-Output mit Rich + optionaler Webhook-Support
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box


class Reporter:
    """
    Reporting und Notifications für Paper-Trading-Bot

    Features:
    - Rich Console-Output mit Live-Updates
    - Stündliche/Tägliche PNL-Summaries
    - Optional: Webhook-Notifications (Discord/Slack)
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger('Reporter')
        self.console = Console()

        # Webhook-Konfiguration
        self.webhook_url = self.config.get('webhook_url')
        self.webhook_enabled = self.webhook_url is not None

        # State
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_hourly_report = datetime.now()
        self.last_daily_report = datetime.now().date()

    async def start(self):
        """Initialisiert Reporter"""
        if self.webhook_enabled:
            self.session = aiohttp.ClientSession()

    async def stop(self):
        """Stoppt Reporter"""
        if self.session:
            await self.session.close()

    def print_startup_banner(self, config: Dict):
        """Zeigt Startup-Banner"""
        banner = Panel(
            Text.from_markup(
                "[bold cyan]Paper-Trading-Bot[/bold cyan]\n\n"
                f"[green]Mode:[/green] {config.get('general', {}).get('mode', 'paper')}\n"
                f"[green]Startkapital:[/green] ${config.get('general', {}).get('start_capital', 10000):,.2f}\n"
                f"[green]Datenfeed:[/green] One Trading (Live WebSocket)\n\n"
                "[yellow]ACHTUNG: Dies ist eine SIMULATION - kein echtes Geld![/yellow]"
            ),
            title="[bold white]🤖 Bot Started[/bold white]",
            border_style="cyan"
        )
        self.console.print(banner)

    def print_portfolio_summary(self, portfolio_state):
        """Zeigt Portfolio-Zusammenfassung"""
        state = portfolio_state

        # Farbe basierend auf PNL
        pnl_color = "green" if state.daily_pnl >= 0 else "red"
        equity_color = "green" if state.equity >= state.balance else "red"

        table = Table(title="Portfolio Status", box=box.ROUNDED)
        table.add_column("Metrik", style="cyan")
        table.add_column("Wert", justify="right")

        table.add_row("Balance", f"${state.balance:,.2f}")
        table.add_row("Equity", f"[{equity_color}]${state.equity:,.2f}[/{equity_color}]")
        table.add_row("Unrealized PNL", f"[{equity_color}]${state.unrealized_pnl:,.2f}[/{equity_color}]")
        table.add_row("Realized PNL", f"[{pnl_color}]${state.realized_pnl:,.2f}[/{pnl_color}]")
        table.add_row("Daily PNL", f"[{pnl_color}]${state.daily_pnl:,.2f}[/{pnl_color}]")
        table.add_row("", "")
        table.add_row("Offene Positionen", str(len(state.positions)))
        table.add_row("Total Trades", str(state.total_trades))
        table.add_row("Win Rate", f"{state.win_count / state.total_trades * 100:.1f}%" if state.total_trades > 0 else "N/A")

        self.console.print(table)

    def print_positions(self, positions: Dict):
        """Zeigt offene Positionen"""
        if not positions:
            self.console.print("[dim]Keine offenen Positionen[/dim]")
            return

        table = Table(title="Offene Positionen", box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Seite", justify="center")
        table.add_column("Größe", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Leverage", justify="center")
        table.add_column("PNL", justify="right")

        for symbol, pos in positions.items():
            side_color = "green" if pos.side == 'long' else "red"
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"

            table.add_row(
                symbol,
                f"[{side_color}]{pos.side.upper()}[/{side_color}]",
                f"${pos.size:,.2f}",
                f"${pos.entry_price:,.4f}",
                f"{pos.leverage}x",
                f"[{pnl_color}]${pos.unrealized_pnl:,.2f}[/{pnl_color}]"
            )

        self.console.print(table)

    def print_recent_trades(self, trades: List, n: int = 5):
        """Zeigt letzte Trades"""
        if not trades:
            self.console.print("[dim]Noch keine Trades[/dim]")
            return

        table = Table(title=f"Letzte {n} Trades", box=box.ROUNDED)
        table.add_column("Zeit", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Seite", justify="center")
        table.add_column("PNL", justify="right")
        table.add_column("Strategie")

        for trade in trades[-n:]:
            side_color = "green" if trade.side == 'long' else "red"
            pnl_color = "green" if trade.pnl >= 0 else "red"

            table.add_row(
                trade.exit_time.strftime("%H:%M:%S"),
                trade.symbol,
                f"[{side_color}]{trade.side.upper()}[/{side_color}]",
                f"[{pnl_color}]${trade.pnl:,.2f}[/{pnl_color}]",
                trade.strategy
            )

        self.console.print(table)

    def print_hourly_report(self, portfolio, risk_metrics: Dict):
        """Zeigt stündlichen Report"""
        state = portfolio.get_state()
        sharpe = portfolio.get_sharpe_ratio()
        max_dd = portfolio.get_max_drawdown()
        avg_win, avg_loss = portfolio.get_avg_win_loss()

        self.console.print("\n")
        self.console.rule("[bold cyan]Stündlicher Report[/bold cyan]")

        # Haupt-Metriken
        pnl_color = "green" if state.daily_pnl >= 0 else "red"
        self.console.print(f"[bold]Balance:[/bold] ${state.balance:,.2f}")
        self.console.print(f"[bold]Daily PNL:[/bold] [{pnl_color}]${state.daily_pnl:,.2f} ({state.daily_pnl/portfolio.daily_start_balance*100:.2f}%)[/{pnl_color}]")
        self.console.print(f"[bold]Sharpe Ratio:[/bold] {sharpe:.2f}")
        self.console.print(f"[bold]Max Drawdown:[/bold] {max_dd:.2%}")

        # Beste/Schlechteste Trades
        recent = portfolio.get_recent_trades(20)
        if recent:
            best = max(recent, key=lambda t: t.pnl)
            worst = min(recent, key=lambda t: t.pnl)
            self.console.print(f"[bold]Best Trade:[/bold] [green]${best.pnl:,.2f}[/green] ({best.symbol})")
            self.console.print(f"[bold]Worst Trade:[/bold] [red]${worst.pnl:,.2f}[/red] ({worst.symbol})")

        # Risk-Status
        status = risk_metrics.get('status', 'OK')
        status_color = {'OK': 'green', 'CAUTION': 'yellow', 'WARNING': 'orange3', 'CRITICAL': 'red'}
        self.console.print(f"[bold]Risk Status:[/bold] [{status_color.get(status, 'white')}]{status}[/{status_color.get(status, 'white')}]")

        self.console.print("\n")
        self.last_hourly_report = datetime.now()

    def print_daily_report(self, portfolio, strategies: Dict):
        """Zeigt ausführlichen täglichen Report"""
        state = portfolio.get_state()
        sharpe = portfolio.get_sharpe_ratio()
        max_dd = portfolio.get_max_drawdown()
        avg_win, avg_loss = portfolio.get_avg_win_loss()
        win_rate = portfolio.get_win_rate()

        self.console.print("\n")
        panel = Panel(
            self._build_daily_report_text(state, sharpe, max_dd, avg_win, avg_loss, win_rate, strategies),
            title="[bold white]📊 Täglicher Report[/bold white]",
            border_style="cyan"
        )
        self.console.print(panel)
        self.last_daily_report = datetime.now().date()

    def _build_daily_report_text(self, state, sharpe, max_dd, avg_win, avg_loss,
                                  win_rate, strategies) -> Text:
        """Baut Text für täglichen Report"""
        text = Text()

        # Performance
        pnl_pct = state.realized_pnl / 10000 * 100  # Annahme: 10k Start
        text.append("PERFORMANCE\n", style="bold underline cyan")
        text.append(f"Realized PNL: ${state.realized_pnl:,.2f} ({pnl_pct:+.2f}%)\n")
        text.append(f"Win Rate: {win_rate:.1%}\n")
        text.append(f"Sharpe Ratio: {sharpe:.2f}\n")
        text.append(f"Max Drawdown: {max_dd:.2%}\n")
        text.append(f"Avg Win: ${avg_win:,.2f} | Avg Loss: ${avg_loss:,.2f}\n")
        text.append(f"Profit Factor: {abs(avg_win/avg_loss):.2f}\n" if avg_loss != 0 else "")
        text.append("\n")

        # Strategie-Breakdown
        text.append("STRATEGIE-BREAKDOWN\n", style="bold underline cyan")
        for name, stats in strategies.items():
            if isinstance(stats, dict):
                trades = stats.get('total_trades', stats.get('trades_executed', 0))
                profit = stats.get('total_profit', 0)
                text.append(f"  {name}: {trades} Trades, ${profit:,.2f} Profit\n")

        text.append("\n")

        # Empfehlungen
        text.append("EMPFEHLUNGEN\n", style="bold underline cyan")
        if win_rate < 0.4:
            text.append("  ⚠️ Win-Rate unter 40% - Strategie-Parameter prüfen\n", style="yellow")
        if max_dd > 0.05:
            text.append("  ⚠️ Drawdown über 5% - Risk-Management anpassen\n", style="yellow")
        if sharpe < 1.0:
            text.append("  💡 Sharpe < 1.0 - Risiko/Return-Verhältnis verbessern\n", style="dim")
        if win_rate >= 0.5 and max_dd <= 0.05 and sharpe >= 1.0:
            text.append("  ✅ Performance im Zielbereich\n", style="green")

        return text

    def print_strategy_signals(self, signals: List, strategy_name: str):
        """Zeigt aktuelle Strategie-Signale"""
        if not signals:
            return

        self.console.print(f"\n[bold cyan]Signale: {strategy_name}[/bold cyan]")
        for signal in signals[-3:]:
            if hasattr(signal, 'signal_type'):
                side = signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type)
            elif hasattr(signal, 'direction'):
                side = signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction)
            else:
                side = 'unknown'

            color = "green" if side in ['long', 'bullish', 'LONG', 'BULLISH'] else "red"
            self.console.print(f"  [{color}]{side.upper()}[/{color}] {signal.symbol if hasattr(signal, 'symbol') else ''} "
                             f"@ ${signal.price if hasattr(signal, 'price') else signal.entry_price:.4f} "
                             f"(Confidence: {signal.confidence:.0%})")

    def print_live_dashboard(self, portfolio, positions: Dict, prices: Dict):
        """Erstellt Live-Dashboard (für Live-Modus)"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        # Header
        layout["header"].update(Panel(
            f"[bold cyan]Paper-Trading-Bot[/bold cyan] | "
            f"Balance: ${portfolio.balance:,.2f} | "
            f"PNL: ${portfolio.realized_pnl:+,.2f}",
            style="on dark_blue"
        ))

        # Main Content
        layout["main"].split_row(
            Layout(name="positions"),
            Layout(name="prices")
        )

        return layout

    async def send_webhook(self, message: str, title: str = "Paper-Trading-Bot"):
        """Sendet Webhook-Notification"""
        if not self.webhook_enabled or not self.session:
            return

        try:
            # Discord-Format
            payload = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": 5814783,  # Cyan
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }

            async with self.session.post(self.webhook_url, json=payload) as resp:
                if resp.status not in [200, 204]:
                    self.logger.warning(f"Webhook-Fehler: {resp.status}")

        except Exception as e:
            self.logger.error(f"Webhook-Fehler: {e}")

    async def send_trade_notification(self, trade):
        """Sendet Trade-Notification via Webhook"""
        pnl_emoji = "🟢" if trade.pnl >= 0 else "🔴"
        message = (
            f"{pnl_emoji} **{trade.side.upper()} {trade.symbol}**\n"
            f"Entry: ${trade.entry_price:,.4f} → Exit: ${trade.exit_price:,.4f}\n"
            f"PNL: ${trade.pnl:,.2f} ({trade.pnl/trade.size*100:.2f}%)\n"
            f"Strategie: {trade.strategy}"
        )
        await self.send_webhook(message, f"Trade: {trade.symbol}")

    async def send_alert(self, alert_type: str, message: str):
        """Sendet Alert via Webhook"""
        emoji = {"warning": "⚠️", "error": "❌", "info": "ℹ️", "success": "✅"}
        await self.send_webhook(f"{emoji.get(alert_type, '📢')} {message}", f"Alert: {alert_type.upper()}")

    def should_send_hourly_report(self) -> bool:
        """Prüft ob stündlicher Report fällig ist"""
        return (datetime.now() - self.last_hourly_report).total_seconds() >= 3600

    def should_send_daily_report(self) -> bool:
        """Prüft ob täglicher Report fällig ist"""
        return datetime.now().date() != self.last_daily_report

    def print_error(self, error: str):
        """Zeigt Fehler-Meldung"""
        self.console.print(f"[bold red]ERROR:[/bold red] {error}")

    def print_warning(self, warning: str):
        """Zeigt Warnung"""
        self.console.print(f"[bold yellow]WARNING:[/bold yellow] {warning}")

    def print_info(self, info: str):
        """Zeigt Info-Meldung"""
        self.console.print(f"[bold cyan]INFO:[/bold cyan] {info}")

    def print_trade_executed(self, trade):
        """Zeigt Trade-Execution"""
        pnl_color = "green" if trade.pnl >= 0 else "red"
        side_color = "green" if trade.side == 'long' else "red"

        self.console.print(
            f"[bold]TRADE:[/bold] "
            f"[{side_color}]{trade.side.upper()}[/{side_color}] {trade.symbol} | "
            f"Entry: ${trade.entry_price:,.4f} → Exit: ${trade.exit_price:,.4f} | "
            f"PNL: [{pnl_color}]${trade.pnl:,.2f}[/{pnl_color}] | "
            f"Strategy: {trade.strategy}"
        )
