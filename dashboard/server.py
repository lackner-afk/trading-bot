"""
"Panda Pro"-Dashboard — kleiner Webserver im Bot-Prozess.

Läuft als weiterer async Loop in main.py und liest den Live-Zustand direkt aus
dem Bot (Portfolio, Risk-Manager, Feed). Der Verlauf kommt aus trades.db.

Endpoints:
    GET  /               Dashboard (statisches HTML)
    GET  /api/state      Kompletter Zustand als JSON (?range=1T|1W|1M|1J|Max)
    POST /api/pause      {"paused": true|false} — neue Einstiege an/aus

Sicherheit: Standardmäßig nur auf 127.0.0.1. Wer den Server im Netz öffnet
(host: 0.0.0.0), MUSS DASHBOARD_TOKEN setzen, sonst startet er nicht.
"""

import asyncio
import hmac
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from aiohttp import web

from dashboard.store import (DashboardStore, coin_name, coin_of, strategy_name,
                             window_start)

if TYPE_CHECKING:
    from main import TradingBot

STATIC_DIR = Path(__file__).parent / 'static'
LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '::1'}
MAX_CHART_POINTS = 240


# ---------------------------------------------------------------------------
# Reine Rechenfunktionen (ohne Bot-Abhängigkeit, damit testbar)
# ---------------------------------------------------------------------------

def risk_score(daily_drawdown: float, max_daily_drawdown: float,
               open_positions: int, max_positions: int,
               locked_margin: float, equity: float,
               consecutive_losses: int) -> int:
    """
    Risiko-Wert 0–100 für die Tacho-Anzeige.

    Gewichtung:
        50 %  Ausnutzung des Tages-Drawdown-Limits
        20 %  belegte Positions-Slots
        20 %  gebundene Margin im Verhältnis zur Equity
        10 %  Verlustserie (3 Verluste = Cooldown-Schwelle)
    """
    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    dd = clamp(daily_drawdown / max_daily_drawdown) if max_daily_drawdown > 0 else 0.0
    slots = clamp(open_positions / max_positions) if max_positions > 0 else 0.0
    margin = clamp(locked_margin / equity) if equity > 0 else 1.0
    streak = clamp(consecutive_losses / 3)
    return round(100 * (0.5 * dd + 0.2 * slots + 0.2 * margin + 0.1 * streak))


def risk_label(score: int) -> str:
    if score < 25:
        return 'Niedrig'
    if score < 50:
        return 'Niedrig bis moderat'
    if score < 75:
        return 'Erhöht'
    return 'Hoch'


def max_drawdown(values: List[float]) -> float:
    """Größter Rückgang vom Hoch zum Tief, als Anteil (0.026 = 2,6 %)."""
    peak, worst = None, 0.0
    for v in values:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def downsample(points: List[Tuple[datetime, float]], limit: int = MAX_CHART_POINTS
               ) -> List[Tuple[datetime, float]]:
    """Dünnt die Kurve gleichmäßig aus, erster und letzter Punkt bleiben."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class DashboardServer:
    """Webserver + Snapshot-Loop für das Dashboard."""

    def __init__(self, bot: 'TradingBot', store: DashboardStore, config: Dict):
        self.bot = bot
        self.store = store
        self.config = config
        self.logger = logging.getLogger('Dashboard')

        self.host: str = config.get('host', '127.0.0.1')
        self.port: int = int(config.get('port', 8080))
        self.bot_name: str = config.get('bot_name', 'Panda-9')
        self.initials: str = config.get('initials', '')
        self.snapshot_seconds: int = int(config.get('snapshot_minutes', 5)) * 60
        self.token: str = os.getenv('DASHBOARD_TOKEN', '')

    # ----------------------------------------------------------------- Ablauf

    async def run(self):
        """Startet den Server und schreibt regelmäßig Equity-Snapshots."""
        if self.host not in LOOPBACK_HOSTS and not self.token:
            self.logger.error(
                f"Dashboard soll auf {self.host} lauschen, aber DASHBOARD_TOKEN fehlt — "
                "aus Sicherheitsgründen NICHT gestartet. Token in config/secrets.env setzen "
                "oder host: 127.0.0.1 verwenden."
            )
            return

        app = web.Application(middlewares=[self._auth_middleware])
        app.router.add_get('/', self._handle_index)
        app.router.add_get('/api/state', self._handle_state)
        app.router.add_post('/api/pause', self._handle_pause)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        try:
            await site.start()
        except OSError as e:
            self.logger.error(f"Dashboard konnte Port {self.port} nicht öffnen: {e} — läuft ohne Dashboard weiter.")
            await runner.cleanup()
            return

        self.logger.info(f"Dashboard läuft auf http://{self.host}:{self.port}")
        try:
            while self.bot.running:
                await self._snapshot()
                await asyncio.sleep(self.snapshot_seconds)
        finally:
            await runner.cleanup()
            self.logger.info("Dashboard gestoppt.")

    async def _snapshot(self):
        p = self.bot.portfolio
        await asyncio.to_thread(self.store.add_snapshot, p.equity, p.balance)

    # ------------------------------------------------------------- Sicherheit

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        if request.path.startswith('/api/'):
            if self.token:
                given = request.headers.get('X-Dashboard-Token') or request.query.get('token', '')
                if not hmac.compare_digest(given, self.token):
                    return web.json_response({'error': 'Token fehlt oder falsch'}, status=401)
            if request.method == 'POST':
                # Fremde Webseiten dürfen den Bot nicht per Browser fernsteuern.
                origin = request.headers.get('Origin')
                if origin and urlparse(origin).netloc != request.host:
                    return web.json_response({'error': 'Fremder Ursprung'}, status=403)
                if request.content_type != 'application/json':
                    return web.json_response({'error': 'JSON erwartet'}, status=415)
        return await handler(request)

    # --------------------------------------------------------------- Handler

    async def _handle_index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC_DIR / 'index.html')

    async def _handle_state(self, request: web.Request) -> web.Response:
        range_key = request.query.get('range', '1W')
        state = await self.build_state(range_key)
        return web.json_response(state)

    async def _handle_pause(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except ValueError:
            return web.json_response({'error': 'Ungültiges JSON'}, status=400)
        paused = body.get('paused')
        if not isinstance(paused, bool):
            return web.json_response({'error': "'paused' muss true oder false sein"}, status=400)
        await self.bot.set_trading_paused(paused, source='Dashboard')
        return web.json_response({'paused': self.bot.trading_paused})

    # ----------------------------------------------------------------- Zustand

    async def build_state(self, range_key: str) -> Dict:
        bot = self.bot
        p = bot.portfolio
        rm = bot.risk_manager
        now = datetime.now()

        positions = list(p.positions.values())
        locked_margin = sum(pos.size / pos.leverage for pos in positions)
        daily_dd = max(0.0, p.get_daily_drawdown())

        # Verlauf & Kennzahlen aus der DB
        since = window_start(range_key, now)
        series = await asyncio.to_thread(self.store.get_equity_series, since)
        if since is not None and series and series[0][0] < since:
            # Stand vor dem Fenster an den Fensteranfang setzen, damit die
            # Kurve links bündig beginnt.
            series[0] = (since, series[0][1])
        series.append((now, p.equity))
        series_30d = await asyncio.to_thread(self.store.get_equity_series, now - timedelta(days=30))
        first_ts = await asyncio.to_thread(self.store.get_first_timestamp)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = await asyncio.to_thread(self.store.count_trades_since, today)
        strat_stats = await asyncio.to_thread(self.store.get_trade_stats, now - timedelta(days=30))
        events = await asyncio.to_thread(self.store.get_events, 50)

        start_value = series[0][1] if series else p.equity
        change = p.equity - start_value
        change_pct = change / start_value if start_value > 0 else 0.0

        score = risk_score(daily_dd, rm.max_daily_drawdown, len(positions),
                           rm.max_concurrent_positions, locked_margin, p.equity,
                           p.consecutive_losses)
        without_sl = sum(1 for pos in positions if not pos.stop_loss)

        return {
            'generated_at': now.isoformat(timespec='seconds'),
            'bot': {
                'name': self.bot_name,
                'initials': self.initials,
                'mode': 'live' if bot.is_live else 'paper',
                'shadow': bool(bot.config.get('general', {}).get('shadow_mode', True)) if bot.is_live else False,
                'paused': bot.trading_paused,
                'status': self._status_text(),
                'insight': self._insight(daily_dd, len(positions), locked_margin),
            },
            'equity': {
                'value': p.equity,
                'balance': p.balance,
                'change': change,
                'change_pct': change_pct,
                'range': range_key,
                'series': [[t.isoformat(timespec='seconds'), round(v, 2)] for t, v in downsample(series)],
            },
            'stats': {
                'trades_today': trades_today,
                'win_rate': p.get_win_rate(),
                'total_trades': p.win_count + p.loss_count,
                'max_drawdown_30d': max_drawdown([v for _, v in series_30d] + [p.equity]),
                'running_since': first_ts.isoformat(timespec='seconds') if first_ts else None,
            },
            'allocation': self._allocation(positions),
            'strategies': self._strategies(strat_stats),
            'risk': {
                'score': score,
                'label': risk_label(score),
                'daily_drawdown': daily_dd,
                'daily_drawdown_limit': rm.max_daily_drawdown,
                'open_positions': len(positions),
                'max_positions': rm.max_concurrent_positions,
                'positions_without_sl': without_sl,
            },
            'activity': [
                {
                    'ts': e.timestamp.isoformat(timespec='seconds'),
                    'kind': e.kind,
                    'coin': coin_of(e.symbol),
                    'title': e.title,
                    'subtitle': e.subtitle,
                    'amount': e.amount,
                }
                for e in events
            ],
            'settings': self._settings(),
        }

    def _status_text(self) -> str:
        bot = self.bot
        if bot.trading_paused:
            return 'Pausiert · nur Exits aktiv'
        if bot._prices_stale:
            return 'Wartet auf frische Kurse'
        cooldown = bot.risk_manager.cooldown_until
        if cooldown and datetime.now() < cooldown:
            return f'Cooldown bis {cooldown:%H:%M}'
        return 'Aktiv · handelt autonom'

    def _insight(self, daily_dd: float, n_positions: int, locked_margin: float) -> str:
        """Ein Satz, der erklärt, was gerade los ist — wichtigstes zuerst."""
        bot = self.bot
        rm = bot.risk_manager
        if bot.trading_paused:
            return (f"{self.bot_name} ist pausiert und eröffnet keine neuen Positionen. "
                    "Offene Positionen werden weiter per Stop-Loss und Take-Profit überwacht.")
        if bot._prices_stale:
            return "Der Kurs-Feed liefert gerade keine frischen Preise. Bis er wieder da ist, ruht der Handel."
        if daily_dd >= rm.max_daily_drawdown * 0.5:
            return (f"Heute liegt der Drawdown bei {daily_dd:.1%} — das Tageslimit ist "
                    f"{rm.max_daily_drawdown:.0%}. Ab dort schließt {self.bot_name} alle Positionen.")
        cooldown = rm.cooldown_until
        if cooldown and datetime.now() < cooldown:
            return f"Nach mehreren Verlusten in Folge macht {self.bot_name} bis {cooldown:%H:%M} Pause."

        regime = getattr(bot, '_last_regime_name', None)
        regime_texts = {
            'high_vol_event': "Sehr hohe Volatilität erkannt. Die Positionsgrößen werden automatisch reduziert.",
            'event_driven': "Ein Makro-Event steht an (z. B. Zinsentscheid). Der Bot handelt mit weniger Risiko.",
            'low_vol_chop': "Der Markt ist sehr ruhig und seitwärts. Es sind kaum Signale zu erwarten.",
            'trending': "Der Markt ist in einem klaren Trend. Gute Bedingungen für Rücksetzer-Käufe.",
            'ranging': "Der Markt pendelt in einer Spanne. Der Bot achtet auf Extreme.",
        }
        if regime in regime_texts:
            return regime_texts[regime]

        equity = bot.portfolio.equity
        free = (1 - locked_margin / equity) if equity > 0 else 1.0
        if n_positions == 0:
            return f"Keine offenen Positionen — {self.bot_name} wartet auf ein sauberes Signal."
        return f"{n_positions} offene Position{'en' if n_positions != 1 else ''}, {free:.0%} des Kapitals frei."

    def _allocation(self, positions) -> List[Dict]:
        items = []
        for pos in positions:
            value = pos.size / pos.leverage + pos.unrealized_pnl
            items.append({
                'coin': coin_of(pos.symbol),
                'name': coin_name(pos.symbol),
                'side': pos.side,
                'value': max(0.0, value),
                'pnl': pos.unrealized_pnl,
            })
        items.append({
            'coin': 'EUR',
            'name': 'EUR (Reserve)',
            'side': None,
            'value': max(0.0, self.bot.portfolio.balance),
            'pnl': 0.0,
        })
        total = sum(i['value'] for i in items) or 1.0
        for i in items:
            i['share'] = i['value'] / total
        return items

    def _strategies(self, stats: List[Tuple[str, int, int, float]]) -> List[Dict]:
        cfg = self.bot.config.get('strategies', {})
        by_key = {s: (n, w, pnl) for s, n, w, pnl in stats}
        keys = [k for k in ('confluence', 'momentum', 'scalper', 'ml') if cfg.get(k, {}).get('enabled')]
        keys += [k for k in by_key if k not in keys]
        out = []
        for k in keys:
            n, w, pnl = by_key.get(k, (0, 0, 0.0))
            out.append({
                'key': k,
                'name': strategy_name(k),
                'enabled': bool(cfg.get(k, {}).get('enabled')),
                'trades': n,
                'win_rate': (w / n) if n else None,
                'pnl': pnl,
            })
        return out

    def _settings(self) -> Dict:
        cfg = self.bot.config
        general = cfg.get('general', {})
        risk = cfg.get('risk', {})
        rm = self.bot.risk_manager
        pairs = list(getattr(self.bot.crypto_feed, 'pairs', []))
        return {
            'mode': general.get('mode', 'paper'),
            'shadow_mode': general.get('shadow_mode', True),
            'start_capital': general.get('start_capital'),
            'base_currency': general.get('base_currency', 'EUR'),
            'pairs': pairs,
            'max_risk_per_trade': rm.max_risk_per_trade,
            'max_daily_drawdown': rm.max_daily_drawdown,
            'max_position_size': risk.get('max_position_size'),
            'max_leverage': rm.max_leverage,
            'max_concurrent_positions': rm.max_concurrent_positions,
            'spot_only': self.bot.spot_only,
            'taker_fee': self.bot.order_engine.fees.get('crypto_taker'),
            'min_orders': {s: self.bot._min_order_amount(s) for s in pairs},
        }
