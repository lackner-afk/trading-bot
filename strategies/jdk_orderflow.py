"""
JDK-Orderflow-Strategie — Key Levels + Orderflow-Bestätigung

Nachbau des Ansatzes von "JDK Analysis" (@The_JDK99 auf X, JDK-Analysis auf
TradingView). Sein Vorgehen, wie er es in seinen Posts beschreibt:

1. Key Levels aus dem Volumenprofil und aus VWAPs bestimmen:
   Range-Value-Area (VAH/VAL/POC), naked POCs (nPOC), Low-Volume-Nodes (LVN),
   Anchored VWAP vom Beginn des Aufwärtstrends, Session-VWAP, 50%-Level der Range.
2. Warten, bis der Preis eine Zone erreicht, in der mehrere Levels
   zusammenfallen ("50% level, AVWAP uptrend, rVAL and nPOC").
3. Erst handeln, wenn der Orderflow am Level Stärke zeigt (Absorption,
   Ablehnung) — nie blind ins Level kaufen.
4. Auktionslogik: Handelt der Preis unter der Range-VAL, ist er bärisch, bis
   er die VAL zurückerobert und darüber akzeptiert wird. Ein gescheiterter
   Ausbruch nach unten (Failed Auction) mit Rückeroberung ist ein Long-Setup
   mit Ziel POC/VAH.

Einschränkung: Echte Footprint-/CVD-Daten liefert CCXT nicht. Das Orderflow-
Delta wird aus OHLCV geschätzt (Close-Lage in der Kerze × Volumen). Das ist
eine Näherung an seine Footprint-Lesart, kein Ersatz.

Shorts werden über gespiegelte Kerzen (Preise negiert) mit derselben
Long-Logik berechnet — so bleibt die Regelmenge für beide Seiten identisch.
Auf Spot-Börsen (Bitpanda Fusion) gilt long_only: true.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .crypto_scalper import ScalperSignal, SignalType


# ============================================================
# Level-Berechnung (reine Funktionen, ohne Zustand)
# ============================================================

@dataclass
class VolumeProfile:
    """Volumenprofil einer Range"""
    poc: float
    vah: float
    val: float
    range_high: float
    range_low: float
    lvns: List[float] = field(default_factory=list)


def estimate_delta(df: pd.DataFrame) -> pd.Series:
    """
    Geschätztes Orderflow-Delta pro Kerze.

    Schließt die Kerze am Hoch, war (näherungsweise) alles aggressives Kaufen
    (+Volumen), am Tief alles Verkaufen (−Volumen), in der Mitte neutral.
    """
    rng = (df['high'] - df['low']).replace(0, np.nan)
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / rng
    return (clv.fillna(0.0) * df['volume']).astype(float)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range der letzten `period` Kerzen"""
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def volume_profile(df: pd.DataFrame, bins: int = 48,
                   value_area_pct: float = 0.70,
                   lvn_threshold: float = 0.35) -> Optional[VolumeProfile]:
    """
    Volumenprofil: verteilt das Volumen jeder Kerze gleichmäßig über ihre
    High-Low-Spanne und bestimmt POC, Value Area (70%) und LVNs.
    """
    if df is None or len(df) < 10:
        return None
    lows = df['low'].to_numpy(dtype=float)
    highs = df['high'].to_numpy(dtype=float)
    vols = df['volume'].to_numpy(dtype=float)
    r_low, r_high = float(lows.min()), float(highs.max())
    if r_high <= r_low or vols.sum() <= 0:
        return None

    edges = np.linspace(r_low, r_high, bins + 1)
    # Überlappung jeder Kerze mit jedem Bin (n × bins)
    overlap = (np.minimum(highs[:, None], edges[None, 1:])
               - np.maximum(lows[:, None], edges[None, :-1])).clip(min=0)
    spans = highs - lows
    weights = np.zeros_like(overlap)
    has_span = spans > 0
    weights[has_span] = overlap[has_span] / spans[has_span, None]
    # Kerzen ohne Spanne: gesamtes Volumen in den Bin ihres Preises
    if (~has_span).any():
        idx = np.clip(np.searchsorted(edges, lows[~has_span], side='right') - 1, 0, bins - 1)
        weights[np.where(~has_span)[0], idx] = 1.0
    hist = (weights * vols[:, None]).sum(axis=0)

    poc_idx = int(hist.argmax())
    centers = (edges[:-1] + edges[1:]) / 2

    # Value Area: vom POC aus jeweils den volumenstärkeren Nachbarn aufnehmen
    total = hist.sum()
    lo = hi = poc_idx
    acc = hist[poc_idx]
    while acc < value_area_pct * total and (lo > 0 or hi < bins - 1):
        below = hist[lo - 1] if lo > 0 else -1.0
        above = hist[hi + 1] if hi < bins - 1 else -1.0
        if above >= below:
            hi += 1
            acc += hist[hi]
        else:
            lo -= 1
            acc += hist[lo]

    # LVNs: lokale Minima im geglätteten Profil, deutlich unter dem POC-Volumen
    smooth = np.convolve(hist, np.ones(3) / 3, mode='same')
    lvns = [
        float(centers[i]) for i in range(2, bins - 2)
        if smooth[i] < smooth[i - 1] and smooth[i] <= smooth[i + 1]
        and smooth[i] < lvn_threshold * smooth[poc_idx]
    ]

    return VolumeProfile(
        poc=float(centers[poc_idx]),
        vah=float(edges[hi + 1]),
        val=float(edges[lo]),
        range_high=r_high,
        range_low=r_low,
        lvns=lvns,
    )


def naked_pocs(df: pd.DataFrame, bins: int = 24) -> List[float]:
    """
    POCs abgeschlossener Tage (UTC), die seither nicht mehr gehandelt wurden.
    Die laufende Tagessession zählt nicht.
    """
    if df is None or len(df) < 48:
        return []
    days = pd.to_datetime(df['timestamp']).dt.date.to_numpy()
    unique_days = list(dict.fromkeys(days))
    result = []
    for day in unique_days[:-1]:
        mask = days == day
        day_df = df[mask]
        if len(day_df) < 12:  # angebrochener erster Tag im Fenster
            continue
        prof = volume_profile(day_df, bins=bins)
        if prof is None:
            continue
        after = df.iloc[np.where(mask)[0][-1] + 1:]
        touched = ((after['low'] <= prof.poc) & (after['high'] >= prof.poc)).any()
        if not touched:
            result.append(prof.poc)
    return result


def session_vwap(df: pd.DataFrame) -> Optional[float]:
    """VWAP der laufenden Tagessession (ab 00:00 UTC)"""
    ts = pd.to_datetime(df['timestamp'])
    session = df[ts.dt.date == ts.iloc[-1].date()]
    vol = session['volume'].sum()
    if vol <= 0:
        return None
    typical = (session['high'] + session['low'] + session['close']) / 3
    return float((typical * session['volume']).sum() / vol)


def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> Optional[float]:
    """VWAP ab der Kerze `anchor_idx` (Positionsindex) bis zur letzten Kerze"""
    part = df.iloc[anchor_idx:]
    vol = part['volume'].sum()
    if vol <= 0:
        return None
    typical = (part['high'] + part['low'] + part['close']) / 3
    return float((typical * part['volume']).sum() / vol)


# ============================================================
# Strategie
# ============================================================

@dataclass
class JDKSignal:
    """Ergebnis der Level-/Orderflow-Analyse (seitenneutral)"""
    side: str              # 'long' | 'short'
    setup: str             # 'level_test' | 'failed_auction'
    entry: float
    stop_loss: float
    take_profit: float
    confidence: float
    rr: float              # Chance/Risiko nach Gebühren
    levels: List[str]      # beteiligte Levels (für Log/Telegram)
    confirmations: List[str]
    target_name: str
    atr: float
    timestamp: Optional[pd.Timestamp] = None

    def reason(self) -> str:
        return (f"JDK {self.setup} {self.side.upper()}: Zone [{', '.join(self.levels)}] | "
                f"Orderflow: {', '.join(self.confirmations)} | Ziel {self.target_name} | "
                f"CRV {self.rr:.1f}")


class JDKOrderflowStrategy:
    """Key-Level + Orderflow-Strategie nach JDK Analysis (siehe Modul-Docstring)"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        c = self.config
        self.logger = logging.getLogger('JDK')

        self.pairs: List[str] = c.get('pairs', ['BTC_EUR', 'ETH_EUR', 'SOL_EUR'])
        self.timeframe: str = c.get('timeframe', '1h')
        self.long_only: bool = c.get('long_only', True)
        self.leverage: int = int(c.get('leverage', 1))

        # Level-Berechnung
        self.lookback_bars: int = c.get('lookback_bars', 240)
        # Die Range-Levels stammen aus der Zeit VOR dem Test — die letzten
        # Kerzen (der Test selbst) sollen das Level nicht mitverschieben.
        self.profile_exclude_bars: int = c.get('profile_exclude_bars', 12)
        self.profile_bins: int = c.get('profile_bins', 48)
        self.value_area_pct: float = c.get('value_area_pct', 0.70)
        self.min_anchor_age: int = c.get('min_anchor_age_bars', 12)

        # Setup 1: Key-Level-Test
        self.min_confluence_levels: int = c.get('min_confluence_levels', 2)
        self.zone_atr_mult: float = c.get('zone_atr_mult', 0.35)
        self.zone_min_pct: float = c.get('zone_min_pct', 0.002)
        self.wick_ratio: float = c.get('wick_ratio', 0.4)
        self.divergence_lookback: int = c.get('divergence_lookback', 12)
        self.volume_spike_mult: float = c.get('volume_spike_mult', 1.5)

        # Setup 2: Failed Auction / VAL-Reclaim
        self.acceptance_bars: int = c.get('acceptance_bars', 2)
        self.reclaim_lookback: int = c.get('reclaim_lookback', 24)
        self.max_deviation_atr: float = c.get('max_deviation_atr', 4.0)

        # Risiko / Ziele
        self.sl_buffer_atr: float = c.get('sl_buffer_atr', 0.25)
        self.min_rr: float = c.get('min_rr', 2.0)
        self.min_tp_pct: float = c.get('min_tp_pct', 0.010)
        self.fee_pct: float = c.get('fee_pct', 0.0025)
        self.breakeven_at_r: float = c.get('breakeven_at_r', 1.0)
        self.cooldown_bars: int = c.get('cooldown_bars', 6)

        self._cooldown_until: Dict[str, pd.Timestamp] = {}
        self.signal_history: List[JDKSignal] = []
        self.highest_prices: Dict[str, float] = {}  # Schnittstelle wie andere Strategien

    # ---------- öffentliche Schnittstelle ----------

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 price: Optional[float] = None) -> Optional[JDKSignal]:
        """
        Analysiert abgeschlossene Kerzen. Zeitbasis für den Cooldown ist der
        Kerzen-Zeitstempel — dadurch identisch im Live-Bot und im Backtest.
        """
        min_bars = self.lookback_bars // 2
        if candles is None or len(candles) < min_bars:
            return None
        df = candles.tail(self.lookback_bars).reset_index(drop=True)
        now = pd.Timestamp(df['timestamp'].iloc[-1])

        blocked = self._cooldown_until.get(symbol)
        if blocked is not None and now < blocked:
            return None

        entry = float(price) if price else float(df['close'].iloc[-1])

        signal = self._evaluate_long(df, entry)
        if signal is None and not self.long_only:
            signal = self._evaluate_short(df, entry)
        if signal is None:
            return None

        signal.timestamp = now
        self._cooldown_until[symbol] = now + self._bar_delta() * self.cooldown_bars
        self.signal_history.append(signal)
        return signal

    def analyze(self, symbol: str, candles: pd.DataFrame,
                current_price: float) -> Optional[ScalperSignal]:
        """Live-Schnittstelle für main.py (gleiches Signalformat wie Momentum)"""
        sig = self.evaluate(symbol, candles, current_price)
        if sig is None:
            return None
        self.logger.info(f"{symbol}: {sig.reason()} @ {sig.entry:.2f} "
                         f"(SL {sig.stop_loss:.2f} / TP {sig.take_profit:.2f})")
        return ScalperSignal(
            signal_type=SignalType.LONG if sig.side == 'long' else SignalType.SHORT,
            symbol=symbol,
            price=current_price,
            confidence=sig.confidence,
            reason=sig.reason(),
            take_profit=sig.take_profit,
            stop_loss=sig.stop_loss,
            suggested_leverage=self.leverage,
            atr_value=sig.atr,
        )

    def breakeven_stop(self, side: str, entry: float, stop_loss: float,
                       current_price: float) -> Optional[float]:
        """
        Neuer Stop auf Break-Even (inkl. Round-Trip-Gebühren), sobald der Trade
        `breakeven_at_r` R im Plus ist. None = Stop unverändert lassen.
        """
        if not self.breakeven_at_r or stop_loss is None or entry <= 0:
            return None
        risk = abs(entry - stop_loss)
        fee_buffer = entry * self.fee_pct * 2
        if side == 'long':
            be = entry + fee_buffer
            if stop_loss < be and current_price >= entry + risk * self.breakeven_at_r:
                return be
        else:
            be = entry - fee_buffer
            if stop_loss > be and current_price <= entry - risk * self.breakeven_at_r:
                return be
        return None

    def get_statistics(self) -> Dict:
        if not self.signal_history:
            return {'total_signals': 0}
        setups: Dict[str, int] = {}
        for s in self.signal_history:
            setups[s.setup] = setups.get(s.setup, 0) + 1
        return {
            'total_signals': len(self.signal_history),
            'setups': setups,
            'avg_confidence': sum(s.confidence for s in self.signal_history) / len(self.signal_history),
            'avg_rr': sum(s.rr for s in self.signal_history) / len(self.signal_history),
        }

    # ---------- Kernlogik (Long; Shorts über Spiegelung) ----------

    def _evaluate_short(self, df: pd.DataFrame, entry: float) -> Optional[JDKSignal]:
        """Short = Long-Logik auf gespiegelten Preisen (Hoch ↔ Tief, Vorzeichen negiert)"""
        mirrored = df.copy()
        mirrored['open'] = -df['open']
        mirrored['close'] = -df['close']
        mirrored['high'] = -df['low']
        mirrored['low'] = -df['high']
        sig = self._evaluate_long(mirrored, -entry)
        if sig is None:
            return None
        sig.side = 'short'
        sig.confirmations = [self._MIRROR_NAMES.get(c, c) for c in sig.confirmations]
        sig.entry = -sig.entry
        sig.stop_loss = -sig.stop_loss
        sig.take_profit = -sig.take_profit
        sig.levels = [self._MIRROR_NAMES.get(lv, lv) for lv in sig.levels]
        sig.target_name = self._MIRROR_NAMES.get(sig.target_name, sig.target_name)
        return sig

    # In gespiegelten Kerzen ist die VAL die echte VAH, das Tief das echte Hoch usw.
    _MIRROR_NAMES = {
        'VAL': 'VAH', 'VAH': 'VAL', 'AVWAP↑': 'AVWAP↓',
        'Range-Hoch': 'Range-Tief', 'VAL-Reclaim': 'VAH-Reclaim',
        'Kaufdelta': 'Verkaufsdelta',
    }

    def _evaluate_long(self, df: pd.DataFrame, entry: float) -> Optional[JDKSignal]:
        atr = calculate_atr(df, 14)
        if not np.isfinite(atr) or atr <= 0:
            return None

        base = df.iloc[:-self.profile_exclude_bars] if self.profile_exclude_bars else df
        profile = volume_profile(base, bins=self.profile_bins, value_area_pct=self.value_area_pct)
        if profile is None:
            return None
        delta = estimate_delta(df)

        # Setup 2 zuerst: der Reclaim ist das stärkere, seltenere Signal
        signal = self._failed_auction_long(df, entry, atr, profile, delta)
        if signal is None:
            signal = self._level_test_long(df, entry, atr, profile, delta)
        return signal

    def _support_levels(self, df: pd.DataFrame, profile: VolumeProfile) -> List[Tuple[str, float]]:
        """Alle Kandidaten-Levels mit Namen"""
        levels: List[Tuple[str, float]] = [
            ('VAL', profile.val),
            ('POC', profile.poc),
            ('VAH', profile.vah),
            ('Range-50%', (profile.range_high + profile.range_low) / 2),
        ]
        levels += [('LVN', p) for p in profile.lvns]
        levels += [('nPOC', p) for p in naked_pocs(df)]

        # AVWAP ab dem Tief der Range = "uptrend AVWAP"
        anchor = int(df['low'].to_numpy().argmin())
        if anchor <= len(df) - 1 - self.min_anchor_age:
            av = anchored_vwap(df, anchor)
            if av is not None:
                levels.append(('AVWAP↑', av))

        sv = session_vwap(df)
        if sv is not None:
            levels.append(('Session-VWAP', sv))
        return levels

    def _orderflow_confirmation(self, df: pd.DataFrame, delta: pd.Series,
                                zone_top: float) -> List[str]:
        """
        Orderflow-Stärke an der Zone (Proxy für Footprint/CVD):
        - Absorption: neues Tief, aber das kumulierte Delta macht ein höheres Tief
          (Verkäufer drücken, der Preis gibt nicht mehr nach)
        - Ablehnung: Docht unter die Zone, Schluss darüber, Kerze mit Kaufdelta
        - Volumen-Spike als Zusatz (nur zusammen mit einem der beiden)
        """
        last = df.iloc[-1]
        confirmations: List[str] = []

        span = last['high'] - last['low']
        lower_wick = min(last['open'], last['close']) - last['low']
        if (span > 0 and last['low'] <= zone_top < last['close']
                and lower_wick >= self.wick_ratio * span and delta.iloc[-1] > 0):
            confirmations.append('Ablehnung')

        n = self.divergence_lookback
        if len(df) > n + 3:
            recent, prior = slice(-3, None), slice(-n, -3)
            cvd = delta.iloc[-n:].cumsum()
            lows = df['low'].iloc[-n:]
            if (lows.iloc[recent].min() < lows.iloc[prior].min()
                    and cvd.iloc[recent].min() > cvd.iloc[prior].min()
                    and last['close'] > zone_top):
                confirmations.append('Absorption')

        if confirmations:
            avg_vol = df['volume'].iloc[-21:-1].mean()
            if avg_vol > 0 and last['volume'] >= self.volume_spike_mult * avg_vol:
                confirmations.append('Volumen')
        return confirmations

    def _pick_target(self, entry: float, stop: float, candidates: List[Tuple[str, float]]
                     ) -> Optional[Tuple[str, float, float]]:
        """Erstes Level über dem Einstieg, das nach Gebühren das Mindest-CRV bringt"""
        risk_pct = (entry - stop) / abs(entry) + 2 * self.fee_pct
        if risk_pct <= 0:
            return None
        for name, price in sorted(candidates, key=lambda x: x[1]):
            reward_pct = (price - entry) / abs(entry)
            if reward_pct < self.min_tp_pct:
                continue
            rr = float((reward_pct - 2 * self.fee_pct) / risk_pct)
            if rr >= self.min_rr:
                return name, float(price), rr
        return None

    def _level_test_long(self, df, entry, atr, profile, delta) -> Optional[JDKSignal]:
        """Setup 1: Test einer Zone aus ≥ N zusammenfallenden Levels + Orderflow-Bestätigung"""
        last = df.iloc[-1]
        # Unter der Range-VAL ist der Markt bärisch — erst nach Reclaim (Setup 2) long
        if last['close'] < profile.val:
            return None

        tol = max(self.zone_atr_mult * atr, self.zone_min_pct * abs(entry))
        levels = self._support_levels(df, profile)
        touched = [(n, p) for n, p in levels
                   if last['low'] - tol <= p <= last['close'] and p < entry]
        # Mehrere LVNs/nPOCs zählen nur einmal als Level-Typ
        kinds = {n for n, _ in touched}
        if len(kinds) < self.min_confluence_levels:
            return None

        # Zone = nur die eng beieinander liegenden Levels um das höchste berührte
        zone_top = max(p for _, p in touched)
        zone = [(n, p) for n, p in touched if zone_top - p <= 2 * tol]
        if len({n for n, _ in zone}) < self.min_confluence_levels:
            return None
        zone_low = min(p for _, p in zone)

        confirmations = self._orderflow_confirmation(df, delta, zone_top)
        if 'Ablehnung' not in confirmations and 'Absorption' not in confirmations:
            return None

        stop = min(last['low'], zone_low) - self.sl_buffer_atr * atr
        targets = [(n, p) for n, p in levels if p > entry]
        targets.append(('Range-Hoch', profile.range_high))
        pick = self._pick_target(entry, stop, targets)
        if pick is None:
            return None
        target_name, target, rr = pick

        confidence = 0.5 + 0.08 * (len({n for n, _ in zone}) - self.min_confluence_levels)
        confidence += 0.12 if 'Absorption' in confirmations else 0.0
        confidence += 0.08 if 'Ablehnung' in confirmations else 0.0
        confidence += 0.05 if 'Volumen' in confirmations else 0.0

        return JDKSignal(
            side='long', setup='level_test', entry=entry, stop_loss=float(stop),
            take_profit=float(target), confidence=min(0.95, confidence), rr=rr,
            levels=sorted({n for n, _ in zone}), confirmations=confirmations,
            target_name=target_name, atr=atr,
        )

    def _failed_auction_long(self, df, entry, atr, profile, delta) -> Optional[JDKSignal]:
        """
        Setup 2: Failed Auction unter der Range-VAL + Reclaim mit Akzeptanz.
        Kürzlich unter VAL geschlossen, jetzt `acceptance_bars` Schlusskurse in
        Folge wieder darüber, getragen von Kaufdelta.
        """
        k = self.acceptance_bars
        closes = df['close']
        if len(df) < self.reclaim_lookback + k + 1:
            return None

        accepted = (closes.iloc[-k:] > profile.val).all()
        fresh = closes.iloc[-k - 1] <= profile.val
        if not (accepted and fresh):
            return None

        # Der Ausflug unter die VAL: zusammenhängende Schlusskurse darunter
        # direkt vor dem Reclaim. Dauert er zu lange, hat der Markt unter der
        # Value Area akzeptiert — dann ist es kein Fehlausbruch mehr.
        start = len(df) - k - 1
        while start > 0 and closes.iloc[start - 1] <= profile.val:
            start -= 1
        if (len(df) - k) - start > self.reclaim_lookback:
            return None
        deviation_low = float(df['low'].iloc[start:].min())
        depth = profile.val - deviation_low
        # Zu tief unter der VAL ist kein Fehlausbruch mehr, sondern ein Bruch der Range
        if depth <= 0 or depth > self.max_deviation_atr * atr:
            return None
        if delta.iloc[-k:].sum() <= 0:
            return None

        stop = deviation_low - self.sl_buffer_atr * atr
        targets = [('POC', profile.poc), ('VAH', profile.vah), ('Range-Hoch', profile.range_high)]
        pick = self._pick_target(entry, stop, targets)
        if pick is None:
            return None
        target_name, target, rr = pick

        confirmations = ['VAL-Reclaim', 'Kaufdelta']
        confidence = 0.6
        avg_vol = df['volume'].iloc[-21 - k:-k].mean()
        if avg_vol > 0 and df['volume'].iloc[-k:].mean() >= self.volume_spike_mult * avg_vol:
            confirmations.append('Volumen')
            confidence += 0.1

        return JDKSignal(
            side='long', setup='failed_auction', entry=entry, stop_loss=float(stop),
            take_profit=float(target), confidence=confidence, rr=rr,
            levels=['VAL'], confirmations=confirmations, target_name=target_name, atr=atr,
        )

    def _bar_delta(self) -> timedelta:
        unit = self.timeframe[-1]
        value = int(self.timeframe[:-1])
        return {'m': timedelta(minutes=value), 'h': timedelta(hours=value),
                'd': timedelta(days=value)}[unit]
