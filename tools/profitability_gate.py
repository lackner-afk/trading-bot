#!/usr/bin/env python3
"""
Profitabilitäts-Gate: Ist der Bot nachweislich profitabel?

Die bestehende Go-Live-Checkliste (tools/paper_to_live_checklist.py) prüft
ausschließlich Config-Dateien und Datei-Existenz per Substring-Match. Sie
kann vollständig grün werden, während der Bot Geld verbrennt — kein einziger
Check liest trades.db.

Dieses Gate schließt die Lücke. Es ist bewusst konservativ: mehrere
Kennzahlen müssen gemeinsam stimmen, weil jede einzelne für sich
manipulierbar oder zufällig ist. Eine hohe Win-Rate ohne Profit Factor
bedeutet nichts (viele kleine Gewinne, ein großer Verlust), ein guter
Profit Factor aus drei Trades ebenso wenig.

Verwendung:
    python tools/profitability_gate.py
    python tools/profitability_gate.py --db trades.db --json

Exit-Code 0 = grün, 1 = nicht bestanden.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.performance import (  # noqa: E402
    DEFAULT_ROUND_TRIP_FEE,
    PerformanceReport,
    compute_performance,
    round_trip_fee_from_config,
)


# Konservatives Profil. Die letzten vier Kriterien sind Robustheitschecks:
# sie verhindern, dass ein Glückstreffer oder reines Gebührenrauschen das
# Gate öffnet.
DEFAULT_CRITERIA = {
    "min_trades": 100,
    "min_days": 30,
    "min_net_pnl": 0.0,
    "min_profit_factor": 1.30,
    "max_drawdown": 0.10,
    "min_sharpe": 1.0,
    "min_win_rate": 0.45,
    "min_expectancy_fee_multiple": 2.0,   # Erwartungswert > 2x Round-Trip-Fee
    "max_largest_win_share": 0.25,
    "min_positive_week_share": 0.55,
}


@dataclass
class Criterion:
    """Ein einzelnes Kriterium mit Ist- und Sollwert."""
    key: str
    label: str
    passed: bool
    actual: str
    required: str

    def line(self) -> str:
        mark = "OK " if self.passed else "NEIN"
        return f"  [{mark}] {self.label:<38} {self.actual:>14}  (Ziel: {self.required})"


@dataclass
class GateResult:
    """Gesamtergebnis der Prüfung."""
    passed: bool
    criteria: List[Criterion] = field(default_factory=list)
    report: Optional[PerformanceReport] = None
    blockers: List[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.criteria if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.criteria)

    def summary(self) -> str:
        return f"{self.passed_count} von {self.total_count} Kriterien erfüllt"

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "summary": self.summary(),
            "blockers": self.blockers,
            "criteria": [
                {
                    "key": c.key, "label": c.label, "passed": c.passed,
                    "actual": c.actual, "required": c.required,
                }
                for c in self.criteria
            ],
        }


def _fmt(value: float, kind: str = "num") -> str:
    if value == float("inf"):
        return "unendlich"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "eur":
        return f"{value:+.2f} EUR"
    return f"{value:.2f}"


def _load_settings() -> Dict:
    """settings.yaml lesen, ohne bei Fehlern das Gate zu blockieren."""
    path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def evaluate_gate(db_path: str = "trades.db",
                  criteria: Dict = None,
                  round_trip_fee: float = None) -> GateResult:
    """
    Wertet die Trade-Historie gegen die Kriterien aus.

    Wird sowohl vom CLI als auch von main.py (als harter Live-Guard) und von
    paper_to_live_checklist.py genutzt.

    `round_trip_fee` kommt per Default aus dem `fees:`-Block der settings.yaml.
    Die frühere feste 0.0012 stammte aus einer Futures-Gebührenstruktur und
    war für Bitpanda Fusion (Level 1: 0,25 % je Seite plus Spread) um Faktor
    ~4,5 zu niedrig — das Kriterium "Erwartungswert > 2x Round-Trip-Fee" wäre
    damit viel zu leicht zu erfüllen gewesen.
    """
    if round_trip_fee is None:
        round_trip_fee = round_trip_fee_from_config(_load_settings())

    crit = {**DEFAULT_CRITERIA, **(criteria or {})}
    report = compute_performance(db_path, round_trip_fee=round_trip_fee)
    result = GateResult(passed=False, report=report)

    if not Path(db_path).exists():
        result.blockers.append(f"Keine Trade-Datenbank unter {db_path}")
        return result

    if report.total_trades == 0:
        result.blockers.append("Keine abgeschlossenen Trades - Testphase hat noch nicht begonnen")
        return result

    add = result.criteria.append

    add(Criterion(
        "trades", "Abgeschlossene Trades",
        report.total_trades >= crit["min_trades"],
        str(report.total_trades), f">= {crit['min_trades']}",
    ))
    add(Criterion(
        "days", "Laufzeit (Tage)",
        report.trading_days >= crit["min_days"],
        str(report.trading_days), f">= {crit['min_days']}",
    ))
    add(Criterion(
        "net_pnl", "Netto-PNL nach Gebuehren",
        report.net_pnl > crit["min_net_pnl"],
        _fmt(report.net_pnl, "eur"), "> 0",
    ))
    add(Criterion(
        "profit_factor", "Profit Factor",
        report.profit_factor >= crit["min_profit_factor"],
        _fmt(report.profit_factor), f">= {crit['min_profit_factor']}",
    ))
    add(Criterion(
        "max_drawdown", "Max Drawdown",
        report.max_drawdown <= crit["max_drawdown"],
        _fmt(report.max_drawdown, "pct"), f"<= {crit['max_drawdown']:.0%}",
    ))
    add(Criterion(
        "sharpe", "Sharpe (taeglich, annualisiert)",
        report.sharpe_ratio >= crit["min_sharpe"],
        _fmt(report.sharpe_ratio), f">= {crit['min_sharpe']}",
    ))
    add(Criterion(
        "win_rate", "Win Rate",
        report.win_rate >= crit["min_win_rate"],
        _fmt(report.win_rate, "pct"), f">= {crit['min_win_rate']:.0%}",
    ))

    # Erwartungswert muss die Round-Trip-Gebühren deutlich schlagen, sonst
    # ist die vermeintliche Edge nur Gebührenrauschen.
    fee_threshold = round_trip_fee * crit["min_expectancy_fee_multiple"]
    add(Criterion(
        "expectancy", "Erwartungswert je Trade",
        report.expectancy_pct > fee_threshold,
        _fmt(report.expectancy_pct, "pct"), f"> {fee_threshold:.2%}",
    ))
    add(Criterion(
        "concentration", "Groesster Einzelgewinn (Anteil)",
        report.largest_win_share <= crit["max_largest_win_share"],
        _fmt(report.largest_win_share, "pct"), f"<= {crit['max_largest_win_share']:.0%}",
    ))
    add(Criterion(
        "consistency", "Anteil positiver Wochen",
        report.positive_week_share >= crit["min_positive_week_share"],
        _fmt(report.positive_week_share, "pct"), f">= {crit['min_positive_week_share']:.0%}",
    ))

    # Gegenprobe Spot-Modus: die Testphase muss gemessen haben, was live
    # ausführbar ist. Ein einziger Short oder gehebelter Trade in der DB
    # bedeutet, dass die Daten eine andere Strategie beschreiben.
    spot_clean = report.short_trades == 0 and report.leveraged_trades == 0
    add(Criterion(
        "spot_konform", "Spot-konform (keine Shorts/Hebel)",
        spot_clean,
        f"{report.short_trades} Shorts, {report.leveraged_trades} gehebelt",
        "0 / 0",
    ))

    result.passed = all(c.passed for c in result.criteria)
    if not result.passed:
        result.blockers = [c.label for c in result.criteria if not c.passed]

    return result


def format_report(result: GateResult) -> str:
    """Menschenlesbare Ausgabe."""
    lines = ["", "=" * 72, "PROFITABILITAETS-GATE", "=" * 72]

    r = result.report
    if r and r.total_trades:
        lines += [
            f"Zeitraum:  {r.first_trade:%Y-%m-%d} bis {r.last_trade:%Y-%m-%d} "
            f"({r.trading_days} Tage)",
            f"Trades:    {r.total_trades} ({r.winning_trades} Gewinne / "
            f"{r.losing_trades} Verluste)",
            f"Netto-PNL: {r.net_pnl:+.2f} EUR   Gebuehren: {r.total_fees:.2f} EUR",
            "-" * 72,
        ]

    if not result.criteria:
        lines.append("Auswertung nicht moeglich:")
        for b in result.blockers:
            lines.append(f"  - {b}")
        lines += ["=" * 72, ""]
        return "\n".join(lines)

    for c in result.criteria:
        lines.append(c.line())

    lines.append("-" * 72)
    lines.append(f"Ergebnis: {result.summary()}")

    if result.passed:
        lines += [
            "",
            "GRUEN - alle Kriterien erfuellt.",
            "Naechster Schritt ist NICHT sofort Live: erst Shadow Mode auf dem",
            "Ziel-Venue, dann kleines Kapital. Siehe LIVE_TRADING.md.",
        ]
    else:
        lines += ["", "ROT - offene Punkte:"]
        for b in result.blockers:
            lines.append(f"  - {b}")

    if r and r.per_strategy:
        lines += ["", "Je Strategie:"]
        for name, s in sorted(r.per_strategy.items(), key=lambda x: -x[1]['pnl']):
            wr = s['wins'] / s['trades'] if s['trades'] else 0
            lines.append(f"  {name:<20} {s['trades']:>4} Trades | "
                         f"WR {wr:>5.1%} | PNL {s['pnl']:+.2f} EUR")

    lines += ["=" * 72, ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prueft, ob der Bot die Profitabilitaets-Kriterien erfuellt"
    )
    parser.add_argument("--db", default="trades.db", help="Pfad zur Trade-Datenbank")
    parser.add_argument("--json", action="store_true", help="Ausgabe als JSON")
    parser.add_argument("--min-trades", type=int, help="Kriterium ueberschreiben")
    parser.add_argument("--min-days", type=int, help="Kriterium ueberschreiben")
    args = parser.parse_args()

    overrides = {}
    if args.min_trades is not None:
        overrides["min_trades"] = args.min_trades
    if args.min_days is not None:
        overrides["min_days"] = args.min_days

    result = evaluate_gate(args.db, criteria=overrides or None)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(result))

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
