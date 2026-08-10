#!/usr/bin/env python3
"""
Laedt die vollstaendige Fear-&-Greed-Historie und legt sie im Cache ab.

Warum das ein eigener Schritt ist: der SentimentFactor ist im aktuellen
Faktorenset der einzige, der zuverlaessig Richtung UND hohen Score liefert.
Ein Backtest, der den heutigen F&G-Wert auf 90 Tage Historie anwendet, misst
genau den Faktor falsch, der die Strategie steuert — und zwar unauffaellig.

Einmal ausfuehren, danach laufen Backtests offline gegen den Cache:

    python tools/fetch_fng_history.py
    python tools/fetch_fng_history.py --show 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.factors.sentiment import SentimentFactor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Laedt die Fear-&-Greed-Historie in den lokalen Cache"
    )
    parser.add_argument("--cache", help="Pfad zur Cache-Datei")
    parser.add_argument("--show", type=int, default=5,
                        help="Zeigt die letzten N Tage")
    parser.add_argument("--offline", action="store_true",
                        help="Nur den vorhandenen Cache pruefen, nichts holen")
    args = parser.parse_args()

    config = {}
    if args.cache:
        config["cache_path"] = args.cache
    if args.offline:
        config["allow_network"] = False

    factor = SentimentFactor(config)

    if args.offline:
        factor._load_from_disk()
    else:
        print("Lade Fear-&-Greed-Historie von alternative.me ...")
        factor._ensure_history()

    history = factor._history
    if not history:
        print("FEHLER: keine Historie verfuegbar.")
        print(f"  Cache: {factor.cache_path}")
        print("  Ohne Historie ist der Backtest des Sentiment-Faktors wertlos.")
        return 1

    days = sorted(history)
    print(f"\nOK: {len(days)} Tage im Cache ({days[0]} bis {days[-1]})")
    print(f"Datei: {factor.cache_path}")

    if not factor.is_historical():
        print("\nWARNUNG: nur ein einziger Tag vorhanden — der Historie-Endpunkt")
        print("war nicht erreichbar. Fuer Backtests reicht das NICHT.")
        return 1

    if args.show > 0:
        print(f"\nLetzte {min(args.show, len(days))} Tage:")
        for day in days[-args.show:]:
            value = history[day]
            zone = ("Extreme Fear" if value < 25 else "Fear" if value < 45
                    else "Neutral" if value <= 55
                    else "Greed" if value <= 75 else "Extreme Greed")
            print(f"  {day}  {value:5.1f}  {zone}")

    # Verteilung — sagt direkt, wie oft der Bot ueberhaupt haette handeln koennen
    fear = sum(1 for v in history.values() if v < 45)
    neutral = sum(1 for v in history.values() if 45 <= v <= 55)
    greed = sum(1 for v in history.values() if v > 55)
    total = len(history)
    print(f"\nVerteilung ueber {total} Tage:")
    print(f"  Fear   (<45): {fear:5d}  ({fear / total:.1%})  -> Entries moeglich")
    print(f"  Neutral      : {neutral:5d}  ({neutral / total:.1%})  -> keine Richtung")
    print(f"  Greed  (>55): {greed:5d}  ({greed / total:.1%})  -> nur Exits")

    return 0


if __name__ == "__main__":
    sys.exit(main())
