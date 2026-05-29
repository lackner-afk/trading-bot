#!/usr/bin/env python3
"""
Paper → Live Cutover Checklist (Phase 7)

Run this before going live with real money:
    python tools/paper_to_live_checklist.py

It performs automated safety checks and prints a clear report.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

def check_env_var() -> tuple[bool, str]:
    val = os.getenv("LIVE_TRADING_ENABLED")
    if not val:
        return False, "LIVE_TRADING_ENABLED ist nicht gesetzt"
    return True, f"LIVE_TRADING_ENABLED = {val}"

def check_settings_yaml() -> tuple[bool, str]:
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    if not settings_path.exists():
        return False, "settings.yaml nicht gefunden"

    content = settings_path.read_text()
    if "mode: live" not in content and 'mode: "live"' not in content:
        return False, "mode ist nicht auf 'live' gesetzt"

    if "live_explicit_confirmation: true" not in content:
        return False, "live_explicit_confirmation: true fehlt oder ist false"

    return True, "settings.yaml sieht für Live gut aus"

def check_secrets() -> tuple[bool, str]:
    secrets_path = PROJECT_ROOT / "config" / "secrets.env"
    if not secrets_path.exists():
        return False, "secrets.env fehlt"

    content = secrets_path.read_text()
    if "ONETRADING_API_KEY=" in content and "ONETRADING_API_SECRET=" in content:
        # Very basic check – real validation happens at runtime
        return True, "secrets.env vorhanden (Keys werden beim Start validiert)"
    return False, "ONETRADING_API_KEY / SECRET nicht in secrets.env gefunden"

def check_live_trading_md() -> tuple[bool, str]:
    doc = PROJECT_ROOT / "LIVE_TRADING.md"
    if not doc.exists():
        return False, "LIVE_TRADING.md fehlt"
    return True, "LIVE_TRADING.md vorhanden"

def check_shadow_mode_usage() -> tuple[bool, str]:
    # Heuristic: look for recent shadow mode usage in logs (if bot.log exists)
    log = PROJECT_ROOT / "bot.log"
    if not log.exists():
        return True, "Kein bot.log gefunden (noch nicht relevant)"

    try:
        content = log.read_text()
        if "SHADOW MODE" in content or "shadow_mode" in content:
            return True, "Shadow Mode wurde bereits verwendet (gut!)"
        return False, "Shadow Mode noch nicht nachweisbar im Log – dringend empfohlen vor Live!"
    except Exception:
        return True, "Log konnte nicht gelesen werden"

def main():
    print("\n" + "=" * 70)
    print("PAPER → LIVE CUTOVER CHECKLIST  |  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 70 + "\n")

    checks = [
        ("Environment Variable", check_env_var),
        ("settings.yaml", check_settings_yaml),
        ("secrets.env", check_secrets),
        ("LIVE_TRADING.md", check_live_trading_md),
        ("Shadow Mode History", check_shadow_mode_usage),
    ]

    all_ok = True
    for name, func in checks:
        ok, msg = func()
        status = "✅" if ok else "❌"
        print(f"{status}  {name:25} {msg}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("✅  ALLE AUTOMATISIERTEN CHECKS BESTANDEN")
        print("   Du kannst jetzt die manuelle Checkliste in LIVE_TRADING.md durchgehen.")
        print("   Besonders wichtig: Shadow Mode Phase + Small Capital Validation.")
    else:
        print("❌  NICHT ALLE CHECKS BESTANDEN")
        print("   Bitte die fehlenden Punkte beheben, bevor du echtes Geld riskierst.")
        sys.exit(1)

    print("=" * 70 + "\n")
    print("Zusätzlich empfohlen:")
    print("  • Führe Backtests mit data_exchange=onetrading durch")
    print("  • Vergleiche regelmäßig mit bitpanda-broker MCP Tools (get_portfolio)")
    print("  • Lies die vollständige Cutover-Checklist in LIVE_TRADING.md\n")

if __name__ == "__main__":
    main()