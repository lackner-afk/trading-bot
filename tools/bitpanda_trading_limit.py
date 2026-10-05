#!/usr/bin/env python3
"""
Tägliche Handelslimits für einen Bitpanda-API-Key setzen.

Doku: docs/BITPANDA_API_LIMITS.md

Ohne --confirm wird nur angezeigt, was geschickt würde (Trockenlauf).

    python tools/bitpanda_trading_limit.py --buy 150 --sell 300 --currency-id <UUID>
    python tools/bitpanda_trading_limit.py --buy 150 --sell 300 --currency-id <UUID> --confirm

Achtung: Bitpanda erlaubt pro Key nur EINMAL ein POST — danach kommt 409.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import aiohttp
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
BASE_URL = 'https://api.public.bitpanda.com'
ENDPOINT = '/v1/trading-limits'


def build_payload(buy: float, sell: float, currency_id: str) -> Dict[str, Any]:
    """Prüft die Eingaben und baut den Request-Body."""
    if buy <= 0 or sell <= 0:
        raise ValueError("Limits müssen größer als 0 sein.")
    try:
        uuid.UUID(currency_id)
    except ValueError:
        raise ValueError(f"'{currency_id}' ist keine gültige UUID.")
    return {'buy_limit': buy, 'sell_limit': sell, 'currency_id': currency_id}


async def set_limits(api_key: str, payload: Dict[str, Any]) -> int:
    """Schickt das POST und gibt einen Exit-Code zurück."""
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        async with session.post(BASE_URL + ENDPOINT, json=payload) as resp:
            text = await resp.text()
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = {'raw': text}

    if resp.status == 200:
        data = body.get('data', {})
        print("✅ Limits gesetzt:")
        print(f"   Kauflimit:    {data.get('buy_limit')}  (heute übrig: {data.get('buy_budget_remaining')})")
        print(f"   Verkaufslimit: {data.get('sell_limit')}  (heute übrig: {data.get('sell_budget_remaining')})")
        print(f"   Währung:      {data.get('currency_id')}")
        return 0

    code = body.get('error', {}).get('code') if isinstance(body, dict) else None
    if resp.status == 409:
        print(f"⚠️  Für diesen Key sind schon Limits gesetzt (409, code={code}).")
        print("   Ändern geht nicht per POST — siehe docs/BITPANDA_API_LIMITS.md.")
    elif resp.status == 400:
        print(f"❌ Ungültige Parameter (400, code={code}).")
    elif resp.status == 401:
        print("❌ API-Key abgelehnt (401) — Key falsch, abgelaufen oder ohne Berechtigung.")
    else:
        print(f"❌ HTTP {resp.status}: {body}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Bitpanda: tägliche Handelslimits pro API-Key setzen")
    parser.add_argument('--buy', type=float, required=True, help="Tägliches Kauflimit")
    parser.add_argument('--sell', type=float, required=True, help="Tägliches Verkaufslimit")
    parser.add_argument('--currency-id', required=True, help="UUID der Währung (z. B. EUR)")
    parser.add_argument('--confirm', action='store_true', help="Wirklich senden (sonst Trockenlauf)")
    args = parser.parse_args()

    try:
        payload = build_payload(args.buy, args.sell, args.currency_id)
    except ValueError as e:
        print(f"❌ {e}")
        return 2

    if args.sell < args.buy:
        print("⚠️  Verkaufslimit ist kleiner als Kauflimit — der Bot könnte Positionen "
              "dann nicht mehr schließen. Bewusst so gewollt?")

    print(f"POST {BASE_URL}{ENDPOINT}")
    print(json.dumps(payload, indent=2))

    if not args.confirm:
        print("\nTrockenlauf — nichts gesendet. Mit --confirm wirklich setzen.")
        return 0

    load_dotenv(PROJECT_ROOT / 'config' / 'secrets.env')
    api_key = os.getenv('BITPANDA_API_KEY')
    if not api_key:
        print("❌ BITPANDA_API_KEY fehlt in config/secrets.env.")
        return 2

    return asyncio.run(set_limits(api_key, payload))


if __name__ == '__main__':
    sys.exit(main())
