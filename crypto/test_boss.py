"""Send one test message as Boss (@bossbtc2_bot) and print Telegram's raw answer.

Run this ON THE SERVER from the crypto/ folder:

    python3 test_boss.py

It tells you exactly why Boss is silent:
  * token missing / wrong      -> "401 Unauthorized"
  * bot not admin in channel   -> "403 Forbidden: bot is not a member..."
  * channel id wrong           -> "400 Bad Request: chat not found"
  * Markdown broken            -> "400 ... can't parse entities"
  * everything fine            -> "200 ok: true" and a message in the channel
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from dotenv import load_dotenv

load_dotenv(os.path.join(HERE, ".env"))
load_dotenv()

import requests

TOKEN = os.getenv("BOT4_TOKEN", "")
ADMIN = os.getenv("ADMIN_CHAT_ID", "") or os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
CHANNEL_FALLBACK = "-1004373560971"


def mask(v: str) -> str:
    return (v[:6] + "..." + v[-4:]) if len(v) > 10 else ("<empty>" if not v else "***")


print("cwd        :", os.getcwd())
print("BOT4_TOKEN :", mask(TOKEN))
print("ADMIN_CHAT_ID:", ADMIN or "(unset)")

if not TOKEN:
    print("\nFAIL: BOT4_TOKEN is missing. Add it to crypto/.env and restart.")
    raise SystemExit(1)

targets = []
for part in ADMIN.split(","):
    part = part.strip()
    if part:
        try:
            targets.append(int(part))
        except ValueError:
            pass
if not targets:
    targets = [int(CHANNEL_FALLBACK)]
print("targets    :", targets)

try:
    me = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe", timeout=15).json()
    print("getMe      :", me.get("ok"), (me.get("result") or {}).get("username"))
    if not me.get("ok"):
        print("FAIL: token rejected ->", me.get("description"))
        raise SystemExit(1)
except Exception as e:
    print("FAIL: cannot reach Telegram ->", repr(e))
    raise SystemExit(1)

text = "✅ *اختبار البوس*\nهذه رسالة تجريبية من @bossbtc2_bot.\nإذا وصلتك، فالتوكن والقناة سليمان."
for chat_id in targets:
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                          timeout=20)
        print(f"send {chat_id}: HTTP {r.status_code} {r.text[:220]}")
        if r.status_code == 400 and "parse" in r.text.lower():
            r2 = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                               json={"chat_id": chat_id, "text": text}, timeout=20)
            print(f"  retry without Markdown: HTTP {r2.status_code} {r2.text[:160]}")
    except Exception as e:
        print(f"send {chat_id}: FAIL {e!r}")

print("\nDONE")
