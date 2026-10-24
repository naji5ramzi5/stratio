"""Verify all four Telegram bots and the channel before deploying.

Read-only by default: it calls getMe / getChat / getChatMember only, so it
sends NOTHING to the channel.

    python3 verify_bots.py            # check tokens + channel rights
    python3 verify_bots.py --send     # also send one real test message as Boss

It answers the exact question "why is Boss silent?":
  * bad token            -> getMe ok=false (401)
  * bot not in channel   -> getChatMember status=left / 400
  * bot cannot post      -> status=member without can_post_messages
  * wrong channel id     -> getChat 400 chat not found
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from dotenv import load_dotenv

load_dotenv(os.path.join(HERE, ".env"))
load_dotenv()
load_dotenv(os.path.join(HERE, "..", ".env"))

import requests

API = "https://api.telegram.org/bot{token}/{method}"
FALLBACK_CHAT = "-1004373560971"

BOTS = [
    ("Bot 1 (Predictions)", "TELEGRAM_PREDICTION_TOKEN"),
    ("Bot 2 (Liquidity)", "TELEGRAM_LIQUIDITY_TOKEN"),
    ("Bot 3 (Predictor)", "TELEGRAM_PREDICTOR_TOKEN"),
    ("Bot 4 (Boss)", "BOT4_TOKEN"),
]


def mask(v):
    return (v[:6] + "..." + v[-4:]) if len(v) > 10 else ("<empty>" if not v else "***")


def call(token, method, **params):
    try:
        r = requests.post(API.format(token=token, method=method), json=params, timeout=20)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"ok": False, "description": r.text[:200]}
    except Exception as e:
        return 0, {"ok": False, "description": repr(e)}


def candidate_chats():
    ids = []
    for key in ("ADMIN_CHAT_ID", "TELEGRAM_ADMIN_CHAT_ID"):
        for part in (os.getenv(key, "") or "").split(","):
            part = part.strip()
            if part:
                try:
                    ids.append((key, int(part)))
                except ValueError:
                    pass
    if not any(i < 0 for _, i in ids):
        ids.append(("fallback", int(FALLBACK_CHAT)))
    # de-duplicate, keep order
    seen, out = set(), []
    for src, i in ids:
        if i not in seen:
            seen.add(i)
            out.append((src, i))
    return out


print("cwd:", os.getcwd())
chats = candidate_chats()
print("channel candidates:", ", ".join(f"{i} ({src})" for src, i in chats))

# ── 1. Tokens ───────────────────────────────────────────────────────────────
print("\n=== TOKENS ===")
bot_ids = {}
for label, key in BOTS:
    token = os.getenv(key, "")
    if not token:
        print(f"  [FAIL] {label:22s} {key} is MISSING")
        continue
    status, body = call(token, "getMe")
    if body.get("ok"):
        me = body["result"]
        bot_ids[label] = me["id"]
        print(f"  [ OK ] {label:22s} @{me.get('username')}  ({mask(token)})")
    else:
        print(f"  [FAIL] {label:22s} HTTP {status}: {body.get('description')}")

# ── 2. Channel access ───────────────────────────────────────────────────────
print("\n=== CHANNEL ACCESS (Bot 4 / Boss must be able to post) ===")
boss = os.getenv("BOT4_TOKEN", "")
if not boss:
    print("  [FAIL] BOT4_TOKEN missing — cannot check the channel")
else:
    for src, chat_id in chats:
        status, body = call(boss, "getChat", chat_id=chat_id)
        if not body.get("ok"):
            print(f"  [FAIL] {chat_id} ({src}) getChat HTTP {status}: {body.get('description')}")
            continue
        chat = body["result"]
        title = chat.get("title") or chat.get("username") or chat.get("type")
        print(f"  [ OK ] {chat_id} ({src}) -> {title} [{chat.get('type')}]")

        status, member = call(boss, "getChatMember", chat_id=chat_id, user_id=bot_ids.get("Bot 4 (Boss)", 0))
        if member.get("ok"):
            m = member["result"]
            st = m.get("status")
            can_post = m.get("can_post_messages", True if st == "creator" else False)
            ok = st in ("administrator", "creator") and can_post
            print(f"         Boss status={st} can_post={can_post} -> "
                  f"{'CAN POST' if ok else 'CANNOT POST'}")
        else:
            print(f"         getChatMember: {member.get('description')}")

# ── 3. Optional real send ───────────────────────────────────────────────────
if "--send" in sys.argv:
    print("\n=== SENDING ONE TEST MESSAGE AS BOSS ===")
    targets = [i for _, i in chats if i < 0]
    # Plain text on purpose: Markdown v1 chokes on the "_" inside "@bossbtc2_bot"
    # ("can't parse entities"), which is exactly why the bot now retries as text.
    text = ("اختبار البوس\n"
            "هذه رسالة تجريبية من @bossbtc2_bot.\n"
            "إذا وصلتك، فالتوكن والقناة سليمان.")
    for chat_id in targets:
        status, body = call(boss, "sendMessage", chat_id=chat_id, text=text)
        print(f"  {chat_id}: HTTP {status} ok={body.get('ok')} {body.get('description', '')}")
else:
    print("\n(read-only run; add --send to post one real test message as Boss)")

print("\nDONE")
