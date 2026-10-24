"""
rich/secret_redactor.py — Secrets masking for logs.

Ensures API keys, secrets, tokens, and signatures never appear
in logs, exceptions, or Telegram messages.
"""
from __future__ import annotations

import os
import re
from typing import Any

# Build the pattern lazily from env vars so the redactor doesn't
# import secrets at module level.
_SENSITIVE_KEYS = {
    "BINANCE_TRADING_API_KEY", "BINANCE_TRADING_API_SECRET",
    "BINANCE_MONITOR_API_KEY", "BINANCE_MONITOR_API_SECRET",
    "TELEGRAM_RICH_BOT_TOKEN", "TELEGRAM_ADMIN_CHAT_ID",
    "BINANCE_API_KEY", "BINANCE_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
}

# Literal values to redact (populated on first call)
_REDACT_LITERALS: list[str] = []
_REDACT_INITIALIZED = False

REDACTED_PLACEHOLDER = "***REDACTED***"


def register_secret(secret: str) -> None:
    """Explicitly register a secret string to be redacted."""
    if secret and len(secret) > 3 and secret not in _REDACT_LITERALS:
        _REDACT_LITERALS.append(secret)


def _init_redact() -> None:
    global _REDACT_INITIALIZED
    if _REDACT_INITIALIZED:
        return
    for key in _SENSITIVE_KEYS:
        val = os.environ.get(key, "")
        if val and len(val) > 4:
            register_secret(val)
    _REDACT_INITIALIZED = True



def redact(text: str) -> str:
    """Return text with all known secrets replaced by REDACTED_PLACEHOLDER."""
    _init_redact()
    if not isinstance(text, str):
        text = str(text)
    for secret in _REDACT_LITERALS:
        text = text.replace(secret, REDACTED_PLACEHOLDER)
    # Also catch common patterns: Bearer tokens, signatures, long hex strings
    text = re.sub(r"signature=[0-9a-fA-F]{32,}", "signature=***REDACTED***", text)
    text = re.sub(r"(X-MBX-APIKEY:\s*)[^\s,]+", r"\1***REDACTED***", text)
    return text


def redact_dict(d: dict) -> dict:
    """Return a copy of d with sensitive values replaced."""
    out = {}
    for k, v in d.items():
        if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "SIGN", "AUTH", "PASSWORD")):
            out[k] = REDACTED_PLACEHOLDER
        elif isinstance(v, str):
            out[k] = redact(v)
        elif isinstance(v, dict):
            out[k] = redact_dict(v)
        else:
            out[k] = v
    return out


def safe_repr(obj: Any) -> str:
    """repr() with secrets redacted."""
    return redact(repr(obj))


class SecretRedactor:
    """Class wrapper for secret redaction operations."""

    @staticmethod
    def redact_text(text: str) -> str:
        return redact(text)

    @staticmethod
    def redact_dict(d: dict) -> dict:
        return redact_dict(d)

    @staticmethod
    def safe_repr(obj: Any) -> str:
        return safe_repr(obj)


class RedactingFormatter:
    """Wrap a logging formatter to redact secrets from all log records."""

    def __init__(self, base_fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"):
        import logging
        self._fmt = logging.Formatter(base_fmt)

    def format(self, record) -> str:
        msg = self._fmt.format(record)
        return redact(msg)

