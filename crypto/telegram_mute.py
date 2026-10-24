"""Local kill-switch for ALL Telegram traffic (user request: verify locally first).

Create/delete `telegram_mute.flag` next to this file to mute/unmute. Every
send path checks suppress() at call time, so no restart is needed to toggle;
muted messages are logged with a short head so we still see what WOULD have
gone out."""
import os

FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telegram_mute.flag")


def is_muted() -> bool:
    try:
        return os.path.exists(FLAG)
    except Exception:
        return False


def suppress(logger, where: str, message: str = "") -> bool:
    """Return True (and log) when mute is active — caller must skip the send."""
    if is_muted():
        try:
            head = (message or "").replace("\n", " ")[:70]
            logger.info(f"[MUTE:{where}] telegram suppressed: {head}")
        except Exception:
            pass
        return True
    return False
