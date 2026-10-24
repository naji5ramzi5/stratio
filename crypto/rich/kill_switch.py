"""
rich/kill_switch.py — Global Kill Switch for Bot5.

States:
  RUNNING         — normal operation
  PAUSED          — manually paused; no new entries
  LIVE_LOCKED     — read-only mode on live exchange
  EMERGENCY_STOP  — all new entries blocked; positions monitored
  DESYNC          — local ≠ exchange state; new entries blocked
  API_FAILURE     — repeated API failures; new entries blocked
  RISK_LIMIT      — daily/drawdown limit; new entries blocked
  PROTECTION_FAILURE — stop creation failed on open position

Transitions:
  Only ADMIN can manually resume from EMERGENCY_STOP.
  PAUSED can self-resume after reconciliation passes.
  DESYNC requires successful reconciliation to clear.

All state changes are logged to DB.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from .models import SystemState

logger = logging.getLogger("rich.kill_switch")
KillSwitchState = SystemState



class KillSwitch:
    """
    Thread-safe global state machine for Bot5.

    Rules:
    - Only RUNNING state allows new entries.
    - Any non-RUNNING state blocks signal consumption.
    - All transitions are logged.
    """

    # States that block new entries completely
    BLOCKING_STATES = {
        SystemState.EMERGENCY_STOP,
        SystemState.DESYNC,
        SystemState.API_FAILURE,
        SystemState.RISK_LIMIT,
        SystemState.PROTECTION_FAILURE,
        SystemState.LIVE_LOCKED,
    }

    # States that require admin to exit
    ADMIN_ONLY_EXIT = {
        SystemState.EMERGENCY_STOP,
        SystemState.PROTECTION_FAILURE,
    }

    def __init__(self, config=None, db=None):
        if db is None and config is not None and hasattr(config, "db_path") is False:
            self._db = config
        else:
            self._db = db
        self._config = config
        self._state = SystemState.RUNNING
        self._lock = threading.Lock()
        self._state_reason: str = "Normal operation"
        self._changed_at: str = datetime.now(timezone.utc).isoformat()
        self._on_change_callbacks: list[Callable] = []

    def can_trade(self) -> bool:
        """Helper alias for can_accept_signals."""
        return self._state == SystemState.RUNNING

    def status(self) -> dict:
        """Helper alias for to_dict."""
        return self.to_dict()

    def system_error(self, title: str, details: str = "") -> None:
        """Helper to trip kill switch on system error."""
        self.transition(SystemState.EMERGENCY_STOP, f"{title}: {details}", "system_error")


    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def state_reason(self) -> str:
        return self._state_reason

    def is_running(self) -> bool:
        return self._state == SystemState.RUNNING

    def can_accept_signals(self) -> bool:
        """True only when system is fully RUNNING."""
        return self._state == SystemState.RUNNING

    def can_place_orders(self) -> bool:
        """True for RUNNING only. PAUSED and others return False."""
        return self._state == SystemState.RUNNING

    def is_blocked(self) -> bool:
        return self._state in self.BLOCKING_STATES

    def transition(self, new_state: SystemState, reason: str,
                   component: str = "") -> bool:
        """
        Transition to a new state. Returns True if transition occurred.

        Some transitions require admin (not implemented here — admin check
        is at the Telegram command level).
        """
        with self._lock:
            if self._state == new_state:
                return False

            old_state = self._state
            self._state = new_state
            self._state_reason = reason
            self._changed_at = datetime.now(timezone.utc).isoformat()

            level = "WARNING" if new_state != SystemState.RUNNING else "INFO"
            logger.log(
                logging.WARNING if level == "WARNING" else logging.INFO,
                f"[KillSwitch] {old_state.value} → {new_state.value}: {reason}"
            )

            if self._db:
                try:
                    self._db.log_system_event(
                        event_type=f"STATE_TRANSITION_{new_state.value}",
                        message=reason,
                        severity=level,
                        component=component or "kill_switch",
                        state_before=old_state.value,
                        state_after=new_state.value,
                    )
                except Exception as e:
                    logger.error(f"[KillSwitch] Failed to log state change: {e}")

            for cb in self._on_change_callbacks:
                try:
                    cb(old_state, new_state, reason)
                except Exception as e:
                    logger.error(f"[KillSwitch] Callback error: {e}")

            return True

    def emergency_stop(self, reason: str) -> None:
        self.transition(SystemState.EMERGENCY_STOP, reason, "emergency")
        logger.critical(f"[KillSwitch] EMERGENCY STOP: {reason}")

    def desync(self, reason: str) -> None:
        self.transition(SystemState.DESYNC, reason, "reconciliation")

    def api_failure(self, reason: str) -> None:
        self.transition(SystemState.API_FAILURE, reason, "api")

    def risk_limit(self, reason: str) -> None:
        self.transition(SystemState.RISK_LIMIT, reason, "risk_engine")

    def protection_failure(self, reason: str) -> None:
        self.transition(SystemState.PROTECTION_FAILURE, reason, "protection")

    def pause(self, reason: str) -> None:
        self.transition(SystemState.PAUSED, reason, "admin")

    def resume(self, reason: str) -> bool:
        """
        Resume to RUNNING. Must only be called after health checks pass.
        EMERGENCY_STOP and PROTECTION_FAILURE require admin confirmation.
        """
        if self._state in self.ADMIN_ONLY_EXIT:
            logger.warning(
                f"[KillSwitch] Cannot auto-resume from {self._state.value}. "
                "Admin action required."
            )
            return False
        return self.transition(SystemState.RUNNING, reason, "resume")

    def ready(self) -> None:
        """Called after successful startup reconciliation."""
        self.transition(SystemState.RUNNING, "Startup reconciliation complete")

    def register_on_change(self, callback: Callable) -> None:
        """Register a callback invoked on every state transition."""
        self._on_change_callbacks.append(callback)

    def to_dict(self) -> dict:
        return {
            "state": self._state.value,
            "reason": self._state_reason,
            "can_accept_signals": self.can_accept_signals(),
            "is_blocked": self.is_blocked(),
            "changed_at": self._changed_at,
        }
