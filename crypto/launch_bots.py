"""Single launcher for the StratoCrypto bot fleet.

Previously three bots (advanced_bot, dashboard_server, kimi_bot) were started as
ad-hoc background processes. When all three ran, they each loaded the tracker
into memory on startup and wrote it back later — silently corrupting each other
(prediction_accuracy.json lost its verified data).

This launcher starts them in one process-tree with:
  * ordered startup (advanced_bot first, then news, then dashboard),
  * a single stdout/stderr log each,
  * graceful shutdown on Ctrl-C (SIGINT propagates to children),
  * a watchdog that restarts a child if it crashes (optional, off by default).

Run:   python launch_bots.py
Stop:  Ctrl-C
"""
import os
import signal
import subprocess
import sys
import time
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LAUNCHER] %(message)s",
)
log = logging.getLogger("launcher")

# (script, restart_on_crash)
CHILDREN = [
    ("advanced_bot.py", True),
    ("news_bot.py", True),
    ("dashboard_server.py", True),
    ("price_comparator.py", True),
]

running = True
children = []  # (proc, script, restart)


def make_env():
    env = os.environ.copy()
    return env


def start_child(script):
    log_path = os.path.join(LOG_DIR, script.replace(".py", ".log"))
    # rotate if >5MB (keep last 2000 lines)
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > 5*1024*1024:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-2000:]
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
    except Exception:
        pass
    log_file = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=BASE_DIR,
        env=make_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    proc._log_file = log_file
    log.info(f"started {script} (pid {proc.pid})")
    return proc


def stop_child(proc, script):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    except Exception:
        pass
    finally:
        try:
            proc._log_file.close()
        except Exception:
            pass
    log.info(f"stopped {script} (pid {proc.pid})")


def handle_signal(signum, frame):
    global running
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_signal)

    # Start in order
    for script, restart in CHILDREN:
        if not running:
            break
        proc = start_child(script)
        children.append((proc, script, restart))
        time.sleep(2)  # stagger: let advanced_bot initialise the tracker first

    log.info("all bots started. Ctrl-C to stop.")

    # Watchdog loop
    check_interval = 10
    while running:
        for i, (proc, script, restart) in enumerate(list(children)):
            ret = proc.poll()
            if ret is not None:
                # exited
                log.warning(f"{script} exited with code {ret}")
                try:
                    proc._log_file.close()
                except Exception:
                    pass
                if not running:
                    continue
                if restart:
                    log.info(f"restarting {script} ...")
                    new_proc = start_child(script)
                    children[i] = (new_proc, script, restart)
                else:
                    children.pop(i)
        if not children:
            log.info("all children gone — exiting")
            break
        time.sleep(check_interval)

    # Shutdown
    log.info("shutting down ...")
    for proc, script, _ in list(children):
        stop_child(proc, script)
    children.clear()
    log.info("all bots stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
