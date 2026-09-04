"""
End-to-end walk-forward run for the sb3 PPO path (crypto only).

Fetches 2 years of 1h klines for BTC/ETH/SOL, runs rolling train/out-of-sample
predict, and prints the OOS report vs the equal-weight benchmark.

Run: python run_walkforward.py [--quick]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from walkforward import walkforward_backtest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small train_steps / n_train for a fast check")
    args = ap.parse_args()

    t0 = time.time()
    outdir = os.path.join("crypto", "ppo_models", "walkforward")
    kw = dict(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="1h",
        total=17520,
        n_train=8760,
        n_test=1450,
        outdir=outdir,
        verbose=1,
    )
    if args.quick:
        kw.update(n_train=4000, n_test=1000, train_steps=5000)
    else:
        kw["train_steps"] = 30000

    rep = walkforward_backtest(**kw)

    print("\n" + "=" * 70)
    print("WALK-FORWARD OOS REPORT (strictly out-of-sample)")
    print("=" * 70)
    print(f"symbols={rep['symbols']}  window={rep['windows']}")
    for name in ("metrics_raw", "metrics_gated", "metrics_eqw"):
        m = rep[name]
        print(f"{name:16s} cum={m['cum_return']:+.3f} "
              f"sharpe={m['sharpe']:+.2f} mdd={m['max_drawdown']:.2f}")
    print(f"excess_raw  (vs eqw) = {rep['excess_raw']:+.3f}")
    print(f"excess_gated(vs eqw) = {rep['excess_gated']:+.3f}")
    for f in rep["folds"]:
        print(f"fold {f['i']}: {f['test_range'][0]}..{f['test_range'][1]} "
              f"raw={f['metrics_raw']['cum_return']:+.3f} "
              f"gated={f['metrics_gated']['cum_return']:+.3f}")
    print(f"artifacts in {outdir}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
