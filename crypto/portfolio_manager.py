"""Smart portfolio manager for the pairs edge.
Uses conservative Kelly sizing (25%), caps each pair at 5%,
and targets 12-20 diversified positions."""
import json, numpy as np

def load_opportunities(path="pairs_db.json"):
    with open(path) as f:
        return json.load(f)["results"]

def kelly_size(avg_ret, var_ret, fraction=0.25):
    """Quarter-Kelly for safety."""
    if var_ret <= 0 or avg_ret <= 0:
        return 0.02  # minimum 2%
    kelly = avg_ret / var_ret
    return max(0.02, min(kelly * fraction, 0.05))  # cap at 5%

def build_portfolio(opps, capital=10000, max_total_exposure=0.6):
    scored = []
    for o in opps:
        edge_score = o["win_rate"] * np.log1p(o["sharpe"]) * np.log1p(o["n_trades"])
        scored.append((edge_score, o))
    scored.sort(key=lambda x: -x[0])

    portfolio = []
    used = 0.0
    for score, opp in scored:
        avg = opp["avg_ret_pct"] / 100.0
        var = abs(avg) * (1 - opp["win_rate"]) * 2 + 1e-6
        size = kelly_size(avg, var)
        size = min(size, 0.05, max_total_exposure - used)
        if size < 0.01 or used >= max_total_exposure:
            break
        portfolio.append({
            "pair": opp["pair"], "tf": opp["tf"],
            "size_pct": round(size * 100, 2), "usd": round(capital * size, 2),
            "sharpe": opp["sharpe"], "win_rate": opp["win_rate"],
            "n_trades": opp["n_trades"], "edge_score": round(score, 3),
        })
        used += size
    return portfolio, used

def main():
    opps = load_opportunities()
    print(f"Opportunities: {len(opps)}\n")
    portfolio, total_exposure = build_portfolio(opps)

    print(f"{'pair':24s} {'tf':4s} {'size%':>6} {'usd':>8} {'sharpe':>8} {'win%':>6} {'trades':>6}")
    print("-" * 75)
    total_usd = 0
    for p in portfolio:
        print(f"{p['pair']:24s} {p['tf']:4s} {p['size_pct']:5.1f}% ${p['usd']:>7.2f} {p['sharpe']:8.2f} {p['win_rate']:5.0%} {p['n_trades']:6d}")
        total_usd += p["usd"]

    print(f"\nTotal positions: {len(portfolio)}")
    print(f"Total exposure: {total_exposure:.0%} = ${total_usd:.2f}")
    print(f"Cash reserve: {(1 - total_exposure):0%} = ${10000 - total_usd:.2f}")

    with open("portfolio.json", "w") as f:
        json.dump({"capital": 10000, "positions": portfolio,
                    "total_exposure": total_exposure, "cash": 1 - total_exposure},
                   f, indent=2)
    print("saved portfolio.json")

if __name__ == "__main__":
    main()
