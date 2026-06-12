"""Advanced EDA for the Structural Break Real-Time challenge.

Produces:
  - reports/series_summary.csv   : one row per training series with segment
                                   lengths, break info, and pre/post break
                                   distributional statistics (break-type taxonomy).
  - reports/step_balance.csv     : per online-step alive / positive / negative counts.
  - reports/figs/*.png           : EDA figures.
  - prints a textual summary.

Run:  uv run python scripts/eda.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sb.data import DATA_DIR, PERIOD_HISTORICAL, PERIOD_ONLINE

REPORTS = "reports"
FIGS = os.path.join(REPORTS, "figs")
os.makedirs(FIGS, exist_ok=True)


def autocorr1(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.nan
    return float(np.dot(x[:-1], x[1:]) / denom)


def main() -> None:
    print("Loading X_train + index ...")
    x = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"), columns=["value", "period"])
    index = pd.read_parquet(os.path.join(DATA_DIR, "y_train_index.parquet"))
    tau_map = index["tau_index"].to_dict()

    rows = []
    # per online-step accumulators
    max_T = 1000
    alive = np.zeros(max_T + 1, dtype=np.int64)
    pos = np.zeros(max_T + 1, dtype=np.int64)

    print("Iterating series ...")
    for sid, sub in x.groupby(level="id", sort=True):
        vals = sub["value"].to_numpy()
        period = sub["period"].to_numpy()
        xh = vals[period == PERIOD_HISTORICAL]
        xo = vals[period == PERIOD_ONLINE]
        ti = tau_map.get(sid, -1)
        ti = None if ti is None or ti < 0 else int(ti)

        n_h, n_o = len(xh), len(xo)
        # per-step balance
        T = min(n_o, max_T)
        alive[:T] += 1
        if ti is not None:
            # positive from step ti onward
            lo = min(ti, T)
            pos[lo:T] += 1

        rec = {
            "id": sid,
            "n_hist": n_h,
            "n_online": n_o,
            "has_break": ti is not None,
            "tau_index": ti if ti is not None else -1,
            "tau_frac": (ti / n_o) if (ti is not None and n_o > 0) else np.nan,
            "hist_mean": xh.mean(),
            "hist_std": xh.std(ddof=1) if n_h > 1 else np.nan,
            "hist_skew": stats.skew(xh) if n_h > 2 else np.nan,
            "hist_kurt": stats.kurtosis(xh) if n_h > 3 else np.nan,
            "hist_acf1": autocorr1(xh),
        }

        if ti is not None and ti >= 2 and (n_o - ti) >= 2:
            pre = xo[:ti]
            post = xo[ti:]
            rec.update({
                "pre_mean": pre.mean(), "post_mean": post.mean(),
                "pre_std": pre.std(ddof=1) if len(pre) > 1 else np.nan,
                "post_std": post.std(ddof=1) if len(post) > 1 else np.nan,
                "pre_acf1": autocorr1(pre), "post_acf1": autocorr1(post),
                "mean_shift": post.mean() - pre.mean(),
                "std_ratio": (post.std(ddof=1) / pre.std(ddof=1))
                if (len(pre) > 1 and pre.std(ddof=1) > 1e-9) else np.nan,
                "ks_stat": stats.ks_2samp(pre, post).statistic,
            })
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("id")
    os.makedirs(REPORTS, exist_ok=True)
    df.to_csv(os.path.join(REPORTS, "series_summary.csv"))

    # step balance frame
    sb_df = pd.DataFrame({
        "t_online": np.arange(max_T + 1),
        "alive": alive,
        "n_pos": pos,
        "n_neg": alive - pos,
    })
    sb_df["weight"] = sb_df["n_pos"] * sb_df["n_neg"]
    sb_df = sb_df[sb_df["alive"] > 0]
    sb_df.to_csv(os.path.join(REPORTS, "step_balance.csv"), index=False)

    # ---------------- textual summary ----------------
    n = len(df)
    nb = int(df["has_break"].sum())
    print("\n" + "=" * 60)
    print(f"Series total              : {n}")
    print(f"With break                : {nb} ({nb / n:.1%})")
    print(f"No break                  : {n - nb} ({(n - nb) / n:.1%})")
    print(f"Historical len  min/med/max: {df.n_hist.min()}/{int(df.n_hist.median())}/{df.n_hist.max()}")
    print(f"Online len      min/med/max: {df.n_online.min()}/{int(df.n_online.median())}/{df.n_online.max()}")
    brk = df[df.has_break]
    print(f"tau_index       min/med/max: {brk.tau_index.min()}/{int(brk.tau_index.median())}/{brk.tau_index.max()}")
    print(f"tau_frac        mean/med    : {brk.tau_frac.mean():.3f}/{brk.tau_frac.median():.3f}")

    # break-type taxonomy among breaks with measurable pre/post
    bt = df[df.has_break & df["mean_shift"].notna()].copy()
    abs_mean = bt["mean_shift"].abs()
    logvar = np.log(bt["std_ratio"].replace(0, np.nan)).abs()
    acf_change = (bt["post_acf1"] - bt["pre_acf1"]).abs()
    mean_break = abs_mean > 0.5
    var_break = logvar > np.log(1.5)
    acf_break = acf_change > 0.2
    print("\nBreak-type taxonomy (|mean shift|>0.5, std ratio>1.5x, |Δacf1|>0.2):")
    print(f"  mean-shift breaks   : {int(mean_break.sum())} ({mean_break.mean():.1%})")
    print(f"  variance breaks     : {int(var_break.sum())} ({var_break.mean():.1%})")
    print(f"  autocorr breaks     : {int(acf_break.sum())} ({acf_break.mean():.1%})")
    pure_subtle = (~mean_break & ~var_break & ~acf_break)
    print(f"  subtle/dist-only    : {int(pure_subtle.sum())} ({pure_subtle.mean():.1%})")
    print(f"  ks_stat med/p90     : {bt.ks_stat.median():.3f}/{bt.ks_stat.quantile(0.9):.3f}")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[0, 0].hist(df.n_hist, bins=50, color="#4f46e5")
    ax[0, 0].set_title("Historical segment length")
    ax[0, 1].hist(df.n_online, bins=50, color="#0891b2")
    ax[0, 1].set_title("Online segment length")
    ax[0, 2].hist(brk.tau_frac.dropna(), bins=40, color="#f59e0b")
    ax[0, 2].set_title("Break position (fraction of online)")
    ax[1, 0].hist(bt.mean_shift.clip(-5, 5), bins=60, color="#16a34a")
    ax[1, 0].set_title("Mean shift (post - pre)")
    ax[1, 1].hist(np.log(bt.std_ratio.replace(0, np.nan)).clip(-3, 3).dropna(), bins=60, color="#dc2626")
    ax[1, 1].set_title("log std ratio (post/pre)")
    ax[1, 2].plot(sb_df.t_online, sb_df.weight, color="#7c3aed")
    ax[1, 2].set_title("TS-AUC weight n_pos*n_neg per step")
    ax[1, 2].set_xlabel("online step")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "eda_overview.png"), dpi=110)
    print(f"\nSaved figures to {FIGS}/eda_overview.png")
    print(f"Saved tables to {REPORTS}/series_summary.csv, step_balance.csv")


if __name__ == "__main__":
    main()
