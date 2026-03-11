"""
MSCI Index Data Pipeline
- Loads raw MSCI price data
- Calculates 5-year annualised returns (CAGR)
- Removes highly correlated indexes (auto + manual override)
- Brute-forces all 5-index combinations and ranks by Sharpe ratio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import math
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_PATH  = Path(r"H:\A01 - Utilità\A11-Cartellona\A11.4-Portafoglio\Cleaned_MSCI_Master_ONLY_IMI.csv")
OUTPUT_PATH = Path(r"H:\A01 - Utilità\A11-Cartellona\A11.4-Portafoglio\MSCI_Final_Calculated.csv")
PORTFOLIO_OUTPUT_PATH = Path(r"H:\A01 - Utilità\A11-Cartellona\A11.4-Portafoglio\MSCI_Best_Portfolios.csv")

TRADING_DAYS_PER_YEAR = 260
RETURN_HORIZON_YEARS  = 5
FFILL_LIMIT           = 5
CORRELATION_THRESHOLD = 0.95

LAG = TRADING_DAYS_PER_YEAR * RETURN_HORIZON_YEARS

PORTFOLIO_SIZE = 5          # number of indexes per portfolio
TOP_N_RESULTS  = 20         # how many top portfolios to display and save
RISK_FREE_RATE = 0.02       # ← set your annual risk-free rate here (e.g. 0.02 = 2%)

# ─────────────────────────────────────────────
# BROAD INDEX KEYWORDS
# ─────────────────────────────────────────────

BROAD_KEYWORDS = [
    "ACWI", "World", "Global", "All Country",
    "Emerging Markets", "Developed Markets",
    "North America", "Europe ", "Pacific", "Asia",
]

# ─────────────────────────────────────────────
# MANUAL OVERRIDES
# Key   = auto-dropped index you want to KEEP
# Value = auto-kept index you want to DROP instead
# ─────────────────────────────────────────────

MANUAL_RESOLVE: dict[str, str] = {
    # "MSCI Italy IMI Index - Ret": "MSCI Italy Index - Ret",
}


# ─────────────────────────────────────────────
# 1. LOAD & PARSE
# ─────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", header=0, usecols=[0])
    raw.columns = ["raw"]
    parsed = raw["raw"].str.split(";", expand=True)
    parsed.columns = ["Date", "Value", "Msci_Name"]
    parsed["Value"] = parsed["Value"].str.replace(",", ".", regex=False).astype(float)
    parsed["Date"]  = pd.to_datetime(parsed["Date"], format="%d-%m-%Y")
    return parsed


# ─────────────────────────────────────────────
# 2. RESHAPE TO WIDE
# ─────────────────────────────────────────────

def reshape_wide(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.drop_duplicates(subset=["Date", "Msci_Name"], keep="last")
    wide = df.pivot(index="Date", columns="Msci_Name", values="Value")
    wide.columns.name = None
    return wide.sort_index().ffill(limit=FFILL_LIMIT)


# ─────────────────────────────────────────────
# 3. CALCULATE ANNUALISED RETURNS (CAGR)
# ─────────────────────────────────────────────

def calculate_annualised_returns(prices: pd.DataFrame, lag: int, years: int) -> pd.DataFrame:
    returns = (prices / prices.shift(lag)) ** (1 / years) - 1
    return returns.add_suffix(" - Ret")


# ─────────────────────────────────────────────
# 4. CORRELATION FILTERING — keep highest Sharpe
# ─────────────────────────────────────────────

def compute_standalone_sharpe(returns: pd.DataFrame, risk_free_rate: float) -> pd.Series:
    """
    Computes the Sharpe ratio for each index individually.
    Uses the same logic as the portfolio engine for consistency.
    """
    mean = returns.mean()
    std  = returns.std()
    return (mean - risk_free_rate) / std


def resolve_correlations(
    returns: pd.DataFrame,
    threshold: float,
    risk_free_rate: float,
    manual_resolve: dict[str, str]
) -> tuple[pd.DataFrame, list[dict]]:
    """
    For each correlated pair, drops the index with the lower standalone Sharpe.
    Manual overrides take priority over the auto decision.
    """
    sharpe = compute_standalone_sharpe(returns, risk_free_rate)

    corr  = returns.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Index_A", "level_1": "Index_B", 0: "Corr"})
        .query("Corr >= @threshold")
        .sort_values("Corr", ascending=False)
    )

    to_drop = set()
    decision_log = []

    for _, row in pairs.iterrows():
        a, b, corr_val = row["Index_A"], row["Index_B"], row["Corr"]

        if a in to_drop or b in to_drop:
            continue

        # Keep the one with the higher Sharpe, drop the lower one
        sharpe_a, sharpe_b = sharpe.get(a, np.nan), sharpe.get(b, np.nan)
        kept  = a if sharpe_a >= sharpe_b else b
        niche = b if kept == a else a

        # Check for manual override
        if niche in manual_resolve and manual_resolve[niche] == kept:
            niche, kept = kept, niche
            source = "manual override"
        else:
            source = f"auto (Sharpe: {sharpe[kept]:.3f} vs {sharpe[niche]:.3f})"

        to_drop.add(niche)
        decision_log.append({
            "kept":    kept,
            "dropped": niche,
            "corr":    corr_val,
            "source":  source
        })

    return returns.drop(columns=list(to_drop), errors="ignore"), decision_log


# ─────────────────────────────────────────────
# 5. PORTFOLIO SHARPE — BRUTE FORCE
# ─────────────────────────────────────────────

def compute_portfolio_sharpe(
    returns: pd.DataFrame,
    portfolio_size: int,
    risk_free_rate: float,
    top_n: int,
    chunk_size: int = 50_000  # process 50k combinations at a time
) -> pd.DataFrame:

    clean  = returns.dropna()
    cols   = clean.columns.tolist()
    n      = len(cols)
    data   = clean.values.astype(np.float32)  # float32 cuts memory in half vs float64

    total_combinations = math.comb(n, portfolio_size)
    print(f"  Testing {total_combinations:,} combinations of {portfolio_size} indexes...")

    index_names = np.array(cols)
    all_combos  = list(combinations(range(n), portfolio_size))

    best_records = []

    for start in range(0, total_combinations, chunk_size):
        chunk = np.array(all_combos[start : start + chunk_size])  # shape: (chunk, 5)

        # shape: (T, chunk, 5) → mean over axis 2 → (T, chunk) → T (chunk,)
        port_ret = data[:, chunk].mean(axis=2)                     # shape: (T, chunk)

        mean_ret = port_ret.mean(axis=0)                           # shape: (chunk,)
        std_ret  = port_ret.std(axis=0)                            # shape: (chunk,)
        sharpe   = np.where(std_ret > 0, (mean_ret - risk_free_rate) / std_ret, np.nan)

        # Only keep top_n from this chunk to avoid accumulating millions of rows
        top_idx = np.argsort(sharpe)[::-1][:top_n]

        for i in top_idx:
            best_records.append({
                "indexes":     " | ".join(index_names[chunk[i]]).replace(" - Ret", ""),
                "sharpe":      round(float(sharpe[i]), 4),
                "mean_return": round(float(mean_ret[i]), 4),
                "volatility":  round(float(std_ret[i]), 4),
            })

        # Progress every 500k combinations
        if (start // chunk_size) % (500_000 // chunk_size) == 0:
            pct = start / total_combinations * 100
            print(f"  {start:>9,} / {total_combinations:,}  ({pct:.1f}%)")

    results = (
        pd.DataFrame(best_records)
        .sort_values("sharpe", ascending=False)
        .drop_duplicates(subset="indexes")
        .reset_index(drop=True)
    )
    results.index += 1

    return results.head(top_n)


# ─────────────────────────────────────────────
# 6. REPORTS
# ─────────────────────────────────────────────

def print_decision_report(log: list[dict], n_before: int, n_after: int) -> None:
    print("=" * 75)
    print(f"  CORRELATION RESOLUTION  (threshold ≥ {CORRELATION_THRESHOLD:.0%})")
    print("=" * 75)
    print(f"  Original indexes : {n_before}")
    print(f"  Dropped          : {n_before - n_after}")
    print(f"  Remaining        : {n_after}\n")

    if log:
        print(f"  {'KEPT':<50} {'DROPPED':<50} {'CORR':>6}  SOURCE")
        print(f"  {'-'*50} {'-'*50} {'-'*6}  {'-'*15}")
        for d in log:
            print(f"  {d['kept']:<50} {d['dropped']:<50} {d['corr']:>6.4f}  {d['source']}")
    print()

def print_portfolio_report(top_portfolios: pd.DataFrame) -> None:
    print("=" * 75)
    print(f"  TOP {len(top_portfolios)} PORTFOLIOS BY SHARPE RATIO  "
          f"(risk-free rate = {RISK_FREE_RATE:.1%})")
    print("=" * 75)
    print(f"  {'RANK':<6} {'SHARPE':>7}  {'MEAN RET':>9}  {'VOLATILITY':>11}  INDEXES")
    print(f"  {'-'*6} {'-'*7}  {'-'*9}  {'-'*11}  {'-'*40}")
    for rank, row in top_portfolios.iterrows():
        print(f"  {rank:<6} {row['sharpe']:>7.4f}  {row['mean_return']:>9.4f}  "
              f"{row['volatility']:>11.4f}  {row['indexes']}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    print(f"Loading data from: {INPUT_PATH}\n")
    long_df = load_data(INPUT_PATH)
    prices  = reshape_wide(long_df)
    returns = calculate_annualised_returns(prices, lag=LAG, years=RETURN_HORIZON_YEARS)
    
    # Step 1 — correlation filtering
    n_before = returns.shape[1]
    returns_filtered, log = resolve_correlations(returns, CORRELATION_THRESHOLD, RISK_FREE_RATE, MANUAL_RESOLVE)
    n_after = returns_filtered.shape[1]
    print_decision_report(log, n_before, n_after)

    # Save filtered returns
    returns_filtered.to_csv(OUTPUT_PATH, sep=",", index=True)
    print(f"Filtered returns saved to: {OUTPUT_PATH}\n")

    # Step 2 — brute force portfolio optimisation
    top_portfolios = compute_portfolio_sharpe(
        returns_filtered,
        portfolio_size=PORTFOLIO_SIZE,
        risk_free_rate=RISK_FREE_RATE,
        top_n=TOP_N_RESULTS
    )
    print_portfolio_report(top_portfolios)

    top_portfolios.to_csv(PORTFOLIO_OUTPUT_PATH, sep=",", index=True)
    print(f"Top portfolios saved to: {PORTFOLIO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

### What the portfolio engine does

#For each combination of 5 indexes it computes an **equal-weight portfolio return series** across all dates, then derives the Sharpe ratio from that series:
#```
#portfolio_return(t) = average of the 5 CAGR values at date t

#Sharpe = (mean(portfolio_return) - risk_free_rate) / std(portfolio_return)
#```#
#
#The output ranks every combination, for example:
#```
#RANK   SHARPE   MEAN RET   VOLATILITY   INDEXES
#1      1.8432     0.1124       0.0608   MSCI USA | MSCI EM | MSCI Japan | ...
#2      1.7901     0.1089       0.0607   MSCI USA | MSCI EM | MSCI India | ...