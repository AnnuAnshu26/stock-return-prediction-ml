"""
Minimal long/flat trading backtest driven by model predictions.

This turns "we predicted next-day returns with MAE=0.02" into an
economically interpretable claim: Sharpe ratio, max drawdown, and
cumulative return vs. buy-and-hold. Reviewers of applied finance ML
papers usually want to see this — pure statistical error metrics alone
don't tell you if a model is tradeable.

Strategy (deliberately simple, on purpose — this is a diagnostic, not a
product): go long when predicted next-day return > 0, stay in cash
(flat) otherwise. Transaction costs are modeled as a flat bps charge
per position change; report results with and without costs.
"""
import numpy as np
import pandas as pd


def simulate_long_flat_strategy(
    y_true_returns: np.ndarray,
    y_pred_returns: np.ndarray,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    y_true_returns: actual next-day log returns (what the position would earn)
    y_pred_returns: model's predicted next-day log returns (signal)
    transaction_cost_bps: round-trip cost in basis points applied on position changes

    Returns a DataFrame with columns: position, strategy_return, buy_hold_return
    """
    position = (y_pred_returns > 0).astype(int)  # 1 = long, 0 = flat
    position_change = np.abs(np.diff(position, prepend=0))
    cost = position_change * (transaction_cost_bps / 10000.0)

    strategy_return = position * y_true_returns - cost
    buy_hold_return = y_true_returns

    return pd.DataFrame({
        "position": position,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
    })


def performance_summary(returns: np.ndarray, periods_per_year: int = 252) -> dict:
    """
    Computes Sharpe ratio, annualized return, max drawdown, and cumulative
    return from a series of periodic (daily) log returns.
    """
    returns = np.asarray(returns)
    mean_r = returns.mean()
    std_r = returns.std(ddof=1) if len(returns) > 1 else np.nan

    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r and std_r > 0 else np.nan
    annualized_return = mean_r * periods_per_year

    cumulative = np.cumsum(returns)  # log returns sum -> cumulative log return
    cumulative_wealth = np.exp(cumulative)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "Annualized_Return": annualized_return,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_drawdown,
        "Cumulative_Return": cumulative_wealth[-1] - 1 if len(cumulative_wealth) else np.nan,
    }


def compare_to_buy_hold(sim_df: pd.DataFrame) -> dict:
    strat_perf = performance_summary(sim_df["strategy_return"].values)
    bh_perf = performance_summary(sim_df["buy_hold_return"].values)
    return {"strategy": strat_perf, "buy_and_hold": bh_perf}
