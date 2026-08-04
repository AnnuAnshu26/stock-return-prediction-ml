"""
Aggregates the saved per-fold prediction files (experiments/results/predictions/*.csv)
and runs the trading simulation from backtest.py for each model, pooling across
all stocks and folds. Run this AFTER re-running train_pipeline.py with the
prediction-saving patch, since the original run's console-only output did not
persist raw predictions.

Usage:
    python -m src.run_backtest
"""
import glob
import os
import pandas as pd
from . import config
from . import backtest


def main():
    pred_dir = os.path.join(config.RESULTS_DIR, 'predictions')
    files = glob.glob(os.path.join(pred_dir, '*.csv'))

    if not files:
        print(f"No prediction files found in {pred_dir}.")
        print("Re-run `python -m src.train_pipeline` with the updated code first —")
        print("the original run did not save raw predictions, only aggregated metrics.")
        return

    all_preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(all_preds)} pooled test-day predictions from {len(files)} fold files.")

    model_cols = [c for c in all_preds.columns if c != 'y_true']
    results_summary = []

    for model_name in model_cols:
        sim = backtest.simulate_long_flat_strategy(
            y_true_returns=all_preds['y_true'].values,
            y_pred_returns=all_preds[model_name].values,
            transaction_cost_bps=5.0,
        )
        perf = backtest.compare_to_buy_hold(sim)
        results_summary.append({
            'Model': model_name,
            'Strategy_Sharpe': perf['strategy']['Sharpe_Ratio'],
            'Strategy_Annualized_Return': perf['strategy']['Annualized_Return'],
            'Strategy_Max_Drawdown': perf['strategy']['Max_Drawdown'],
            'Strategy_Cumulative_Return': perf['strategy']['Cumulative_Return'],
            'BuyHold_Sharpe': perf['buy_and_hold']['Sharpe_Ratio'],
            'BuyHold_Cumulative_Return': perf['buy_and_hold']['Cumulative_Return'],
        })

    summary_df = pd.DataFrame(results_summary).sort_values('Strategy_Sharpe', ascending=False)
    out_path = os.path.join(config.RESULTS_DIR, 'backtest_summary.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved backtest summary to {out_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()