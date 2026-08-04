"""
Main experiment orchestration.

Run with:
    python -m src.train_pipeline --quick     # 2-3 stocks, few epochs, smoke test
    python -m src.train_pipeline             # full run, all stocks, all folds

This script:
  1. Fetches + caches data for the stock universe + market context
  2. Engineers features per stock
  3. Generates walk-forward folds
  4. Fits all 8 model families per fold
  5. Computes metrics + DM significance tests per fold
  6. Runs the trading backtest on the ensemble's predictions
  7. Saves everything to experiments/results/ for the paper
"""
import argparse
import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data_pipeline import fetch_ohlcv, fetch_market_context
from .features import engineer_features
from .walk_forward import generate_folds
from .models import model_zoo
from . import evaluation
from . import backtest


def structure_sequences(feature_arr, target_arr, time_steps):
    X, y = [], []
    for i in range(time_steps, len(feature_arr)):
        X.append(feature_arr[i - time_steps:i])
        y.append(target_arr[i])
    return np.array(X), np.array(y)


def run_fold(processed_df: pd.DataFrame, fold, symbol: str, epochs_scale: float = 1.0):
    """
    Fits every model on one walk-forward fold and returns per-model metrics
    plus raw predictions (needed later for DM tests and backtesting).
    """
    feature_cols = config.FEATURE_COLUMNS
    X_all = processed_df[feature_cols].values
    y_all = processed_df[config.TARGET_COLUMN].values

    X_train_raw = X_all[fold.train_start:fold.train_end]
    y_train_raw = y_all[fold.train_start:fold.train_end]

    # Pull in `ts` days of context *before* the test window purely so we
    # can build lookback sequences for every test-window day. Without this,
    # the first `ts` days of any test window are consumed just forming the
    # first sequence, and if TEST_WINDOW_DAYS is close to TIME_STEPS
    # (e.g. 63 vs 60) almost the entire test window silently disappears.
    # This context is only ever transformed with the scaler already fit on
    # the training set — no leakage, since the scaler isn't refit here.
    context_start = max(fold.test_start - config.TIME_STEPS, 0)
    X_test_context_raw = X_all[context_start:fold.test_end]
    y_test_context_raw = y_all[context_start:fold.test_end]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_context_scaled = scaler.transform(X_test_context_raw)

    ts = config.TIME_STEPS
    X_train_3d, y_train_seq = structure_sequences(X_train_scaled, y_train_raw, ts)
    X_test_3d, y_test_seq = structure_sequences(X_test_context_scaled, y_test_context_raw, ts)

    if len(X_train_3d) < 50 or len(X_test_3d) < 5:
        return None  # not enough data in this fold, skip

    X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

    predictions = {}

    # 1. Naive persistence — predicts tomorrow's return equals today's
    # actual (unscaled) log return. y_test_seq[k] is the Target at raw row
    # (test_start + k); "today's known return" for that row is simply the
    # Log_Return value at that same row.
    log_return_col_idx = feature_cols.index('Log_Return')
    predictions['Naive'] = X_test_context_raw[ts: ts + len(y_test_seq), log_return_col_idx]

    # 2. Linear Regression
    lr = model_zoo.fit_linear_regression(X_train_2d, y_train_seq)
    predictions['Ridge_Regression'] = lr.predict(X_test_2d)

    # 3. Random Forest
    rf = model_zoo.fit_random_forest(X_train_2d, y_train_seq)
    predictions['Random_Forest'] = rf.predict(X_test_2d)

    # 4. XGBoost
    xgb = model_zoo.fit_xgboost(X_train_2d, y_train_seq)
    predictions['XGBoost'] = xgb.predict(X_test_2d)

    # 5. LSTM
    lstm = model_zoo.fit_sequence_model(
        model_zoo.build_lstm, X_train_3d, y_train_seq,
        epochs=max(1, int(config.LSTM_EPOCHS * epochs_scale))
    )
    predictions['LSTM'] = lstm.predict(X_test_3d, verbose=0).flatten()

    # 6. GRU
    gru = model_zoo.fit_sequence_model(
        model_zoo.build_gru, X_train_3d, y_train_seq,
        epochs=max(1, int(config.GRU_EPOCHS * epochs_scale))
    )
    predictions['GRU'] = gru.predict(X_test_3d, verbose=0).flatten()

    # 7. Transformer
    transformer = model_zoo.fit_sequence_model(
        model_zoo.build_transformer, X_train_3d, y_train_seq,
        epochs=max(1, int(config.LSTM_EPOCHS * epochs_scale))
    )
    predictions['Transformer'] = transformer.predict(X_test_3d, verbose=0).flatten()

    # 8. Stacked ensemble over the 6 learned models (excludes Naive)
    #    Meta-training uses in-sample train predictions as a pragmatic
    #    approximation; swap in out-of-fold CV predictions for a stricter setup.
    base_train_preds = {
        'lr': lr.predict(X_train_2d),
        'rf': rf.predict(X_train_2d),
        'xgb': xgb.predict(X_train_2d),
        'lstm': lstm.predict(X_train_3d, verbose=0).flatten(),
        'gru': gru.predict(X_train_3d, verbose=0).flatten(),
        'transformer': transformer.predict(X_train_3d, verbose=0).flatten(),
    }
    base_test_preds = {
        'lr': predictions['Ridge_Regression'],
        'rf': predictions['Random_Forest'],
        'xgb': predictions['XGBoost'],
        'lstm': predictions['LSTM'],
        'gru': predictions['GRU'],
        'transformer': predictions['Transformer'],
    }
    ensemble_pred, _ = model_zoo.fit_stacked_ensemble(base_train_preds, y_train_seq, base_test_preds)
    predictions['Stacked_Ensemble'] = ensemble_pred

    # --- metrics per model ---
    fold_metrics = []
    for name, pred in predictions.items():
        m = evaluation.compute_metrics(y_test_seq, pred)
        m['Model'] = name
        m['Fold'] = fold.fold_id
        fold_metrics.append(m)

    result = {
        'metrics': fold_metrics,
        'predictions': predictions,
        'y_test': y_test_seq,
    }

    # Persist raw predictions + actuals for this fold. The aggregated
    # metrics above are enough for the Results tables, but the trading
    # backtest (backtest.py) and any later DM significance re-analysis
    # need the actual day-by-day predicted vs. realized returns, not
    # just summary statistics — those can't be reconstructed from MAE/R2
    # after the fact.
    os.makedirs(os.path.join(config.RESULTS_DIR, 'predictions'), exist_ok=True)
    pred_df = pd.DataFrame({name: pred for name, pred in predictions.items()})
    pred_df['y_true'] = y_test_seq
    pred_df.to_csv(
        os.path.join(config.RESULTS_DIR, 'predictions', f'{symbol}_fold{fold.fold_id}.csv'),
        index=False
    )

    # Release the LSTM/GRU/Transformer models explicitly and clear Keras's
    # backend graph state. Without this, building a fresh Sequential/Model
    # every fold (up to ~380 times across a full 20-stock x 19-fold run)
    # steadily accumulates TensorFlow graph state until the process runs
    # out of memory — which is what caused the XGBoost bad_malloc errors
    # partway through a full run (XGBoost just happened to be the model
    # that hit the wall first, the leak itself was upstream).
    del lstm, gru, transformer
    tf.keras.backend.clear_session()
    gc.collect()

    return result


def run_stock(symbol: str, market_context, epochs_scale: float = 1.0):
    print(f"=== Processing {symbol} ===")
    raw = fetch_ohlcv(symbol)
    processed = engineer_features(raw, market_context)

    if len(processed) < config.MIN_TOTAL_DAYS:
        print(f"  Skipping {symbol}: insufficient history ({len(processed)} rows)")
        return []

    folds = generate_folds(len(processed))
    print(f"  {len(folds)} walk-forward folds generated")

    all_rows = []
    for fold in folds:
        try:
            result = run_fold(processed, fold, symbol, epochs_scale=epochs_scale)
        except Exception as e:
            # A single fold failing (e.g. transient OOM) should not discard
            # every fold already completed for this stock. Log and move on.
            print(f"  Fold {fold.fold_id}: FAILED ({e}) — skipping this fold")
            tf.keras.backend.clear_session()
            gc.collect()
            continue

        if result is None:
            continue
        for row in result['metrics']:
            row['Stock'] = symbol
            all_rows.append(row)

        # DM test: XGBoost vs LSTM vs Ensemble as the headline comparison
        preds = result['predictions']
        dm_stat, p_val = evaluation.diebold_mariano(
            result['y_test'], preds['XGBoost'], preds['Stacked_Ensemble']
        )
        print(f"  Fold {fold.fold_id}: DM(XGBoost vs Ensemble) p={p_val:.4f}" if not np.isnan(p_val) else
              f"  Fold {fold.fold_id}: DM test undefined (degenerate variance)")

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smoke test: 2 stocks, fewer epochs")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing results and start over")
    parser.add_argument("--stock", type=str, default=None,
                         help="Process only this one stock symbol, then exit. "
                              "Used by run_all_stocks.ps1 to isolate each stock in its own "
                              "OS process so Windows fully reclaims memory between stocks, "
                              "instead of one long-lived process accumulating memory over "
                              "~15+ stocks until XGBoost/TensorFlow hit a bad_alloc.")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "walk_forward_results.csv")

    if args.stock:
        stocks = [args.stock]
    elif args.quick:
        stocks = config.STOCKS[:2]
    else:
        stocks = config.STOCKS
    epochs_scale = 0.2 if args.quick else 1.0

    # --- Resume support: skip stocks already fully recorded from a prior
    # (possibly crashed) run, so a failure late in a 20-stock run doesn't
    # force you to redo everything from scratch. ---
    existing_df = None
    completed_stocks = set()
    if os.path.exists(out_path) and not args.fresh:
        existing_df = pd.read_csv(out_path)
        completed_stocks = set(existing_df["Stock"].unique())
        if completed_stocks:
            print(f"Resuming: found {len(completed_stocks)} already-completed stocks in {out_path}")
            print(f"  ({sorted(completed_stocks)})")

    print("Fetching market context (S&P 500, VIX)...")
    market_context = fetch_market_context()

    all_results = [] if existing_df is None else existing_df.to_dict("records")

    for symbol in stocks:
        if symbol in completed_stocks:
            print(f"=== Skipping {symbol} (already completed, use --fresh to redo) ===")
            continue
        try:
            rows = run_stock(symbol, market_context, epochs_scale=epochs_scale)
            all_results.extend(rows)
        except Exception as e:
            print(f"[ERROR] {symbol} failed: {e}")

        # Save after every stock, not just at the end — so a crash on
        # stock N doesn't cost you the results from stocks 1..N-1.
        pd.DataFrame(all_results).to_csv(out_path, index=False)

    results_df = pd.DataFrame(all_results)
    print(f"\nSaved {len(results_df)} rows to {out_path}")

    if not results_df.empty:
        macro = results_df.groupby("Model")[
            ["MSE", "RMSE", "MAE", "R2_Score", "Directional_Accuracy"]
        ].mean().reset_index().sort_values("MAE")
        print("\n=== MACRO SUMMARY (mean across all stocks & folds) ===")
        print(macro.to_string(index=False))
        macro.to_csv(os.path.join(config.RESULTS_DIR, "macro_summary.csv"), index=False)


if __name__ == "__main__":
    main()