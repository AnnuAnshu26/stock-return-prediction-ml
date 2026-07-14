# Multi-Asset Return Prediction: A Walk-Forward Comparative Study

## What this project does

Predicts next-day **log returns** (not raw prices — raw price regression
artificially inflates R² via trend and is not taken seriously in the finance-ML
literature) for a 20-stock universe, comparing 8 model families under
**walk-forward validation**, with statistical significance testing and a
trading backtest.

## Architecture

```
src/
├── config.py          # all tunable parameters — start here
├── data_pipeline.py    # yfinance fetch + local parquet cache
├── features.py         # technical indicators + market context features
├── walk_forward.py      # rolling-origin fold generator
├── models/
│   └── model_zoo.py    # Naive, LR, RF, XGBoost, LSTM, GRU, Transformer, Ensemble
├── evaluation.py        # metrics + Diebold-Mariano significance test
├── backtest.py          # long/flat trading simulation, Sharpe, drawdown
└── train_pipeline.py    # orchestrates everything, saves results CSVs
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

**Always smoke-test first** — this catches bugs before you burn hours on the
full run:

```bash
python -m src.train_pipeline --quick
```

This runs 2 stocks with reduced epochs. Check `experiments/results/` for
output. Once it runs clean end-to-end:

```bash
python -m src.train_pipeline
```

Expect this to take a while — 20 stocks × ~10-15 walk-forward folds each ×
8 models per fold, with 2 of those models being trained neural networks.
Run it overnight or on a machine you're not actively using, not in the
middle of your last hour before a deadline.

## What to check after a run

1. `experiments/results/walk_forward_results.csv` — every fold × model × stock row
2. `experiments/results/macro_summary.csv` — averaged across everything, this is your headline table
3. Console output during the run shows per-fold Diebold-Mariano p-values — save these, they go in your significance table

## Known design decisions worth knowing before you touch the code

- **Expanding window by default** (`walk_forward.generate_folds`): training
  data grows each fold rather than staying a fixed size. There's a
  `rolling_window_folds` alternative in the same file for a fixed-size
  variant — running both and comparing is a good robustness check /
  ablation for the paper.
- **Stacked ensemble meta-learner** currently trains on in-sample base-model
  predictions rather than proper out-of-fold CV predictions. This is a
  simplification — flagged in the code comments. If a reviewer would push
  back on this, add k-fold CV within the training window to generate
  honest out-of-fold meta-features before submission.
- **Naive baseline** is deliberately pulled from unscaled data — a common
  bug is accidentally comparing it against scaled features, which produces
  a nonsensical, inflated error.

## Suggested experiment order (maps to the paper's Results section)

1. Full walk-forward run → macro summary table (Table 1 in paper)
2. Per-stock breakdown → appendix table
3. Diebold-Mariano matrix on pooled fold predictions → significance table (Table 2)
4. Backtest simulation using best model's predictions vs buy-and-hold → Figure (equity curve) + Table 3 (Sharpe/drawdown)
5. Ablation: expanding vs rolling window; with vs without market context features

## Collaboration workflow (2 people)

- Use feature branches (`feature/xyz`), PRs for anything touching `src/`
- Never commit `data/cache/*.parquet` or trained model weights — add to `.gitignore`
- Keep `experiments/results/*.csv` in git (small, and this is your reproducibility record)
- Suggested split: Person A owns `models/`, `evaluation.py`, `backtest.py`;
  Person B owns paper writing, related-work, and turning results CSVs into
  figures/tables. Both review the final Results section together against
  the actual numbers before submission.

## Extending further (optional, if you have time before the deadline)

- Sentiment features from news/Twitter via an NLP pipeline (adds novelty, adds significant scope)
- Multi-asset joint modeling (predict all 20 stocks jointly with a shared
  network + asset embeddings) instead of one model per stock
- Regime-conditioning: split results by VIX-defined high/low volatility
  regimes to show when models do/don't work — this is often a compelling
  extra figure
