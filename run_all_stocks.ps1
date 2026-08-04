# run_all_stocks.ps1
#
# Runs the walk-forward pipeline one stock at a time, each in its OWN
# python process. This is the fix for the "bad allocation" crashes that
# happen partway into a single long-running "python -m src.train_pipeline"
# process: TensorFlow/XGBoost memory does not fully release between folds
# even with explicit cleanup, so a long-lived process slowly accumulates
# memory until an allocation fails. Starting a fresh OS process per stock
# guarantees Windows fully reclaims all memory before the next stock starts.
#
# Safe to re-run if interrupted: each stock checks the existing results CSV
# and skips itself if already completed (same resume logic as before), so
# this script can just be re-run from the top after any crash.
#
# Usage (from the project root, with venv activated):
#   .\run_all_stocks.ps1

$stocks = @(
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "WMT", "JNJ",
    "BAC", "NFLX", "AMD", "INTC", "KO",
    "PEP", "DIS", "IBM", "ORCL", "PFE"
)

$total = $stocks.Count
$i = 0

foreach ($stock in $stocks) {
    $i++
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "[$i/$total] Starting fresh process for $stock"
    Write-Host "=================================================="

    python -m src.train_pipeline --stock $stock

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$stock] Process exited with an error (code $LASTEXITCODE)."
        Write-Host "Continuing to the next stock anyway. Re-run this script afterward"
        Write-Host "to retry any stock that did not make it into the results CSV."
    }
}

Write-Host ""
Write-Host "All stocks attempted. Checking final results..."
python check_results.py