import pandas as pd

df = pd.read_csv("experiments/results/walk_forward_results.csv")
print(f"Stocks in results file: {sorted(df['Stock'].unique())}")
print(f"Total rows: {len(df)}")