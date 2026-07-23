import pandas as pd
import requests

response = requests.get("http://localhost:8000/logs", timeout=10)
response.raise_for_status()
logs = response.json()

if not logs:
    raise SystemExit("No prediction logs found. Run simulator.py first.")

df = pd.DataFrame(logs)
df = df[df['true_label'].notnull()]

df["correct"] = df["prediction"] == df["true_label"]
summary = df.groupby("model").agg(
    requests=("model", "size"),
    accuracy=("correct", "mean"),
)

print(summary)
