import pandas as pd

# Load the parquet file
df = pd.read_parquet("data/raw/yellow_tripdata_2024-02.parquet")

# Sample a single row
sample_row = df.sample(n=1)

# Save as a JSON array (list of dicts, ready for FastAPI)
sample_row.to_json("data/processed/sample_input.json", orient="records", lines=False, indent=2)
