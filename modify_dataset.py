import pandas as pd

# Load current dataset
df = pd.read_csv("data/iris.csv")

# Add version information
df["dataset_version"] = "v2"

# Save as the new dataset version
df.to_csv("data/iris.csv", index=False)

print("Dataset updated to version 2")
print("Rows:", len(df))
print("Columns:", len(df.columns))