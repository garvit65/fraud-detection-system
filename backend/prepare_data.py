import pandas as pd

df = pd.read_csv("data/paysim.csv")

# Take smaller sample (fast processing)
df = df.sample(10000, random_state=42)

# Save smaller dataset
df.to_csv("data/paysim_small.csv", index=False)

print("Small dataset created!")