from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"

def balanced_sample(df, n_per_class, seed=42):
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), n_per_class), random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

train_small = balanced_sample(train_df, n_per_class=1000)
test_small = balanced_sample(test_df, n_per_class=250)

train_small.to_csv(PROCESSED_DIR / "train_small.csv", index=False)
test_small.to_csv(PROCESSED_DIR / "test_small.csv", index=False)

print("Saved train_small:", train_small.shape)
print(train_small["label_name"].value_counts())

print("Saved test_small:", test_small.shape)
print(test_small["label_name"].value_counts())