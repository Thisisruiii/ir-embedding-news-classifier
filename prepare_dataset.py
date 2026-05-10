from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT 

label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

def prepare_split(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)

    # Combine the title and description into one text field.
    df["text"] = df["Title"].astype(str) + ". " + df["Description"].astype(str)

    # Keep the numeric label and also add a readable label name.
    df["label"] = df["Class Index"].astype(int)
    df["label_name"] = df["label"].map(label_map)

    # Keep only the columns needed for training and explanation.
    df = df[["text", "label", "label_name"]]

    # Save processed data.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())

def main():
    prepare_split(DATA_DIR / "train.csv", DATA_DIR / "processed" / "train.csv")
    prepare_split(DATA_DIR / "test.csv", DATA_DIR / "processed" / "test.csv")

if __name__ == "__main__":
    main()