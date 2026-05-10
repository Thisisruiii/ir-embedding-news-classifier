from pathlib import Path
import json
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
#folder containing the processed datasets
DATA_DIR = ROOT / "processed"
#folder where the trained classifier and label map will be saved
MODEL_DIR = ROOT / "models"

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
#sentence embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    #create the model output folder if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True) 

    #load data
    train_df = pd.read_csv(DATA_DIR / "train_small.csv")
    test_df = pd.read_csv(DATA_DIR / "test_small.csv")
    
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    #extract embeddings
    print("Loading embedding model")
    embedder = SentenceTransformer(EMBED_MODEL)
    
    print("Encoding training texts")
    X_train = embedder.encode(
        train_texts,
        batch_size=32,
        show_progress_bar=True
    )

    print("Encoding test texts...")
    X_test = embedder.encode(
        test_texts,
        batch_size=32,
        show_progress_bar=True
    )

    #training classifier
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_labels)

    #evaluate
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    print(f"Macro F1: {f1_score(test_labels, y_pred, average='macro'):.4f}")

    print(
        classification_report(
            test_labels,
            y_pred,
            labels=[1, 2, 3, 4],
            target_names=list(LABEL_MAP.values())
        )
    )

    #saving model and labeling map
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")

    with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()