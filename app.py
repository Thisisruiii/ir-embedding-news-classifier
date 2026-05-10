import gradio as gr
import joblib
import json
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

MODEL_REPO = "Thisisruiii/AG-News-Classifier"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

classifier_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="classifier.joblib"
)
clf = joblib.load(classifier_path)

label_map_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="label_map.json"
)

with open(label_map_path, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

embedder = SentenceTransformer(EMBED_MODEL)

def classify(text):
    if not text.strip():
        return {}
    proba = clf.predict_proba(embedder.encode([text]))[0]
    return {label_map[i+1]: float(proba[i]) for i in range(4)}

gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, placeholder="Paste a news headline here..."),
    outputs=gr.Label(num_top_classes=4),
    title="AG News Classifier",
    description="Classify news into: World, Sports, Business, or Sci/Tech",
    examples=[
    ["Global leaders meet to discuss peace talks after regional conflict"],
    ["Election results spark debate among world leaders"],

    ["The football team won the championship after a dramatic final"],
    ["Tennis star advances to the semifinal after straight-set victory"],

    ["Stock markets rise after strong company earnings report"],
    ["Oil prices fall as investors worry about weak demand"],

    ["NASA launches a new telescope to explore deep space"],
    ["New AI model improves performance on coding and reasoning tasks"],
]
).launch()