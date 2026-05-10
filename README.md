# AG News Topic Classifier Using Sentence Embeddings

This repository contains an embedding-based text classification project for an Information Retrieval assignment. The goal is to classify short news texts into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

The system uses sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2` and a Logistic Regression classifier. A working demo is deployed on Hugging Face Spaces.

## Links

- **Hugging Face Dataset:** https://huggingface.co/datasets/Thisisruiii/ag-news-topic-classification-processed
- **Hugging Face Model:** https://huggingface.co/Thisisruiii/AG-News-Classifier
- **Hugging Face Demo:** https://huggingface.co/spaces/Thisisruiii/AG-News-demo

## Dataset

The original data source is the AG News Classification Dataset from Kaggle.

For this project, I created a processed version by:

1. Combining `Title` and `Description` into one `text` column.
2. Mapping numeric class labels to readable topic names.
3. Creating a balanced subset for training and testing.

The processed dataset contains:

| Split | Examples |
|---|---:|
| Train | 4000 |
| Test | 1000 |

Label mapping:

| Label | Class |
|---:|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

## Method

The classification pipeline is:

news text → sentence embedding → Logistic Regression → topic label
The embedding model is sentence-transformers/all-MiniLM-L6-v2.

## Results

The model was evaluated on a balanced test set of 1000 examples.

Accuracy: 0.8830
Macro F1: 0.8834

## Repository Structure
models/classifier.joblib: trained Logistic Regression classifier
models/label_map.json: label mapping
processed/train_small.csv: processed training subset
processed/test_small.csv: processed test subset
prepare_dataset.py: preprocessing script
make_subset.py: balanced subset creation script
train_model.py: model training and evaluation script
app.py: Gradio demo application
requirements.txt: required Python packages

## How to Run

Install dependencies:

`pip install -r requirements.txt`

Train the classifier:

`python train_model.py`

Run the Gradio demo locally:

`python app.py`

## Demo Usage

Open the Hugging Face Space and enter a short news headline or description. The model will predict topic from four categories: World, Sports, Business, or Sci/Tech.

Example input:

NASA launches new telescope to explore deep space

Expected output:

Sci/Tech

## Tools

This project uses Python, pandas, sentence-transformers, scikit-learn, joblib, Gradio, and Hugging Face Hub.
