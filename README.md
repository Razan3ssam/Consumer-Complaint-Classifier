# Consumer Complaint Classifier

Deep learning text classification for consumer complaints. The model reads a written complaint and predicts which product category (and therefore which team) it belongs to, with a confidence score. Built as an end-to-end NLP pipeline, comparing three sequential networks built from scratch with a fine-tuned transformer, and deployed as a Gradio web app.

**Demo video:** [Watch the app in action](assets/Screen%20Recording%202026-11-01%20182914.mp4)

**Links:** [Kaggle notebook](https://www.kaggle.com/code/razanelnabawy/ccd-fornlp) · [Dataset](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp) · [Fine-tuned model (Releases)](../../releases)

## Categories

`credit_card` · `credit_reporting` · `debt_collection` · `mortgages_and_loans` · `retail_banking`

## Dataset

[Consumer Complaints Dataset for NLP](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp) (CFPB data on Kaggle, file `complaints_processed.csv`). 162,421 rows in the original file. The full training code is in the [Kaggle notebook](https://www.kaggle.com/code/razanelnabawy/ccd-fornlp) and in `notebook/ccd-fornlp.ipynb`. The narratives are already lowercased, stripped of punctuation, numbers and stop words, and lemmatized.

## Pipeline

1. Loaded the data, dropped 10 empty rows and 37,939 duplicate narratives, leaving **124,472** complaints (duplicates removed before splitting to avoid train/test leakage).
2. Class distribution after cleaning: credit_reporting 45.2%, debt_collection 16.9%, mortgages_and_loans 15.0%, credit_card 12.0%, retail_banking 10.8%. Imbalance handled with `class_weight` (computed on the training set only).
3. Stratified split: 80% train / 10% validation / 10% test, identical for all four models.
4. RNN family: Keras tokenizer (20,000 words, OOV token), padding/truncation to 200 tokens, Embedding(128) → recurrent layer (64 units) → Dropout → Dense(64) → Softmax, with early stopping.
5. Transformer: `distilbert-base-uncased` fine-tuned with HuggingFace `Trainer` (max length 256, 2 epochs, lr 3e-5, class-weighted loss).
6. Evaluation on the held-out test set: accuracy, macro precision, macro recall, macro F1, confusion matrices.

## Results (test set, 12,448 complaints)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Training time |
|---|---|---|---|---|---|
| SimpleRNN | 0.7953 | 0.7674 | 0.7841 | 0.7731 | 70 s |
| LSTM | 0.8206 | 0.7931 | 0.8398 | 0.8122 | 74 s |
| GRU | 0.8373 | 0.8067 | 0.8508 | 0.8257 | 47 s |
| **DistilBERT** | **0.8527** | **0.8287** | **0.8609** | **0.8433** | 2,397 s |

DistilBERT is the best on every metric, but GRU gets within about 1.5 accuracy points at a fraction of the training cost. Most errors are between `credit_reporting` and `debt_collection`, whose complaints share a lot of vocabulary. Full numbers are in [`results/`](results/).

## Run the app

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

The app opens at http://127.0.0.1:7860. If the confidence is below 60%, it suggests manual review instead of routing automatically.

### Model files

The fine-tuned DistilBERT model (about 250 MB) is too large to store in the repository, so it is published as a release asset.

1. Open the [Releases](../../releases) page and download `distilbert_final.zip` from the latest release.
2. Unzip it into a folder named `model/` next to `app.py`, so that `model/config.json` and `model/model.safetensors` exist directly inside it (not inside a nested folder).
3. Run `python app.py`.

The model was trained in the notebook in `notebook/`.

## Project structure

```
├── app.py                 # Gradio web app
├── requirements.txt
├── notebook/              # Kaggle training notebook
├── results/               # model comparison table and metrics
├── assets/                # demo video
└── model/                 # fine-tuned DistilBERT (download from Releases)
```

## Limitations

- English only; the text cleaning removes any non-English characters.
- Complaints that span several products (for example a collector reporting a debt to the credit bureaus) are inherently ambiguous, and each complaint has a single label.
- The training text had stop words removed, which limits how much context the transformer can use.

## Author

[Razan3ssam](https://github.com/Razan3ssam)
