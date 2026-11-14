# Sarcasm Detection

Type a sentence and the model tells you whether it is sarcastic, with a confidence score.

```
>> Wow, I just love being stuck in traffic for two hours.
Sarcastic  (confidence 97%)

>> The meeting is moved to 3 pm tomorrow.
Not sarcastic  (confidence 98%)
```

The project compares two approaches built on the `all-MiniLM-L6-v2` Sentence Transformer:

1. **Frozen embeddings + classifier.** Each sentence becomes a 384-dimensional vector, and a classifier (KNN, logistic regression or a small neural network) is trained on those vectors.
2. **Fine-tuned transformer.** Every layer of the language model is trained on the sarcasm data.

## Results

Trained on 50,483 sentences and scored on a held-out test set of 6,311 sentences that the models never saw.

| Model | Test accuracy | F1 |
|---|---|---|
| Embeddings + KNN (k=5) | 64.3% (validation) | 0.635 |
| Embeddings + KNN (k=25) | 66.3% (validation) | 0.655 |
| Embeddings + logistic regression | 66.0% (validation) | 0.646 |
| Embeddings + neural network | 70.6% | 0.687 |
| **Fine-tuned MiniLM** | **81.3%** | **0.805** |

Accuracy of the two saved models by type of text:

| Source | Test rows | Embeddings + neural network | Fine-tuned MiniLM |
|---|---|---|---|
| News headlines | 2,851 | 79.3% | 91.8% |
| Reddit comments | 3,000 | 63.0% | 73.2% |
| Tweets | 460 | 66.1% | 69.4% |

Fine-tuning adds about 11 percentage points overall. KNN, the starting point of this project, is the weakest option: nearest-neighbour voting on general-purpose embeddings does not separate sarcasm well.

## Data

`prepare_data.py` downloads and merges three public datasets, 63,104 sentences in total, each labelled sarcastic or not:

| Source | Rows | Description |
|---|---|---|
| News headlines | 28,503 | The Onion (sarcastic) and HuffPost (not sarcastic) |
| Reddit comments | 30,000 | Balanced sample from the Self-Annotated Reddit Corpus (SARC) |
| Tweets | 4,601 | SemEval-2018 irony detection task (TweetEval) |

The data is split 80% train, 10% validation and 10% test, balanced across source and label. The classifier is chosen on the validation split and the test split is used once, for the final score.

## Usage

Requires Python 3.10 or newer.

```
pip install -r requirements.txt

python prepare_data.py     # download and merge the datasets (about 1 minute)
python train.py            # embeddings + classifier (about 10 minutes on CPU)
python finetune.py         # optional: fine-tuned transformer (about 1 hour on CPU)

python sarcasm_detector.py                         # interactive
python sarcasm_detector.py "Oh great, more rain."  # single sentence
```

`sarcasm_detector.py` uses the fine-tuned model when it exists and otherwise falls back to the embedding classifier. The embedding classifier is included in this repository (`model/sarcasm_model.joblib`), so prediction works straight after installing the requirements. The fine-tuned model is about 87 MB and is not committed; run `finetune.py` to build it.

## Project structure

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads the three datasets and writes `data/sarcasm_dataset.csv` |
| `train.py` | Encodes sentences, compares four classifiers, saves the best |
| `finetune.py` | Fine-tunes MiniLM end to end with PyTorch |
| `sarcasm_detector.py` | Interactive and command-line prediction |
| `model/metrics.json`, `model/finetuned_metrics.json` | Full evaluation results |

## Limitations

- **Sarcasm often depends on context.** The model sees one sentence with no tone of voice, speaker or conversation, so short conversational text is the hardest case. Reddit accuracy is 73%.
- **Sincere enthusiasm can be misread.** Sarcasm frequently uses exaggerated positive wording, so a sincere sentence such as "I love spending time with my family on weekends" is wrongly flagged as sarcastic.
- **Headline accuracy is partly style.** The headline data comes from two publications, so some of the 92% reflects learning their writing styles as well as sarcasm itself.
- **English only.**
