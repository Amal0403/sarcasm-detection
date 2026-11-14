"""Train and evaluate the sarcasm classifier.

Sentences are turned into 384-dimensional vectors with a Sentence Transformer,
then several classifiers are compared on a validation split. The best one is
scored once on a held-out test split and saved for sarcasm_detector.py.

Run prepare_data.py first.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

SEED = 42
ENCODER_NAME = "all-MiniLM-L6-v2"

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "sarcasm_dataset.csv"
EMBED_FILE = ROOT / "data" / "embeddings.npy"
MODEL_DIR = ROOT / "model"
MODEL_FILE = MODEL_DIR / "sarcasm_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"


def embed(texts: list[str]) -> np.ndarray:
    """Encode sentences, reusing a cached file when the dataset is unchanged."""
    if EMBED_FILE.exists():
        cached = np.load(EMBED_FILE)
        if len(cached) == len(texts):
            print(f"Using cached embeddings from {EMBED_FILE.name}")
            return cached
    print(f"Encoding {len(texts)} sentences with {ENCODER_NAME} (a few minutes on CPU)...")
    encoder = SentenceTransformer(ENCODER_NAME)
    vectors = encoder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMBED_FILE, vectors)
    return vectors


def candidates() -> dict:
    return {
        "knn_k5": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "knn_k25": KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
        "logistic_regression": LogisticRegression(C=4.0, max_iter=2000),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 64),
            early_stopping=True,
            max_iter=200,
            random_state=SEED,
        ),
    }


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("data/sarcasm_dataset.csv not found. Run: python prepare_data.py")

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = embed([r["text"] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    sources = np.array([r["source"] for r in rows])
    strata = np.array([f"{r['source']}_{r['label']}" for r in rows])

    # 80% train, 10% validation, 10% test, balanced across source and label
    idx = np.arange(len(rows))
    train_idx, rest_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, random_state=SEED, stratify=strata[rest_idx]
    )
    print(f"\nTrain {len(train_idx)}  validation {len(val_idx)}  test {len(test_idx)}")

    print("\nComparing classifiers on the validation split:")
    results, fitted = {}, {}
    for name, clf in candidates().items():
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[val_idx])
        results[name] = {
            "val_accuracy": round(float(accuracy_score(y[val_idx], pred)), 4),
            "val_f1": round(float(f1_score(y[val_idx], pred)), 4),
        }
        fitted[name] = clf
        print(f"  {name:22s} accuracy {results[name]['val_accuracy']:.4f}  F1 {results[name]['val_f1']:.4f}")

    best = max(results, key=lambda n: results[n]["val_accuracy"])
    clf = fitted[best]
    print(f"\nBest on validation: {best}")

    test_pred = clf.predict(X[test_idx])
    test_sources = sources[test_idx]
    per_source = {}
    for source in sorted(set(test_sources)):
        mask = test_sources == source
        per_source[source] = {
            "rows": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y[test_idx][mask], test_pred[mask])), 4),
        }

    metrics = {
        "encoder": ENCODER_NAME,
        "best_model": best,
        "rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "validation": results,
        "test_accuracy": round(float(accuracy_score(y[test_idx], test_pred)), 4),
        "test_f1": round(float(f1_score(y[test_idx], test_pred)), 4),
        "test_accuracy_by_source": per_source,
    }

    print("\nHeld-out test results:")
    print(f"  overall accuracy {metrics['test_accuracy']:.4f}  F1 {metrics['test_f1']:.4f}")
    for source, m in per_source.items():
        print(f"  {source:10s} accuracy {m['accuracy']:.4f}  ({m['rows']} rows)")
    print("\n" + classification_report(y[test_idx], test_pred, target_names=["not sarcastic", "sarcastic"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"classifier": clf, "encoder": ENCODER_NAME, "name": best}, MODEL_FILE, compress=3)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()
