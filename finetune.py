"""Fine-tune the MiniLM transformer end to end for sarcasm classification.

train.py keeps the language model frozen and only trains a classifier on its
sentence embeddings. Here every layer is updated, which lets the model learn
sarcasm-specific cues. Uses the same train / validation / test split as
train.py so the two approaches can be compared directly.

Runs on CPU in roughly 40 to 60 minutes. Run prepare_data.py first.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 42
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "sarcasm_dataset.csv"
MODEL_DIR = ROOT / "model" / "finetuned"
METRICS_FILE = ROOT / "model" / "finetuned_metrics.json"


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def predict(model, tokenizer, texts) -> np.ndarray:
    """Return the probability of sarcasm for each text."""
    model.eval()
    order = np.argsort([len(t) for t in texts])  # similar lengths together = less padding
    probs = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad():
        for chunk in batches(order, 128):
            enc = tokenizer([texts[i] for i in chunk], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            probs[chunk] = torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy()
    return probs


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("data/sarcasm_dataset.csv not found. Run: python prepare_data.py")

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r["text"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    sources = np.array([r["source"] for r in rows])
    strata = np.array([f"{r['source']}_{r['label']}" for r in rows])

    idx = np.arange(len(rows))
    train_idx, rest_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strata)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=SEED, stratify=strata[rest_idx])
    print(f"Train {len(train_idx)}  validation {len(val_idx)}  test {len(test_idx)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    steps_per_epoch = int(np.ceil(len(train_idx) / BATCH_SIZE))
    total_steps = steps_per_epoch * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    val_texts = [texts[i] for i in val_idx]
    best_val, history, start = 0.0, [], time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(train_idx)
        running = 0.0
        for step, chunk in enumerate(batches(order, BATCH_SIZE), start=1):
            enc = tokenizer([texts[i] for i in chunk], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            loss = model(**enc, labels=torch.tensor(y[chunk])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 200 == 0:
                minutes = (time.time() - start) / 60
                print(f"epoch {epoch} step {step}/{steps_per_epoch}  loss {running / 200:.4f}  {minutes:.1f} min", flush=True)
                running = 0.0

        val_pred = (predict(model, tokenizer, val_texts) >= 0.5).astype(int)
        val_acc = float(accuracy_score(y[val_idx], val_pred))
        history.append({"epoch": epoch, "val_accuracy": round(val_acc, 4)})
        print(f"epoch {epoch} done  validation accuracy {val_acc:.4f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"  saved best model so far to {MODEL_DIR}", flush=True)

    # Score the best checkpoint once on the held-out test split
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    test_pred = (predict(model, tokenizer, [texts[i] for i in test_idx]) >= 0.5).astype(int)
    test_sources = sources[test_idx]
    per_source = {}
    for source in sorted(set(test_sources)):
        mask = test_sources == source
        per_source[source] = {
            "rows": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y[test_idx][mask], test_pred[mask])), 4),
        }

    metrics = {
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "rows": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "validation_history": history,
        "test_accuracy": round(float(accuracy_score(y[test_idx], test_pred)), 4),
        "test_f1": round(float(f1_score(y[test_idx], test_pred)), 4),
        "test_accuracy_by_source": per_source,
        "training_minutes": round((time.time() - start) / 60, 1),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))

    print("\nHeld-out test results:")
    print(f"  overall accuracy {metrics['test_accuracy']:.4f}  F1 {metrics['test_f1']:.4f}")
    for source, m in per_source.items():
        print(f"  {source:10s} accuracy {m['accuracy']:.4f}  ({m['rows']} rows)")
    print("\n" + classification_report(y[test_idx], test_pred, target_names=["not sarcastic", "sarcastic"]))
    print(f"Saved metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()
