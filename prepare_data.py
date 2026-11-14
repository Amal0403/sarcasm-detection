"""Download three public sarcasm datasets and merge them into one CSV.

Sources (all labelled sarcastic = 1, not sarcastic = 0):
  headlines  News Headlines Dataset for Sarcasm Detection (The Onion vs HuffPost)
  reddit     Self-Annotated Reddit Corpus (SARC), balanced sample of comments
  tweets     SemEval-2018 Task 3 irony detection tweets (TweetEval "irony")

Output: data/sarcasm_dataset.csv with columns text, label, source
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SEED = 42
REDDIT_PER_CLASS = 15000
OUT_FILE = Path(__file__).parent / "data" / "sarcasm_dataset.csv"


def fetch(repo: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def clean(text) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_headlines() -> list[tuple[str, int]]:
    path = fetch("raquiba/Sarcasm_News_Headline", "train.json")
    raw = Path(path).read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        records = json.loads(raw)
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [(clean(r["headline"]), int(r["is_sarcastic"])) for r in records]


def load_reddit() -> list[tuple[str, int]]:
    path = fetch("marcbishara/sarcasm-on-reddit", "data/holdout-00000-of-00001.parquet")
    table = pq.read_table(path, columns=["comment", "label"]).to_pydict()

    by_label: dict[int, list[str]] = {0: [], 1: []}
    seen = set()
    for comment, label in zip(table["comment"], table["label"]):
        if comment is None or label is None:
            continue
        text = clean(comment)
        if text in seen or not 4 <= len(text.split()) <= 60:
            continue
        seen.add(text)
        by_label[int(label)].append(text)

    rng = random.Random(SEED)
    rows = []
    for label, texts in by_label.items():
        for text in rng.sample(texts, min(REDDIT_PER_CLASS, len(texts))):
            rows.append((text, label))
    return rows


def load_tweets() -> list[tuple[str, int]]:
    rows = []
    for split in ("train", "validation", "test"):
        path = fetch("cardiffnlp/tweet_eval", f"irony/{split}-00000-of-00001.parquet")
        table = pq.read_table(path, columns=["text", "label"]).to_pydict()
        rows += [(clean(t), int(l)) for t, l in zip(table["text"], table["label"])]
    return rows


def main() -> None:
    rows, seen = [], set()
    for source, loader in (("headlines", load_headlines), ("reddit", load_reddit), ("tweets", load_tweets)):
        print(f"Downloading {source}...")
        for text, label in loader():
            if text and text not in seen:
                seen.add(text)
                rows.append((text, label, source))

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "source"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_FILE}")
    counts = Counter((source, label) for _, label, source in rows)
    for source in ("headlines", "reddit", "tweets"):
        print(f"  {source:10s} not sarcastic {counts[(source, 0)]:6d}   sarcastic {counts[(source, 1)]:6d}")


if __name__ == "__main__":
    main()
