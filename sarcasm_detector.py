"""Predict whether a sentence is sarcastic.

Interactive:   python sarcasm_detector.py
One-off:       python sarcasm_detector.py "Oh great, another Monday."

Uses the fine-tuned transformer from finetune.py when it exists, and otherwise
falls back to the lighter embedding classifier from train.py.
"""

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"
FINETUNED_DIR = MODEL_DIR / "finetuned"
CLASSIFIER_FILE = MODEL_DIR / "sarcasm_model.joblib"


class SarcasmDetector:
    def __init__(self):
        if (FINETUNED_DIR / "config.json").exists():
            self._load_finetuned()
        elif CLASSIFIER_FILE.exists():
            self._load_classifier()
        else:
            raise SystemExit("No trained model found. Run: python prepare_data.py, then python train.py")

    def _load_finetuned(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
        model.eval()

        def probability(text: str) -> float:
            enc = tokenizer([text], truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                return float(torch.softmax(model(**enc).logits, dim=-1)[0, 1])

        self.name = "fine-tuned MiniLM"
        self._probability = probability

    def _load_classifier(self) -> None:
        import joblib
        from sentence_transformers import SentenceTransformer

        saved = joblib.load(CLASSIFIER_FILE)
        classifier = saved["classifier"]
        encoder = SentenceTransformer(saved["encoder"])

        def probability(text: str) -> float:
            vector = encoder.encode([text], normalize_embeddings=True)
            return float(classifier.predict_proba(vector)[0][1])

        self.name = f"MiniLM embeddings + {saved['name']}"
        self._probability = probability

    def predict(self, text: str) -> tuple[bool, float]:
        """Return (is_sarcastic, probability that the text is sarcastic)."""
        probability = self._probability(text)
        return probability >= 0.5, probability


def describe(text: str, detector: SarcasmDetector) -> str:
    sarcastic, probability = detector.predict(text)
    label = "Sarcastic" if sarcastic else "Not sarcastic"
    confidence = probability if sarcastic else 1 - probability
    return f"{label}  (confidence {confidence:.0%})"


def main() -> None:
    print("Loading model...")
    detector = SarcasmDetector()
    print(f"Using: {detector.name}")

    if len(sys.argv) > 1:
        print(describe(" ".join(sys.argv[1:]), detector))
        return

    print("Type a sentence to check it for sarcasm. Type 'exit' to quit.\n")
    while True:
        try:
            text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        print(describe(text, detector) + "\n")


if __name__ == "__main__":
    main()
