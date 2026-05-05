#!/usr/bin/env python3
"""
Train a logistic-regression router classifier from router_train.jsonl.

Usage:
    python scripts/train_router.py --data-dir data/<vendor>/router
    python scripts/train_router.py --data-dir data/<vendor>/router --out data/<vendor>/router/router_classifier.joblib
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    questions, routes = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            questions.append(rec["question"])
            routes.append(rec["route_key"])
    return questions, routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing router_train.jsonl and router_test.jsonl")
    parser.add_argument("--out", default=None, help="Output path for classifier (default: <data-dir>/router_classifier.joblib)")
    parser.add_argument(
        "--C", type=float, default=4.0, help="LogisticRegression C (default: 4.0)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "router_train.jsonl"
    test_path = data_dir / "router_test.jsonl"
    out_path = Path(args.out) if args.out else data_dir / "router_classifier.joblib"

    for p in (train_path, test_path):
        if not p.exists():
            sys.exit(f"Missing: {p}\nRun gen_router_data.py first.")

    try:
        import joblib
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install sentence-transformers scikit-learn joblib")

    print("Loading training data...")
    train_q, train_r = load_jsonl(train_path)
    test_q, test_r = load_jsonl(test_path)
    print(f"  Train: {len(train_q)} records, Test: {len(test_q)} records")

    route_counts: dict[str, int] = {}
    for r in train_r:
        route_counts[r] = route_counts.get(r, 0) + 1
    print(f"  Routes: {len(route_counts)}")

    print("\nEmbedding questions...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    X_train = model.encode(train_q, normalize_embeddings=True, show_progress_bar=True)
    X_test = model.encode(test_q, normalize_embeddings=True, show_progress_bar=True)

    le = LabelEncoder()
    y_train = le.fit_transform(train_r)
    y_test = le.transform(test_r)

    print(f"\nTraining LogisticRegression (C={args.C})...")
    clf = LogisticRegression(max_iter=1000, C=args.C)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    overall_acc = np.mean(y_pred == y_test)
    print(f"\nOverall accuracy: {overall_acc:.4f} ({overall_acc * 100:.1f}%)")

    if overall_acc < 0.95:
        print("WARNING: accuracy below 95% target — consider increasing C or using SVM")

    print("\nPer-route accuracy:")
    for label_idx, route in enumerate(le.classes_):
        mask = y_test == label_idx
        if mask.sum() == 0:
            continue
        acc = np.mean(y_pred[mask] == label_idx)
        n = mask.sum()
        flag = "  <<" if acc < 0.90 else ""
        print(f"  {route:<40} {acc:.3f}  (n={n}){flag}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "label_encoder": le}, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
