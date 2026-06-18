#!/usr/bin/env python3
"""
Train a logistic-regression router classifier from code router data.

Route keys are relative file paths (e.g. "src/utils/helpers.py") produced by
gen_code_router_data.py. This is the code-unit analog of train_router.py, which
uses "vendor/api/name" as route keys for API endpoints.

Usage:
    python scripts/repo/train_code_router.py --data-dir data/repos/<name>/router
    python scripts/repo/train_code_router.py --data-dir data/repos/<name>/router \
        --out data/repos/<name>/router/code_router_classifier.joblib
"""

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Load a router JSONL file. Return (questions, route_keys) as parallel lists."""
    questions, routes = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            questions.append(rec["question"])
            routes.append(rec["route_key"])
    return questions, routes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing router_train.jsonl and router_test.jsonl "
             "(produced by gen_code_router_data.py).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for classifier joblib file "
             "(default: <data-dir>/code_router_classifier.joblib).",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=4.0,
        help="LogisticRegression regularisation strength C (default: 4.0).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "router_train.jsonl"
    test_path = data_dir / "router_test.jsonl"
    out_path = Path(args.out) if args.out else data_dir / "code_router_classifier.joblib"

    for p in (train_path, test_path):
        if not p.exists():
            sys.exit(f"Missing: {p}\nRun gen_code_router_data.py first.")

    # Heavy ML deps are imported here (inside main) so that importing this
    # module at the top level — e.g. in tests — never fails even when the
    # optional packages are absent. This mirrors train_router.py exactly.
    try:
        import joblib
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        sys.exit(
            f"Missing dependency: {e}\n"
            "Run: pip install sentence-transformers scikit-learn joblib"
        )

    print("Loading training data...")
    train_q, train_r = load_jsonl(train_path)
    test_q, test_r = load_jsonl(test_path)
    print(f"  Train: {len(train_q)} records, Test: {len(test_q)} records")

    route_counts: dict[str, int] = {}
    for r in train_r:
        route_counts[r] = route_counts.get(r, 0) + 1
    print(f"  Routes (train): {len(route_counts)}")

    print("\nEmbedding questions...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    X_train = model.encode(train_q, normalize_embeddings=True, show_progress_bar=True)
    X_test = model.encode(test_q, normalize_embeddings=True, show_progress_bar=True)

    le = LabelEncoder()
    y_train = le.fit_transform(train_r)
    unseen = set(test_r) - set(le.classes_)
    if unseen:
        sys.exit(
            f"Test set has route_key labels not present in training data: {unseen}\n"
            "Increase records per file or re-seed. "
            "(With very few records per file, global shuffle may place a file "
            "entirely in the test split — see gen_code_router_data.py docstring.)"
        )
    y_test = le.transform(test_r)

    print(f"\nTraining LogisticRegression (C={args.C})...")
    clf = LogisticRegression(max_iter=1000, C=args.C)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    overall_acc = np.mean(y_pred == y_test)
    print(f"\nOverall accuracy: {overall_acc:.4f} ({overall_acc * 100:.1f}%)")

    if overall_acc < 0.95:
        print("WARNING: accuracy below 95% target — consider increasing C or more training data")

    print("\nPer-route accuracy:")
    for label_idx, route in enumerate(le.classes_):
        mask = y_test == label_idx
        if mask.sum() == 0:
            continue
        acc = np.mean(y_pred[mask] == label_idx)
        n = mask.sum()
        flag = "  <<" if acc < 0.90 else ""
        print(f"  {route:<60} {acc:.3f}  (n={n}){flag}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "label_encoder": le}, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
