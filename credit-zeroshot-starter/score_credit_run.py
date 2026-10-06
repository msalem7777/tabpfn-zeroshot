"""Score a completed app run only after checking its identity against the test data.

True labels are read here, after prediction. They never enter the prediction app.
This script requires the installed ZeroShot package supplied previously.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def binary_scores(y, probability):
    """Probability quality and ranking quality, with explicit absent-class handling."""
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if not len(y) or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Expected nonempty 0/1 outcomes.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Predicted probabilities must be finite and between 0 and 1.")
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    positives = int(y.sum())
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        ranks = rankdata(probability, method="average")
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    return {"brier_score": float(np.mean((probability-y)**2)),
            "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            "roc_auc": auc}


def score(run_path, dataset_folder):
    from zeroshot.evaluate import evaluate
    folder = Path(dataset_folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["file_sha256"].items():
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared file changed: {relative}. Recreate the split or use the original files.")
    target = json.loads((folder / "target.json").read_text(encoding="utf-8"))
    test = pd.read_csv(folder / "upload/test_features.csv")
    context = pd.read_csv(folder / "upload/context_features.csv")
    with zipfile.ZipFile(run_path) as archive:
        audit = json.loads(archive.read("audit.json"))
        predictions = pd.read_csv(archive.open("predictions.csv"))
    if audit.get("prediction_mode") != "separate_test_rows":
        raise ValueError("Upload the separate test_features.csv as test rows before scoring this prepared split.")
    for key, frame in [("query_sha256", test), ("context_sha256", context)]:
        digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        if audit.get(key) != digest:
            raise ValueError(f"{key} mismatch: this run used different data, column names, values or row order.")
    if audit["target"]["name"] != manifest["target"] or audit["target"]["family"] != "logistic" or audit["target"]["positive_class"] != target["positive_class"]:
        raise ValueError("Run target/type/positive-class meaning does not match this benchmark.")
    # Reuse the app's strict row matching and original probability scoring.
    report = evaluate(run_path, folder / "evaluation_only/truth.csv", manifest["target"])
    truth = pd.read_csv(folder / "evaluation_only/truth.csv")
    matched = predictions.merge(truth, on="row_position", validate="one_to_one")
    y = matched[manifest["target"]].to_numpy()
    for method in ["direct", "tabpfn"]:
        if method+"_mean" in matched:
            report["scores"][method] = binary_scores(y, matched[method+"_mean"].to_numpy())
    report["scores"]["fixed_0.5_reference"] = binary_scores(y, np.full(len(y), .5))
    report["reference_note"] = "0.5 is a fixed, deliberately uninformative reference. It was not estimated from hidden labels. Beating it alone is not strong evidence of useful prediction."
    report["dataset"] = manifest["dataset"]
    report["source_citation"] = manifest["citation"]
    report["evidence_review_note"] = "Review the run's evidence sources for overlap with this benchmark. Data hashes and target masking cannot establish source-study independence or rule out pretrained-model contamination."
    report["uncertainty_note"] = "These binary probability scores do not establish that uncertainty intervals are calibrated. A single 0/1 label per row cannot directly validate an interval around that row's unknown default probability."
    return report


def self_test():
    y = np.array([0, 0, 1, 1])
    good = binary_scores(y, np.array([.1, .2, .8, .9]))
    tied = binary_scores(y, np.full(4, .5))
    assert good["roc_auc"] == 1 and tied["roc_auc"] == .5
    assert np.isclose(tied["brier_score"], .25)
    assert np.isclose(tied["log_loss"], np.log(2))
    assert good["brier_score"] < tied["brier_score"]
    print("PASS scoring self-test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if not all([args.run, args.dataset, args.out]):
        parser.error("--run, --dataset and --out are required unless using --self-test.")
    if args.out.exists():
        parser.error("Output already exists; choose a new --out filename to preserve previous scores.")
    report = score(args.run, args.dataset)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
