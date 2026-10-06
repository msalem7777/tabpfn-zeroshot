"""Evaluate an already completed run. Truth is never passed to prediction/retrieval."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def empirical_crps(samples, truth):
    """CRPS: smaller is better; scores the entire predictive distribution.

    CRPS = E|prediction - truth| - 0.5 E|prediction - independent prediction|.
    Sorting computes the second term without an enormous pairwise matrix.
    """
    samples = np.asarray(samples)
    n = len(samples)
    ordered = np.sort(samples, axis=0)
    ranks = (2*np.arange(1,n+1)-n-1)[:,None]
    return np.mean(np.abs(samples-truth),axis=0) - np.sum(ranks*ordered,axis=0)/(n*n)


def evaluate(run_file, truth_file, target_column):
    truth = pd.read_csv(truth_file)
    if "row_position" not in truth or target_column not in truth:
        raise ValueError("Truth CSV must contain row_position and the explicitly selected target column.")
    if truth.row_position.duplicated().any():
        raise ValueError("Truth row_position values must be unique.")
    with zipfile.ZipFile(run_file) as archive:
        predictions = pd.read_csv(archive.open("predictions.csv"))
        audit = json.loads(archive.read("audit.json"))
        with np.load(io.BytesIO(archive.read("predictive_distributions.npz")), allow_pickle=False) as data:
            arrays = {k:data[k] for k in data.files}
    merged = predictions.merge(truth[["row_position",target_column]],on="row_position",validate="one_to_one")
    if len(merged) != len(predictions) or len(truth) != len(predictions):
        raise ValueError("Truth must cover exactly the prediction rows; no silent row dropping.")
    merged = merged.sort_values("row_position")
    y = pd.to_numeric(merged[target_column],errors="raise").to_numpy()
    if not np.isfinite(y).all():
        raise ValueError("Truth contains missing/non-finite values.")
    scores = {}
    for method in ["direct","tabpfn"]:
        if method+"_mean" not in merged:
            continue
        mean = merged[method+"_mean"].to_numpy()
        if audit["target"]["family"] == "logistic":
            if not set(np.unique(y)) <= {0,1}:
                raise ValueError("Binary truth must use 0/1, with 1 matching the reviewed positive class.")
            p = np.clip(mean,1e-12,1-1e-12)
            scores[method] = {"brier_score":float(np.mean((mean-y)**2)),
                              "log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))}
        else:
            samples = arrays[method+"_samples"].reshape(-1,len(y))
            scores[method] = {"rmse":float(np.sqrt(np.mean((mean-y)**2))),
                "mae":float(np.mean(np.abs(mean-y))),
                "interval_coverage":float(np.mean((y>=merged[method+"_lower"])&(y<=merged[method+"_upper"]))),
                "mean_interval_width":float(np.mean(merged[method+"_upper"]-merged[method+"_lower"])),
                "crps":float(np.mean(empirical_crps(samples,y)))}
    return {"rows":len(y),"nominal_interval":audit["config"]["interval"],"scores":scores,
            "note":"Evaluation only. Truth was read after prediction and never supplied to either prediction path."}


def main():
    parser=argparse.ArgumentParser(description="Score an existing run using separately held-out truth")
    parser.add_argument("--run",required=True);parser.add_argument("--truth",required=True)
    parser.add_argument("--target-column",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result=evaluate(args.run,args.truth,args.target_column)
    with Path(args.out).open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
