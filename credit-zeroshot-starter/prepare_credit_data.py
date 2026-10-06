"""Download public UCI credit datasets and isolate labels from zero-shot inputs.

No estimator is fitted. Splits depend only on row positions and a fixed seed.
Every normal invocation runs the preparation smoke checks before completing.
Run --self-test for small offline checks that require no download.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd


SOURCES = {
    "taiwan": {
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "page": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H",
        "target": "default_next_month",
        "positive": "default",
        "definition": "Whether a credit-card client defaults on payment in the following month; 1 means default and 0 means no default.",
        "population": "Taiwan credit-card clients, with recorded payment and bill histories from April–September 2005",
        "horizon": "the month following the September 2005 predictor records",
    },
    "south_german": {
        "url": "https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip",
        "page": "https://archive.ics.uci.edu/dataset/573/south+german+credit+update",
        "citation": "South German Credit (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5QG88",
        "target": "bad_credit",
        "positive": "bad credit",
        "definition": "Whether the credit contract was not complied with; 1 means bad credit and 0 means good credit. This is not the Taiwan next-month-default endpoint.",
        "population": "Historical South German credit contracts from 1973–1975, with bad credits oversampled in the original benchmark",
        "horizon": "compliance over the recorded credit contract; no common next-month horizon is specified",
    },
}

GERMAN_NAMES = dict(zip(
    ["laufkont", "laufzeit", "moral", "verw", "hoehe", "sparkont", "beszeit", "rate", "famges", "buerge", "wohnzeit", "verm", "alter", "weitkred", "wohn", "bishkred", "beruf", "pers", "telef", "gastarb"],
    ["checking_account_status", "loan_duration_months", "credit_history", "loan_purpose", "credit_amount_transformed", "savings_category", "employment_duration_category", "installment_rate_category", "personal_status_sex", "other_debtors", "residence_duration_category", "property_category", "age_years", "other_installment_plans", "housing", "number_credits_category", "job_category", "people_liable_category", "telephone", "foreign_worker"],
))

TAIWAN_DICTIONARY = """Predictor documentation: UCI Default of Credit Card Clients.
ID is a source record identifier and is removed from prediction inputs.
LIMIT_BAL: granted credit amount in New Taiwan dollars (NTD).
SEX: category code, 1 male, 2 female.
EDUCATION: documented codes 1 graduate school, 2 university, 3 high school, 4 other. Additional codes in the raw file are retained, not silently reclassified.
MARRIAGE: documented codes 1 married, 2 single, 3 other. Additional raw codes are retained.
AGE: age in years.
PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6: repayment-status codes for September, August, July, June, May, April 2005 respectively. The UCI description gives -1 for duly paid and 1–9 for delays of one through nine-or-more months. Raw -2/0 and any other undocumented codes must be reconciled with the chosen study; no meaning is invented here. PAY_0 is deliberately not renamed PAY_1.
BILL_AMT1 through BILL_AMT6: bill statement amounts in NTD for September through April 2005. Negative amounts are retained.
PAY_AMT1 through PAY_AMT6: previous payment amounts in NTD for September through April 2005.
The target column 'default payment next month' is removed completely from predictors and renamed default_next_month in the evaluation-only file. Its coding is 1 default, 0 no default.
Category numbers are labels, not automatically equal-spaced measurements. Check every literature model's category coding, transforms, centering, scales and adjustment set.
Do not use the dataset's introductory paper, or any paper fitting these same UCI rows, as transfer evidence in a claim of independent zero-shot evaluation.
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""


def download(dataset: str, cache: Path) -> bytes:
    """Cache original bytes separately from uploadable files; never execute downloads."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{dataset}-original.zip"
    if not path.exists():
        request = urllib.request.Request(SOURCES[dataset]["url"], headers={"User-Agent": "ZeroShotCreditPreparation/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError("Download exceeds the expected 20 MB safety limit.")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.testzip() is not None:
                raise ValueError("Downloaded archive failed its integrity check.")
        path.write_bytes(data)
    return path.read_bytes()


def decode(dataset: str, data: bytes):
    """Read verified UCI layouts, preserve predictor values and map target polarity."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if dataset == "taiwan":
            raw = pd.read_excel(archive.open("default of credit card clients.xls"), header=1, engine="xlrd")
            if raw.shape != (30000, 25) or "default payment next month" not in raw or "ID" not in raw:
                raise ValueError("Unexpected Taiwan dataset layout; inspect the source before proceeding.")
            x = raw.drop(columns=["ID", "default payment next month"])
            y = raw["default payment next month"].astype(int)
            dictionary = TAIWAN_DICTIONARY
        elif dataset == "south_german":
            raw = pd.read_csv(archive.open("SouthGermanCredit.asc"), sep=r"\s+")
            if raw.shape != (1000, 21) or set(raw.columns) != set(GERMAN_NAMES) | {"kredit"}:
                raise ValueError("Unexpected South German layout; inspect the source before proceeding.")
            # UCI's corrected codetable states 0=bad, 1=good. Reverse to make 1 the adverse event.
            y = 1 - raw["kredit"].astype(int)
            x = raw.drop(columns="kredit").rename(columns=GERMAN_NAMES)
            table = archive.read("codetable.txt").decode("utf-8-sig")
            mapping = "\n".join(f"{old} -> {new}" for old, new in GERMAN_NAMES.items())
            dictionary = (
                "South German Credit corrected predictor coding.\n"
                "Uploaded column names are translated below; numeric predictor values are unchanged.\n"
                "credit_amount_transformed is a monotonically transformed amount; the exact transformation is unknown. Do not treat it as raw currency or convert it to a modern currency.\n"
                "personal_status_sex combines marital status and sex ambiguously; sex cannot be recovered reliably. Category values are not automatically numeric linear slopes.\n"
                "Bad credits were oversampled in the original dataset. Absolute default probabilities on a natural borrower population are therefore not directly validated by this benchmark.\n"
                "The raw kredit target has 0=bad and 1=good. Evaluation uses bad_credit=1-kredit.\n"
                "Original target counts/prevalence are not used to choose a split, a prior or a model.\n\n"
                + mapping + "\n\nOfficial corrected code table:\n" + table
                + "\nSource: " + SOURCES[dataset]["page"] + "\n"
            )
        else:
            raise ValueError("Unknown dataset.")
    if set(y.unique()) != {0, 1} or x.isna().any().any() or x.columns.duplicated().any():
        raise ValueError("Unexpected labels, missing predictors or duplicate columns.")
    y.name = SOURCES[dataset]["target"]
    return x, y, dictionary


def split_positions(n: int, context_rows: int, test_rows: int, seed: int):
    """Intentionally label-blind: this function cannot receive target values."""
    if context_rows < 8 or test_rows < 1 or context_rows + test_rows > n:
        raise ValueError(f"Need at least 8 context rows, 1 test row, and at most {n} total rows.")
    order = np.random.default_rng(seed).permutation(n)
    return order[:context_rows], order[context_rows:context_rows + test_rows]


def assert_prepared(folder: Path):
    """Smoke-test target isolation, row alignment, untouched predictors and hashes."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    x_context = pd.read_csv(folder / "upload/context_features.csv")
    x_test = pd.read_csv(folder / "upload/test_features.csv")
    truth = pd.read_csv(folder / "evaluation_only/truth.csv")
    positions = json.loads((folder / "evaluation_only/split_positions.json").read_text(encoding="utf-8"))
    forbidden = {manifest["target"], "default payment next month", "kredit", "ID", "row_position"}
    assert not set(x_context) & forbidden, "Target/identifier leaked into context features."
    assert not set(x_test) & forbidden, "Target/identifier leaked into test features."
    assert list(x_context) == list(x_test)
    assert len(x_context) == manifest["context_rows"] and len(x_test) == manifest["test_rows"]
    assert list(truth) == ["row_position", manifest["target"]]
    assert truth.row_position.tolist() == list(range(len(x_test)))
    assert set(truth[manifest["target"]].unique()) <= {0, 1}
    assert not set(positions["context_source_positions"]) & set(positions["test_source_positions"])
    for relative, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((folder / relative).read_bytes()).hexdigest() == expected
    print(f"PASS {folder.name}: {len(x_context)} context rows, {len(x_test)} test rows, {x_test.shape[1]} predictors; labels isolated.")


def prepare(dataset: str, data: bytes, root: Path, context_rows: int, test_rows: int, seed: int):
    x, y, dictionary = decode(dataset, data)
    context_positions, test_positions = split_positions(len(x), context_rows, test_rows, seed)
    folder = root / dataset
    if folder.exists():
        raise FileExistsError(f"{folder} already exists. Choose a new --out directory; nothing was overwritten.")
    (folder / "upload").mkdir(parents=True)
    (folder / "evaluation_only").mkdir()
    context = x.iloc[context_positions].reset_index(drop=True)
    test = x.iloc[test_positions].reset_index(drop=True)
    context.to_csv(folder / "upload/context_features.csv", index=False)
    test.to_csv(folder / "upload/test_features.csv", index=False)
    truth = pd.DataFrame({"row_position": np.arange(len(test)), SOURCES[dataset]["target"]: y.iloc[test_positions].to_numpy()})
    truth.to_csv(folder / "evaluation_only/truth.csv", index=False)
    positions = {"context_source_positions": context_positions.tolist(), "test_source_positions": test_positions.tolist()}
    (folder / "evaluation_only/split_positions.json").write_text(json.dumps(positions, indent=2), encoding="utf-8")
    spec = SOURCES[dataset]
    target = {"name": spec["target"], "definition": spec["definition"], "unit": "0/1",
              "population": spec["population"], "time_horizon": spec["horizon"], "family": "logistic", "positive_class": spec["positive"]}
    (folder / "target.json").write_text(json.dumps(target, indent=2), encoding="utf-8")
    (folder / "data_dictionary.txt").write_text(dictionary, encoding="utf-8")
    # Empty starter bundle supplies metadata, not invented evidence or coefficients.
    digest = hashlib.sha256(dictionary.encode()).hexdigest()
    bundle = {"target": target, "models": [], "sources": [{"id": digest[:16], "title": f"{dataset}: predictor definitions, not coefficient evidence",
               "text": dictionary, "url": spec["page"], "kind": "manual", "sha256": digest}]}
    (folder / "starter_evidence.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    files = ["upload/context_features.csv", "upload/test_features.csv", "evaluation_only/truth.csv"]
    manifest = {"dataset": dataset, "source_url": spec["url"], "citation": spec["citation"],
                "license": "CC BY 4.0", "source_archive_sha256": hashlib.sha256(data).hexdigest(),
                "source_rows": len(x), "predictors": len(x.columns), "target": spec["target"],
                "context_rows": context_rows, "test_rows": test_rows, "seed": seed,
                "split_method": "random row positions without label stratification or label-based filtering",
                "file_sha256": {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Validate the saved evaluation mapping against the original labels without reporting outcomes.
    restored = pd.read_csv(folder / "evaluation_only/truth.csv")
    np.testing.assert_array_equal(restored[spec["target"]], y.iloc[test_positions].to_numpy())
    pd.testing.assert_frame_equal(pd.read_csv(folder / "upload/test_features.csv"), test, check_dtype=False)
    assert_prepared(folder)


def self_test():
    """Pure split tests: deterministic, disjoint and independent of label order."""
    a, b = split_positions(1000, 512, 256, 42)
    c, d = split_positions(1000, 512, 256, 42)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)
    assert len(a) == 512 and len(b) == 256 and not set(a) & set(b)
    try:
        split_positions(10, 8, 8, 42)
    except ValueError:
        pass
    else:
        raise AssertionError("Oversized splits were not rejected.")
    print("PASS offline preparation self-test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["taiwan", "south_german", "both"], default="both")
    parser.add_argument("--out", default="credit_data")
    parser.add_argument("--cache", default="credit_originals_DO_NOT_UPLOAD")
    parser.add_argument("--context-rows", type=int, default=512)
    parser.add_argument("--test-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", type=Path, help="Verify an already prepared dataset folder")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if args.verify:
        assert_prepared(args.verify)
        return
    datasets = ["taiwan", "south_german"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        prepare(dataset, download(dataset, Path(args.cache)), Path(args.out), args.context_rows, args.test_rows, args.seed)


if __name__ == "__main__":
    main()
