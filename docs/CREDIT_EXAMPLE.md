# Reproduce the masked credit workflow

Run commands from the repository root. This downloads the data afresh; it does not reproduce the old LLM's response or the exact old model proposals.

1. Install and start the app using the README. In another PowerShell window at the same folder, install the preparation dependencies and create the split:

```powershell
.\.venv\Scripts\python.exe -m pip install -r credit-zeroshot-starter/requirements.txt
.\.venv\Scripts\python.exe credit-zeroshot-starter/prepare_credit_data.py --dataset taiwan --out credit_data --cache credit_originals_DO_NOT_UPLOAD --context-rows 512 --test-rows 256 --seed 42
```

2. In the web interface, load `credit_data/taiwan/upload/context_features.csv` as context and `credit_data/taiwan/upload/test_features.csv` as test covariates. Set the binary target to default payment next month, with 1 meaning default, using the definition in `credit_data/taiwan/target.json`. Keep the original data and everything in `evaluation_only` outside the LLM/UI pipeline.

3. Load `credit_data/taiwan/starter_evidence.json` to supply target metadata and the predictor dictionary. It intentionally contains no coefficients or invented paper evidence. Connect your LLM, then use the delegated model-building workflow with your approved sources. Review the source and numerical-provenance labels. The old run's numbers were proposed assumptions, not extracted published fits.

4. Select the literature-generated context → TabPFN path. To resemble the pilot, use 32 worlds, 128 predictive draws per world, seed 42 and one estimator. These settings are a starting experiment, not optimized defaults. Save the full research workspace and export the completed run ZIP. Save the run as `runs/zeroshot-run.zip`; create `runs` if necessary. Keep exports outside Git.

5. Only after freezing predictions, evaluate against the held-out truth:

```powershell
.\.venv\Scripts\python.exe credit-zeroshot-starter/score_credit_run.py --run runs/zeroshot-run.zip --dataset credit_data/taiwan --out runs/credit-evaluation.json
```

The scorer checks input identity and preserves existing evaluation files. Choose another output filename if it already exists. See [the known cross-platform newline issue](EXPERIMENT.md) if hashes disagree; do not remove the check to force a score.

6. Interpret AUC as ranking, Brier/log loss as probability scores, and calibration as agreement between predicted and observed frequencies. A single binary outcome cannot directly validate an interval for that individual's unknown probability. Once results influence model choices, use a new untouched evaluation split or dataset for confirmation.

Dataset: Yeh, I. (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository, [doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H). The preparation manifest records CC BY 4.0, source URL, original archive hash, row positions and file hashes. The script also supports South German Credit; its generated manifest supplies that dataset's attribution. No real dataset is bundled here. Synthetic examples in `examples/synthetic` are demonstration fixtures, not empirical research evidence.
