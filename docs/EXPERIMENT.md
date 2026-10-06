# Exploratory credit experiment

This reports the user's completed v0.4.0 run, audited against the prepared masked Taiwan credit dataset. It is not a fresh experiment performed while packaging this repository. Raw inputs, papers and run archives are not included in this publication copy.

## Setup

- Dataset: UCI Default of Credit Card Clients; original 30,000 rows, 23 predictors. ID and target excluded from input features.
- Split: 512 context rows and 256 disjoint test rows, random row positions, seed 42, without stratification or label-based filtering.
- Inference: 32 worlds, 128 predictive draws per world, one TabPFN estimator, automatic device, seed 42. No context/query rows skipped.
- Three proposed logistic scenarios: minimal repayment history (PAY_0, LIMIT_BAL, AGE, SEX); fuller repayment history (PAY_0 through the available PAY_6 history columns, LIMIT_BAL, AGE); bill/payment amounts (BILL_AMT1, PAY_AMT1, LIMIT_BAL, PAY_0). Selected in 5, 15 and 12 worlds respectively.
- All three sets of coefficients were LLM-proposed numerical assumptions. A predictor dictionary provided qualitative grounding; the run did not recover a complete eligible published fitted equation. Five candidate papers did not establish eligible study evidence. This is chiefly an experiment with LLM-elicited priors.
- Recorded runtime: TabPFN 9.0.0, torch 2.11.0+cu128, numpy 2.5.0, pandas 2.3.3, scipy 1.18.0. Broad installation ranges do not recreate this environment exactly.

## Results

| Measure | Value |
| --- | ---: |
| Test defaults / nondefaults | 52 / 204 |
| ROC AUC | 0.698718 |
| Brier score | 0.158651 |
| Log loss | 0.493716 |
| Mean predicted probability | 0.139917 |
| Observed default fraction | 0.203125 |
| Expected defaults, sum of probabilities | 35.8187 |
| Minimum / maximum prediction | 0.092247 / 0.271862 |
| Accuracy at 0.5 | 0.796875 |
| Sensitivity/default recall at 0.5 | 0 |
| Defaults in highest-risk quartile | 25 / 64 (39.0625%) |

At threshold 0.5 every row was classified as nondefault (TN 204, FN 52, TP 0, FP 0). Accuracy therefore equals the majority-class baseline. Ranking is more promising than this threshold result, but risks are low on average relative to observed outcomes.

A constant prediction equal to the *observed test prevalence* gives Brier 0.161865 and log loss 0.504704. This is a retrospective comparator that uses test labels; it is not a deployable zero-shot baseline. Constant probability 0.5 gives Brier 0.25 and log loss 0.693147.

The sorted risk quartiles had observed default rates 12.5%, 12.5%, 17.1875%, 39.0625%, versus mean predictions 10.931%, 12.764%, 14.274%, 17.997%. These are descriptive summaries of a small sample, not proof of population calibration.

## Verification and limitations

The exported CSV matched the predictions in the run ZIP. Context and test data hashes matched the prepared data after reproducing the originating Windows CRLF serialization. The current fingerprint uses platform-default CSV newlines: comparing a Windows run on Linux may falsely flag a mismatch. Do not override a mismatch blindly; verify values, row order, columns and the alternate newline serialization. This portability issue remains in v0.4.0.

Masked labels were used only for this retrospective scoring. Source independence and absence of benchmark exposure during pretraining are not established. No direct-equation ablation, repeated-seed study, external validation, or uncertainty-coverage study has been completed. Therefore we cannot yet attribute the result to TabPFN rather than its generators, or claim that the reported uncertainties are calibrated. After inspecting these results, this split is development data for subsequent decisions; use untouched data for confirmation.
