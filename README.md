# ZeroShot: agent-assisted transfer learning for TabPFN

A general-purpose transfer-learning framework for TabPFN, designed for tabular prediction with **zero observed target labels in the supplied context**. LLM agents search the literature, extract coefficients and their reported uncertainties, and propose candidate models. Monte Carlo sampling—repeatedly drawing plausible model parameters—turns those models into alternative labeled context datasets. TabPFN uses each dataset as examples for predicting test rows, without retraining or changing its pretrained weights. The final prediction combines the resulting TabPFN predictive distributions.

**Status:** experimental. The framework transfers external knowledge through generated labels and is intended for use across application domains. The current implementation supports regression (numeric outcomes) and binary classification (two possible outcomes), using linear or logistic label-generating models. Generalization across domains and uncertainty calibration—whether stated uncertainty agrees with observed errors—remain to be evaluated. Credit default is an initial testing example, not the framework's scope. This project is independent of Prior Labs.

## What happens

1. Supply context covariates (input columns used to generate example labels), test covariates (rows to predict), and a target description. Connect a supported LLM provider with your own API token.
2. Delegate literature discovery, equation extraction, model proposal, and review to the agents described below. Uploaded papers and user-specified models are also supported.
3. Review the selected sources and candidate models. Published numerical estimates, agent-proposed assumptions, and user-specified values are recorded separately. When no eligible sources are available, proposals are explicitly labeled **unsourced LLM priors**: model assumptions supplied by the LLM without supporting source evidence.
4. Sample model choices and uncertain parameters, generate context labels, and run TabPFN. Repeat and combine its predictive distributions—the possible outcomes and their assigned probabilities.

## Agent workflow

An agent here is an LLM assigned a specific task within a controlled workflow. The current implementation runs the following roles sequentially through the same connected LLM:

| Role | Responsibility |
| --- | --- |
| Researcher | Discover relevant papers when no sources have been supplied, reuse saved papers, and screen sources against the task's restrictions. |
| Extractor | Recover usable published equations, coefficients, reported uncertainties, variable definitions, and required transformations. |
| Model proposer | Propose candidate linear/logistic models when extraction is insufficient or the user requests custom models. Explicitly label proposed numbers as assumptions. |
| Reviewer | Critique proposed models for consistency and suitability, and provide feedback for a bounded revision step. |

The backend then checks the model format and compatibility with the supplied columns before generating labels. The agents automate research and model construction; using the same LLM in several roles does not provide independent scientific verification. Exact user-specified models remain supported alongside the delegated workflow.

## Statistical idea

Let $X_C$ be unlabeled context covariates, $x_*$ a test row, and $E$ the accepted evidence plus explicitly declared assumptions. In world $m$, sample an equation and its parameters, then generate synthetic context labels:

$$
(J_m,\theta_m)\sim\pi(J,\theta\mid E),\qquad \tilde y_C^{(m)}\sim p_{J_m,\theta_m}(\cdot\mid X_C).
$$

For binary targets this means Bernoulli draws from a sampled logistic equation. For regression the default uses the sampled equation's conditional means; sampling noisy outcomes is optional. Thus the regression default deliberately differs from drawing full outcomes above.

A fixed pretrained TabPFN then supplies an in-context predictive distribution:

$$
q_m(y_*\mid x_*)=q_{\mathrm{PFN}}(y_*\mid x_*,X_C,\tilde y_C^{(m)}),\qquad
\hat q(y_*\mid x_*)=\frac1M\sum_{m=1}^{M}q_m(y_*\mid x_*).
$$

**Transfer** occurs through synthetic labels, not neural weight updates. **In-context learning** means predictions depend on the supplied labeled examples while TabPFN's pretrained weights remain fixed. Each `.fit` replaces/prepares that context; there is no gradient fine-tuning. Literature equations generate context labels; they are not separately blended into the final prediction.

With world means $\mu_m$ and variances $v_m$, the mixture obeys:

$$
\bar\mu=\frac1M\sum_m\mu_m,\qquad
V=\underbrace{\frac1M\sum_m v_m}_{\text{within-world}}
+\underbrace{\frac1M\sum_m(\mu_m-\bar\mu)^2}_{\text{between-world}}.
$$

Intervals come from the pooled predictive distribution, not averaged interval endpoints. More worlds reduce Monte Carlo integration error; they do not eliminate predictive uncertainty. This propagates specified uncertainty, not every possible source of uncertainty.

See [the statistical specification](docs/STATISTICS.md) for parameter distributions, normalization, missing predictors, model weights, and limitations of the Bayesian interpretation.

## Design choices and limits

- **Numerical provenance matters.** Proposed coefficients, scales, standard deviations, and weights are assumptions, even when papers qualitatively motivate them. Model weights are scenario weights, not fitted posterior probabilities. This is not an implemented hierarchical meta-analysis.
- **Units and definitions must match.** Published transformations are explicit. Missing published predictors require a separately justified reduced equation or a declared distribution for marginalization. Dropping a coefficient does not generally recover a valid reduced model. Extra context columns may enter TabPFN; no literature effect is invented for them.
- **Zero-shot is narrowly defined.** The local target labels are unused until evaluation. External literature and pretraining already contain information. Familiar benchmark exposure through the LLM or TabPFN cannot be ruled out. Unlabeled covariates alone do not identify the true target relationship.
- **Bayesian-inspired, not an exact posterior.** The chosen parameter distributions and TabPFN's learned prior need not form one coherent generative model. Uncertainty can be misspecified, altered, or double-counted. Calibration and transportability require empirical evaluation.
- **Auditable but not deterministic forever.** Exports include assumptions, sources, sampled parameters/context labels, configuration, hashes, and versions. Provider responses and broad dependency ranges can change. The LLM receives dataset summaries and source excerpts, not individual context rows; summaries may still reveal sensitive information. Tokens are excluded from workspace exports.

## Example evaluation: credit default

One exploratory Taiwan credit run used 512 context rows, 256 test rows, 23 predictors, and 32 Monte Carlo worlds. Its three logistic generators used **LLM-proposed numbers**, not complete published fitted equations.

| Metric | Observed result |
| --- | ---: |
| ROC AUC | 0.6987 |
| Brier score / log loss | 0.1587 / 0.4937 |
| Mean predicted / observed default rate | 13.99% / 20.31% |
| Default recall at threshold 0.5 | 0% |

There is some ranking signal, alongside underprediction of default risk. No direct-generator ablation has established that TabPFN improves over its synthetic-label generators. One small split establishes neither general performance nor uncertainty calibration. This split has now been inspected and is development data for future changes. [Full pilot details](docs/EXPERIMENT.md).

## Run locally

Python 3.11+; run these commands from the repository root in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[tabpfn,test]"
.\.venv\Scripts\tabpfn-zeroshot.exe
```

Open http://127.0.0.1:7432. Connect an LLM, load covariates and target metadata, build/review models, and run the literature-context-to-TabPFN path. TabPFN weights may require download/access under the upstream model terms. LLM availability and cost depend on your provider. Save a full workspace before restarting; sessions are in memory and expire after eight hours. Workspace exports exclude raw dataset rows, which must be reloaded.

[Credit data walkthrough](docs/CREDIT_EXAMPLE.md)

Offline tests (no paid inference):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Core code under `zeroshot/`: `engine.py` (sampling and combining predictions), `tabpfn_adapter.py` (TabPFN prediction without weight updates), `model_agents.py` (agent coordination), `research.py` and `retrieval.py` (literature discovery and extraction), `schema.py` (required data/model formats), `marginalization.py` (integrating over missing predictors), and `web.py` (local UI). Synthetic fixtures are included; user papers, real datasets, credentials, weights, and saved runs are excluded. Legacy direct-equation modes remain for diagnostics/demo only.

## Attribution and licensing

Built around the separately distributed [TabPFN](https://github.com/PriorLabs/TabPFN) package. Dataset attribution is in the credit walkthrough. No project license has been selected yet; this repository does not grant a new license to third-party software or weights. The publication copy uses a plain ZS mark; the borrowed Prior Labs mascot is omitted. See [asset notes](THIRD_PARTY_ASSETS.md).
