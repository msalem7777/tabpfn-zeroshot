# ZeroShot: literature-informed context for TabPFN

A research prototype for tabular prediction with **zero observed target labels in the supplied context**. An LLM extracts or proposes simple equations, Monte Carlo sampling turns those equations into alternative labeled context datasets, and a frozen TabPFN predicts test rows from each context. The final prediction mixes the TabPFN outputs.

**Status:** experimental. This is a proposal for transferring external knowledge through synthetic labels, not a demonstrated solution to unrestricted zero-shot learning or a validated credit-decision system. It is independent of Prior Labs.

## What happens

1. Supply context covariates, test covariates, and a target description; connect your chosen supported LLM endpoint with your own token.
2. Let the assistant retrieve sources or upload papers. Sources can be approved/rejected without a written justification; optional manual controls remain available.
3. Extract usable published equations or delegate proposal of linear/logistic models. Published numerical evidence, LLM-proposed assumptions, and user-specified models have different provenance. If no eligible sources exist, proposals are explicitly **unsourced LLM priors**.
4. Sample model choices and uncertain parameters, generate context labels, and run TabPFN. Repeat and aggregate predictive distributions.

The default delegated workflow uses bounded researcher/extractor/proposer/reviewer prompts through the same connected LLM. These are workflow roles, not independent experts. A second prompt and structural validation cannot establish scientific correctness.

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

## Initial credit experiment

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

[Credit data walkthrough](docs/CREDIT_EXAMPLE.md) · [GitHub publishing instructions](PUBLISH_TO_GITHUB.md)

Offline tests (no paid inference):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Core code: `engine.py` (worlds/aggregation), `tabpfn_adapter.py` (frozen predictor), `model_agents.py` (delegation), `schema.py` (contracts), `marginalization.py` (missing predictors), and `web.py` (local UI), under `zeroshot/`. Synthetic fixtures are included; user papers, real datasets, credentials, weights, and saved runs are excluded. Legacy direct-equation modes remain for diagnostics/demo only.

## Attribution and licensing

Built around the separately distributed [TabPFN](https://github.com/PriorLabs/TabPFN) package. Dataset attribution is in the credit walkthrough. No project license has been selected yet; this repository does not grant a new license to third-party software or weights. The publication copy uses a plain ZS mark; the borrowed Prior Labs mascot is omitted. See [asset notes](THIRD_PARTY_ASSETS.md).
