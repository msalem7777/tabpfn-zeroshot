"""Whole-world Monte Carlo simulation and independent direct/TabPFN prediction."""
import hashlib
import importlib.metadata
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import expit

from .distributions import summarize
from .evidence import design_matrix, validate_models
from .schema import RunConfig, Target
from .tabpfn_adapter import TabPFNAdapter
from .screening import check_models
from .marginalization import eligibility, sample_frame


class Cancelled(ValueError):
    pass


def world_parameters(model, rng):
    """One joint coefficient draw governs ALL rows of the sampled dataset."""
    if model.parameter_distribution == "empirical":
        index = rng.integers(len(model.parameter_draws))
        beta = np.array(model.parameter_draws[index], dtype=float)
        sigma = (model.residual_draws[index] if model.residual_draws is not None
                 else model.residual_sd * np.exp(rng.normal(0, model.residual_log_sd)))
    else:
        beta = rng.multivariate_normal(model.coefficients, model.coefficient_covariance(), check_valid="raise")
        sigma = model.residual_sd * np.exp(rng.normal(0, model.residual_log_sd))
    beta[0] += rng.normal(0, model.transfer_intercept_sd)
    return beta, sigma


def world_prediction(model, design, beta, sigma, count, rng):
    eta = design @ beta
    if model.family == "logistic":
        p = expit(eta)
        return p, p*(1-p), rng.binomial(1, p, size=(count, len(p)))
    mu = model.target_center + model.target_scale * eta
    sd = model.target_scale * sigma
    samples = rng.normal(mu, sd, size=(count, len(mu)))
    if model.target_transform == "log":
        # E[exp(Z)] is NOT exp(E[Z]); retain the lognormal correction.
        with np.errstate(over="ignore", invalid="ignore"):
            samples = np.exp(samples)
            variance = np.expm1(sd**2) * np.exp(2*mu + sd**2)
            mean = np.exp(mu + .5*sd**2)
    else:
        mean, variance = mu, np.full(len(mu), sd**2)
    if not all(np.isfinite(a).all() for a in (mean, variance, samples)):
        raise ValueError("A sampled model overflowed. Narrow/review uncertainty or fix target transformations.")
    return mean, variance, samples


def run(context, test, bundle, config, progress=lambda *_: None, cancelled=lambda: False,
        adapter_factory=TabPFNAdapter):
    check_models(bundle)
    config = RunConfig.model_validate(config)
    target = Target.model_validate(bundle["target"])
    if not config.assumptions_approved:
        raise ValueError("Approve the displayed statistical assumptions and run settings first.")
    if target.name in context.columns or (test is not None and target.name in test.columns):
        raise ValueError("The target must be absent from both datasets. Remove it before zero-shot prediction.")
    if context.columns.duplicated().any() or len(context) < 8:
        raise ValueError("Use at least 8 context rows and unique column names.")
    query = context if test is None else test
    if not len(query) or len(query) > 10000:
        raise ValueError("Use 1–10,000 prediction rows per run.")
    if config.worlds * config.samples_per_world * len(query) > 20_000_000:
        raise ValueError("Prediction sample budget exceeds 20 million values. Reduce worlds, samples or query rows.")
    models = validate_models(bundle["models"], bundle["sources"], target, context, config.allow_synthetic, check_data=False)
    original_context, original_query = context, query
    c_masks = np.stack([eligibility(context, m)[0] for m in models])
    # In the primary pipeline literature is evaluated ONLY on context inputs.
    pfn_only = config.method == "tabpfn"
    q_masks = None if pfn_only else np.stack([eligibility(query, m)[0] for m in models])
    c_positions = np.flatnonzero(c_masks.any(axis=0))
    q_positions = np.arange(len(query)) if pfn_only else np.flatnonzero(q_masks.any(axis=0))
    if not len(q_positions):
        raise ValueError("No prediction rows have a usable equation or supported missing-predictor distribution.")
    if config.method != 'direct' and len(c_positions) < 8:
        raise ValueError("TabPFN needs at least 8 context rows with supported generated labels.")
    context = context.iloc[c_positions].reset_index(drop=True)
    query = query.iloc[q_positions].reset_index(drop=True)
    c_masks = c_masks[:, c_positions]
    if q_masks is not None:
        q_masks = q_masks[:, q_positions]
    evidence_columns = list(dict.fromkeys(p.column for m in models for p in m.predictors if p.column in context))
    columns = list(context.columns) if config.use_all_features else evidence_columns
    if config.method != 'direct' and any(c not in query for c in columns):
        raise ValueError("Prediction rows lack a selected TabPFN feature column.")
    # Separate deterministic RNG streams: turning TabPFN on cannot change direct worlds.
    streams = np.random.SeedSequence(config.seed).spawn(4)
    world_rng, direct_rng, label_rng, pfn_rng = [np.random.default_rng(s) for s in streams]
    missing_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 81731]))
    split_rng = np.random.default_rng(config.seed)
    fold_indices = np.array_split(split_rng.permutation(len(query)), config.folds) if test is None else []
    splits = []
    if test is None:
        for holdout in fold_indices:
            # Match original positions: filtered context indices are not query indices.
            train = np.flatnonzero(~np.isin(c_positions, q_positions[holdout]))
            if config.method != "direct" and not len(train):
                raise ValueError("A held-out group has no supported context rows.")
            if len(train) > config.context_limit:
                train = split_rng.choice(train, config.context_limit, replace=False)
            splits.append((train, holdout))
    else:
        train = np.arange(len(context))
        if len(train) > config.context_limit:
            train = split_rng.choice(train, config.context_limit, replace=False)
        splits = [(train, np.arange(len(query)))]
    weights = np.array([m.weight for m in models]); weights /= weights.sum()
    # One shared uniform model draw per world preserves shared study uncertainty.
    # Where a model is inapplicable, renormalize weights over eligible equations.
    def choose_rows(mask, u):
        probability = mask * weights[:, None]
        probability /= probability.sum(axis=0)
        return np.minimum((np.cumsum(probability, axis=0) <= u).sum(axis=0), len(models)-1)
    direct, pfn, worlds = [], [], []
    adapter = None
    try:
        if config.method != "direct":
            progress("Loading TabPFN; first use may require a model download and license acceptance", 0)
            adapter = adapter_factory(target.family, config)
        for k in range(config.worlds):
            if cancelled():
                raise Cancelled("Run cancelled between inference calls.")
            u = world_rng.random()
            c_choice = choose_rows(c_masks, u)
            q_choice = np.array([], dtype=int) if pfn_only else choose_rows(q_masks, u)
            dm, dv = np.empty(len(query)), np.empty(len(query))
            ds = np.empty((config.samples_per_world, len(query)))
            labels = np.empty(len(context))
            parameters = []
            for mi in sorted(set(c_choice) | set(q_choice)):
                model = models[mi]
                beta, sigma = world_parameters(model, world_rng)
                ci, qi = np.flatnonzero(c_choice == mi), np.flatnonzero(q_choice == mi)
                # For cross-fitting, the context and query are the same rows:
                # reuse the same sampled missing values for both roles.
                c_frame = sample_frame(context.iloc[ci], model, missing_rng)
                q_frame = None if pfn_only else (c_frame if test is None else sample_frame(query.iloc[qi], model, missing_rng))
                if len(qi):
                    dm[qi], dv[qi], ds[:, qi] = world_prediction(model, design_matrix(q_frame, model), beta, sigma, config.samples_per_world, direct_rng)
                if adapter and len(ci):
                    cm, _, cs = world_prediction(model, design_matrix(c_frame, model), beta, sigma, 1, label_rng)
                    # Regressors can consume the literature prediction directly.
                    # Classifiers require sampled 0/1 labels, not soft probabilities.
                    labels[ci] = (cm if pfn_only and target.family == "linear"
                                  and config.regression_context_labels == "mean" else cs[0])
                parameters.append({'model_id': model.id, 'coefficients': beta.tolist(), 'residual_sd': float(sigma)})
            if not pfn_only:
                direct.append((dm, dv, ds))
            entry = {"world": k, 'parameters': parameters,
                     'context_model_indices': c_choice.tolist(), 'query_model_indices': q_choice.tolist()}
            # Preserve the old audit fields when a world uses a single equation.
            if len(parameters) == 1:
                entry.update(parameters[0])
            if pfn_only:
                entry.pop('query_model_indices')
                entry['context_labels'] = labels.tolist()
            worlds.append(entry)
            if adapter:
                mean, variance = np.empty(len(query)), np.empty(len(query))
                samples = np.empty((config.samples_per_world, len(query)))
                for fi, (train, holdout) in enumerate(splits):
                    if cancelled():
                        raise Cancelled("Run cancelled between inference calls.")
                    progress(f"World {k+1}/{config.worlds} · prediction group {fi+1}/{len(splits)}", k/config.worlds)
                    a, b, c = adapter.predict(context.iloc[train][columns], labels[train],
                                              query.iloc[holdout][columns], config.samples_per_world, pfn_rng)
                    mean[holdout], variance[holdout], samples[:, holdout] = a, b, c
                pfn.append((mean, variance, samples))
            progress(f"Completed world {k+1}/{config.worlds}", (k+1)/config.worlds)
    finally:
        if adapter:
            adapter.close()
    output = pd.DataFrame({"row_position": q_positions})
    arrays = {}
    for name, values in [("direct", direct), ("tabpfn", pfn)]:
        if not values:
            continue
        means = np.stack([v[0] for v in values]); variances = np.stack([v[1] for v in values])
        samples = np.stack([v[2] for v in values])
        stats = summarize(means, variances, samples, config.interval)
        for key, val in stats.items():
            output[f"{name}_{key}"] = val
        if target.family == "logistic":
            alpha = (1-config.interval)/2
            # This interval describes disagreement about probability, not a 0/1 outcome interval.
            output[f"{name}_probability_lower"] = np.quantile(means, alpha, axis=0)
            output[f"{name}_probability_upper"] = np.quantile(means, 1-alpha, axis=0)
        arrays[name + "_samples"] = samples
        arrays[name + "_component_means"] = means
        arrays[name + "_component_variances"] = variances
    versions = {}
    for package in ["numpy", "pandas", "scipy", "tabpfn", "torch"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    warnings = ["Intervals are conditional on reviewed evidence and assumptions; target-population coverage is unverified.",
                "Candidate weights are recorded scenario weights, not automatically learned posterior model probabilities.",
                "More generated rows/worlds do not create observed evidence.",
                "Mixture quantiles use finite predictive samples; rerun with more draws to assess numerical stability."]
    proposed = [m.id for m in models if m.model_origin != 'published']
    if proposed:
        warnings.insert(0, 'ASSUMPTION-BASED MODELS: ' + ', '.join(proposed) +
                        '. These are LLM/user-proposed scenarios, not extracted published fits. '
                        'Their coefficients and uncertainty are assumptions; the TabPFN output is conditional on them.')
    if any(m.synthetic for m in models):
        warnings.insert(0, "DEMO: fictional evidence, not research-backed predictions.")
    if config.method != "direct":
        warnings.append("TabPFN mixture is a two-stage predictive procedure, not guaranteed to be one coherent Bayesian posterior.")
    used = Counter(p['model_id'] for w in worlds for p in w['parameters'])
    missing = [m.id for m in models if m.id not in used]
    if missing:
        warnings.append("No Monte Carlo draw selected these candidates: " + ", ".join(missing) + ". Increase worlds.")
    skipped_rows = {'context': np.flatnonzero(~np.isin(np.arange(len(original_context)), c_positions)).tolist(),
                    'query': np.flatnonzero(~np.isin(np.arange(len(original_query)), q_positions)).tolist()}
    if any(skipped_rows.values()):
        warnings.append(f"Insufficient literature evidence for {len(skipped_rows['context'])} context rows; these were not used as labeled examples. Skipped {len(skipped_rows['query'])} prediction rows. Original row positions are retained.")
    if any(m.missing_distributions for m in models):
        warnings.append("Missing predictors are sampled from quoted source distributions per row/world. Their distribution parameters are treated as fixed; unreported uncertainty is not invented.")
    if config.use_all_features:
        warnings.append("Extra observed columns are passed to TabPFN, but source equations supply no independent evidence of their additional effects.")
    audit = {"version": "0.4.0", "created_utc": datetime.now(timezone.utc).isoformat(),
             "research_policy": bundle.get("research_policy", {"excluded_datasets": []}),
             "model_policy": bundle.get('model_policy', {}),
             "agent_log": bundle.get('agent_log', []),
             "config": config.model_dump(), "target": target.model_dump(),
             "evidence": [m.model_dump() for m in models], "worlds": worlds,
             "source_manifest": [{k: v for k, v in s.items() if k != "text"} for s in bundle["sources"]],
             "context_sha256": hashlib.sha256(original_context.to_csv(index=False).encode()).hexdigest(),
             "query_sha256": hashlib.sha256(original_query.to_csv(index=False).encode()).hexdigest(),
             "prediction_mode": "cross_fitted" if test is None else "separate_test_rows",
             "splits": [{"context_positions": c_positions[a].tolist(), "prediction_positions": q_positions[b].tolist()} for a, b in splits],
             "skipped_rows": skipped_rows, "model_index": [m.id for m in models],
             "preparation": bundle.get('preparation', {}),
             "final_prediction": "mean of TabPFN predictive distributions" if pfn_only else "legacy diagnostic",
             "literature_evaluated_on_test": not pfn_only, "tabpfn_features": columns, "model_draw_counts": dict(used), "versions": versions, "warnings": warnings}
    return {"predictions": output, "arrays": arrays, "audit": audit}
