"""Eligibility and Monte Carlo integration of missing source predictors.

No fitting uses hidden outcomes. Unknown categories and invalid observed values
are not silently imputed. Only absent columns and null cells may be sampled.
"""
import numpy as np
import pandas as pd


def numeric(frame, column):
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def transformed(series, predictor):
    if predictor.category_map is not None:
        raw = series.map(lambda v: str(int(v)) if isinstance(v, (int, float)) and np.isfinite(v) and v == int(v) else str(v))
        x = raw.map(predictor.category_map).to_numpy(dtype=float)
    else:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x * predictor.multiplier + predictor.offset
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if predictor.transform == "log":
            x = np.log(x)
        elif predictor.transform == "log1p":
            x = np.log1p(x)
        elif predictor.transform == "square":
            x = np.square(x)
        return (x - predictor.source_center) / predictor.source_scale


def eligibility(frame, model):
    """Return a per-row mask and reasons without drawing any random numbers."""
    ok = np.ones(len(frame), dtype=bool)
    reasons = [[] for _ in range(len(frame))]
    distributions = {d.column: d for d in model.missing_distributions}
    missing_names = [set() for _ in range(len(frame))]

    def block(mask, message):
        nonlocal ok
        ok &= ~mask
        for i in np.flatnonzero(mask):
            if message not in reasons[i]:
                reasons[i].append(message)

    for predictor in model.predictors:
        series = frame[predictor.column] if predictor.column in frame else pd.Series(np.nan, index=frame.index)
        missing = series.isna().to_numpy()
        observed_bad = ~missing & ~np.isfinite(transformed(series, predictor))
        block(observed_bad, f"{predictor.column}: invalid observed value, category or transformation")
        if not missing.any():
            continue
        distribution = distributions.get(predictor.column)
        if distribution is None:
            block(missing, f"{predictor.column}: missing with no source-supported distribution")
            continue
        for column in distribution.conditional_mean_coefficients:
            block(missing & ~np.isfinite(numeric(frame, column)),
                  f"{predictor.column}: conditioning variable {column} unavailable")
        # Do not truncate a normal distribution after drawing: that would change
        # its meaning. Reject unsupported transformations before sampling.
        if distribution.family == "normal" and distribution.sd > 0 and (
                predictor.category_map is not None or predictor.transform in ("log", "log1p")):
            block(missing, f"{predictor.column}: normal support incompatible with category/log transform")
        if distribution.family == "lognormal" and (predictor.category_map is not None or
                (predictor.transform in ("log", "log1p") and
                 (predictor.multiplier <= 0 or predictor.offset < (0 if predictor.transform == 'log' else -1)))):
            block(missing, f"{predictor.column}: lognormal support incompatible with mapping")
        if distribution.family == "categorical":
            support = pd.Series(distribution.values)
            if not np.isfinite(transformed(support, predictor)).all():
                block(missing, f"{predictor.column}: sampled category support has invalid mappings")
        elif distribution.sd == 0:
            values = np.full(len(frame), float(distribution.mean))
            for column, slope in distribution.conditional_mean_coefficients.items():
                values += slope * numeric(frame, column)
            if distribution.family == "lognormal":
                with np.errstate(over="ignore"):
                    values = np.exp(values)
            block(missing & ~np.isfinite(transformed(pd.Series(values), predictor)),
                  f"{predictor.column}: fixed distribution value has invalid transformation")
        for i in np.flatnonzero(missing):
            missing_names[i].add(predictor.column)
    for i, names in enumerate(missing_names):
        if len(names) > 1 and any(not distributions[name].independence_basis.strip() for name in names):
            mask = np.zeros(len(frame), dtype=bool); mask[i] = True
            block(mask, "Joint distribution of missing predictors unavailable; independence not established")
    return ok, reasons


def sample_frame(frame, model, rng):
    """Sample each missing raw predictor once per row/world, shared by its terms.

    Conditional models use observed inputs only; no order-dependent chained
    imputation or hidden assumption about dependencies is introduced.
    """
    output = frame.copy()
    for distribution in model.missing_distributions:
        name = distribution.column
        if name not in output:
            output[name] = np.nan
        mask = output[name].isna().to_numpy()
        if not mask.any():
            continue
        n = int(mask.sum())
        if distribution.family == "categorical":
            draws = rng.choice(np.asarray(distribution.values, dtype=object), n, p=distribution.probabilities)
            output[name] = output[name].astype(object)
        else:
            mean = np.full(n, distribution.mean)
            for column, slope in distribution.conditional_mean_coefficients.items():
                mean += slope * numeric(frame, column)[mask]
            draws = rng.normal(mean, distribution.sd)
            if distribution.family == "lognormal":
                draws = np.exp(draws)
            if not np.isfinite(draws).all():
                raise ValueError(f"{name}: missing-predictor distribution overflowed.")
        output.loc[mask, name] = draws
    return output
