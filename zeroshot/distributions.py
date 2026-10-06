"""Distribution mixtures: keep within-world and between-world uncertainty distinct."""
import numpy as np


def summarize(means, variances, samples, interval):
    """Equal-weight worlds have already been sampled using evidence-model weights.

    Variances use population moments of the finite mixture (ddof=0), not the
    standard error of its estimated mean. Quantiles come from pooled samples.
    """
    means, variances, samples = map(np.asarray, (means, variances, samples))
    if not all(np.isfinite(a).all() for a in (means, variances, samples)):
        raise ValueError("Non-finite predictions; inspect evidence scales and parameter uncertainty.")
    mean = means.mean(axis=0)
    within = variances.mean(axis=0)
    between = means.var(axis=0, ddof=0)
    if np.any(within < -1e-8):
        raise ValueError("Invalid negative predictive variance.")
    within = np.maximum(within, 0)
    alpha = (1 - interval) / 2
    low, median, high = np.quantile(samples.reshape(-1, means.shape[1]), [alpha, .5, 1-alpha], axis=0)
    return {"mean": mean, "median": median, "lower": low, "upper": high,
            "within_variance": within, "between_variance": between,
            "total_variance": within + between, "sd": np.sqrt(within + between)}


def bar_samples(criterion, logits, count, rng):
    """Sample TabPFN's full-support bars, including its half-normal tails.

    Some versions' inherited icdf interpolates over finite edge borders even
    though mean/variance use unbounded half-normal tails. Sampling bins and
    tails explicitly keeps these definitions consistent. Unsupported criteria
    fail visibly; there is no point-estimate or Gaussian fallback.
    """
    def array(t):
        return t.detach().cpu().numpy()
    if not hasattr(criterion, "halfnormal_with_p_weight_before"):
        raise ValueError("Unsupported TabPFN distribution: full-support half-normal tails required.")
    borders = array(criterion.borders).astype(float)
    probs = array(logits.softmax(-1)).astype(float)
    if probs.ndim != 2 or len(borders) != probs.shape[1] + 1:
        raise ValueError("Unsupported TabPFN output shape.")
    if not np.isfinite(probs).all() or not np.isfinite(borders).all() or np.any(np.diff(borders) <= 0):
        raise ValueError("Invalid TabPFN probabilities or borders.")
    out = np.empty((count, len(probs)))
    left_scale = float(criterion.halfnormal_with_p_weight_before(criterion.bucket_widths[0]).scale.cpu())
    right_scale = float(criterion.halfnormal_with_p_weight_before(criterion.bucket_widths[-1]).scale.cpu())
    for i, p in enumerate(probs):
        p = p / p.sum()
        bins = rng.choice(len(p), size=count, p=p)
        values = borders[bins] + rng.random(count) * (borders[bins + 1] - borders[bins])
        left, right = bins == 0, bins == len(p)-1
        values[left] = borders[1] - np.abs(rng.normal(size=left.sum())) * left_scale
        values[right] = borders[-2] + np.abs(rng.normal(size=right.sum())) * right_scale
        out[:, i] = values
    return out

