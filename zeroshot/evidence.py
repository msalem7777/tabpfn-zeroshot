"""Validate provenance and apply explicitly declared unit/scale transformations."""
import hashlib
import re

import numpy as np
import pandas as pd

from .schema import EvidenceModel, Target
from .marginalization import transformed


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def validate_draft_shape(model):
    """Allow unknown scientific values, but reject malformed containers early."""
    if not isinstance(model, dict):
        raise ValueError("Each model draft must be a JSON object.")
    for key in ["predictors", "coefficients", "assumptions", "anchors"]:
        value = model.get(key)
        if value is not None and not isinstance(value, list):
            raise ValueError(f"Draft {key} must be a list or null.")
    for key in ["predictors", "anchors"]:
        if any(not isinstance(x, dict) for x in model.get(key) or []):
            raise ValueError(f"Every draft {key} entry must be an object.")


def source_record(title, text, url="", kind="manual", **metadata):
    """Keep a stable digest of the retrieved bytes' extracted text."""
    text = normalize_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"id": digest[:16], "title": title, "text": text, "url": url,
            "kind": kind, "sha256": digest, **metadata}


def design_matrix(frame: pd.DataFrame, model: EvidenceModel):
    """No inferred normalization, imputation, categorical coding, or executable formulas."""
    columns = [np.ones(len(frame))]
    for p in model.predictors:
        if p.column not in frame:
            raise ValueError(f"{model.id}: missing predictor {p.column!r}.")
        x = transformed(frame[p.column], p)
        if not np.isfinite(x).all():
            raise ValueError(f"{model.id}/{p.column}: missing, nonnumeric, unmapped values or transformation produces invalid values.")
        columns.append(x)
    return np.column_stack(columns)


def validate_models(raw_models, sources, target: Target, frame, allow_synthetic=False, check_data=True):
    """Quote matching verifies presence, NOT scientific correctness; review is mandatory."""
    if not raw_models:
        raise ValueError("No evidence models. Retrieve or import evidence first.")
    if len(raw_models) > 100:
        raise ValueError("At most 100 candidate models per run.")
    index = {s["id"]: s for s in sources}
    models = [EvidenceModel.model_validate(m) for m in raw_models]
    if len({m.id for m in models}) != len(models):
        raise ValueError("Evidence model IDs must be unique.")
    for m in models:
        if not m.reviewed:
            raise ValueError(f"{m.id}: review the source, mappings and assumptions before approving.")
        if m.synthetic and not allow_synthetic:
            raise ValueError("Synthetic evidence is demo-only; enable the clearly labeled demo option.")
        if (m.target != target.name or m.family != target.family or m.target_unit != target.unit
                or m.positive_class != target.positive_class):
            raise ValueError(f"{m.id}: target name, units, family or positive class differs from the requested target.")
        for a in [*m.anchors, *(a for d in m.missing_distributions for a in d.anchors)]:
            source = index.get(a.source_id)
            if source is None or normalize_text(a.quote) not in normalize_text(source["text"]):
                raise ValueError(f"{m.id}: supporting quote is absent from its source.")
            if source["kind"] == "synthetic" and not m.synthetic:
                raise ValueError("Synthetic source cannot support a real-evidence model.")
        if check_data:
            design_matrix(frame, m)
    return models


def profile(frame):
    """Only summaries are sent to the LLM, never individual uploaded rows."""
    items = []
    for name in frame.columns:
        s = frame[name]
        numeric = pd.api.types.is_numeric_dtype(s)
        item = {"name": str(name), "type": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique()), "numeric": bool(numeric)}
        if numeric and s.notna().any():
            finite = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite):
                item.update(min=float(finite.min()), max=float(finite.max()), mean=float(finite.mean()))
        items.append(item)
    return {"rows": len(frame), "columns": items}
