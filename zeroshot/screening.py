"""Dataset-overlap review, separate from coefficient extraction.

Automated suggestions are not proof of independence. A reviewer records the
study population and why it does not overlap the benchmark before prediction.
"""
import hashlib
import json
import re


def policy(value=None):
    """Validate explicit user exclusions; do not infer a benchmark from columns."""
    value = value or {}
    aliases = value.get("excluded_datasets", [])
    if not isinstance(aliases, list) or len(aliases) > 30 or any(
        not isinstance(x, str) or not 3 <= len(x.strip()) <= 200 for x in aliases
    ):
        raise ValueError("Use at most 30 excluded dataset names, each 3–200 characters.")
    return {"excluded_datasets": list(dict.fromkeys(x.strip() for x in aliases))}


def normalize(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def review_key(source, settings):
    """Changing source content or exclusions invalidates its human review."""
    content = [source.get(k, "") for k in ("title", "url", "text")]
    content.append(policy(settings))
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def source_flags(source, settings):
    text = source.get("text", "")
    searchable = normalize(source.get("title", "") + " " + text)
    matches = [a for a in policy(settings)["excluded_datasets"]
               if normalize(a) in searchable]
    flags = []
    if len(text.strip()) < 200:
        flags.append("Insufficient retrieved text: fetch a PDF or upload the paper.")
    if source.get("kind") == "abstract":
        flags.append("Abstract only: full methods and coefficient tables may be missing.")
    if matches:
        flags.append("Possible dataset overlap: " + ", ".join(matches) +
                     ". A mention may be a citation; inspect the study's own data section.")
    if not policy(settings)["excluded_datasets"]:
        flags.append("No evaluation dataset exclusions configured.")
    return flags


def require_source_review(source, settings):
    """Background definitions are permitted, but cannot supply a fitted model."""
    if source.get("dataset_review", {}).get("decision") == "exclude":
        raise ValueError("Review dataset independence: source is explicitly excluded: " + source["title"])
    review = source.get("dataset_review", {})
    if review.get("decision") == "approve":
        if review.get("key") != review_key(source, settings):
            raise ValueError("Source approval is stale: " + source["title"])
        report = source.get("screening_report", {})
        if policy(settings)["excluded_datasets"] and not (
                report.get("status") in ("potentially_independent", "background")
                and report.get("key") == review_key(source, settings)
                and report.get("quote", "").strip()
                and report["quote"] in source.get("text", "")):
            raise ValueError("Automatic screening could not establish eligible study data: " + source["title"])
        return
    if not policy(settings)["excluded_datasets"]:
        return
    review = source.get("dataset_review", {})
    if (review.get("key") != review_key(source, settings)
            or review.get("decision") not in ("independent", "background")
            ):
        raise ValueError("Review dataset independence for source: " + source["title"])
    if review["decision"] == "independent" and len(source.get("text", "").strip()) < 200:
        raise ValueError("Insufficient source text for model evidence: " + source["title"])


def check_models(bundle):
    """Enforce the same evidence gate in the web app and command-line engine."""
    settings = policy(bundle.get("research_policy"))
    sources = {s["id"]: s for s in bundle["sources"]}
    for model in bundle["models"]:
        proposed = model.get('model_origin', 'published') != 'published'
        if proposed and bundle.get('model_policy', {}).get('mode') != 'delegate':
            raise ValueError('Proposed models require delegated modeling to be enabled.')
        independent = False
        anchors = [*(model.get("anchors") or []), *(a for d in model.get("missing_distributions") or [] for a in d.get("anchors") or [])]
        for anchor in anchors:
            source = sources.get(anchor.get("source_id"))
            if source is None:
                raise ValueError("Model references a missing source.")
            require_source_review(source, settings)
            decision = source.get("dataset_review", {}).get("decision")
            # User approval and automated provenance advice remain distinct.
            independent |= decision == "independent" or (decision == "approve" and
                source.get("screening_report", {}).get("status") == "potentially_independent")
        if settings["excluded_datasets"] and not independent and not proposed:
            raise ValueError("Each model needs a reviewed independent study, not only background definitions.")
