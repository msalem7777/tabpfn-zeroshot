"""Offline checks for evidence provenance, retrieval recovery and exports."""
import copy

import pytest

from zeroshot import research, retrieval, screening
from zeroshot.demo import demo_bundle
from zeroshot.web import create_app


SETTINGS = {"excluded_datasets": ["Taiwan credit card", "Default of Credit Card Clients"]}


def reviewed_bundle():
    _, bundle = demo_bundle()
    bundle["research_policy"] = SETTINGS
    for source in bundle["sources"]:
        source["text"] += " Independent fictional population for software testing." * 10
        source["dataset_review"] = {"decision": "independent", "reason": "Fictional population used only in this unit test.",
                                    "key": screening.review_key(source, SETTINGS)}
    return bundle


def test_review_is_invalidated_by_content_or_policy_change():
    bundle = reviewed_bundle()
    screening.check_models(bundle)
    changed = copy.deepcopy(bundle)
    changed["sources"][0]["text"] += " changed"
    with pytest.raises(ValueError, match="Review dataset independence"):
        screening.check_models(changed)
    bundle["research_policy"] = {"excluded_datasets": ["Different benchmark"]}
    with pytest.raises(ValueError, match="Review dataset independence"):
        screening.check_models(bundle)


def test_background_cannot_supply_a_model_and_excluded_cannot_be_extracted(monkeypatch):
    bundle = reviewed_bundle()
    source = bundle["sources"][0]
    source["dataset_review"]["decision"] = "background"
    with pytest.raises(ValueError, match="independent study"):
        screening.check_models(bundle)
    source["dataset_review"]["decision"] = "exclude"
    monkeypatch.setattr(research.providers, "json_reply", lambda *a: pytest.fail("Must block before an LLM call"))
    with pytest.raises(ValueError, match="Review dataset independence"):
        research.extract({}, bundle["target"], {}, bundle["sources"], SETTINGS)


def test_alias_flag_is_not_a_claim_of_overlap():
    flags = screening.source_flags({"title": "Review cites Taiwan credit-card work", "text": "x" * 300}, SETTINGS)
    assert any("Possible dataset overlap" in x and "citation" in x for x in flags)


def test_screening_rejects_invented_quote(monkeypatch):
    source = reviewed_bundle()["sources"][0]
    monkeypatch.setattr(research.providers, "json_reply", lambda *a: {
        "status": "potentially_independent", "quote": "Fabricated population passage"})
    screened = research.screen_sources({}, [source], SETTINGS)
    assert screened[0]["screening_report"]["status"] == "unknown"


def test_short_download_falls_back_to_abstract(monkeypatch):
    monkeypatch.setattr(retrieval, "get_bytes", lambda url: (b"Bad Gateway", "text/html", url))
    sources, errors = retrieval.collect([{"title": "A study", "url": "https://example.org/a", "abstract": "Study abstract", "doi": ""}])
    assert sources[0]["kind"] == "abstract"
    assert "Insufficient retrieved text" in errors[0]


def test_linked_pdf_is_followed(monkeypatch):
    calls = []
    def get(url):
        calls.append(url)
        if url.endswith(".pdf"):
            return b"%PDF-test", "application/pdf", url
        return b'<meta name="citation_pdf_url" content="/study.pdf">', "text/html", url
    monkeypatch.setattr(retrieval, "get_bytes", get)
    monkeypatch.setattr(retrieval, "text_from_bytes", lambda data, *args: "full coefficients " * 30 if data.startswith(b"%PDF") else "landing")
    source = retrieval.retrieve("https://example.org/article")
    assert calls == ["https://example.org/article", "https://example.org/study.pdf"]
    assert source["url"].endswith("study.pdf")


def test_research_policy_reaches_query_planning_and_ranking(monkeypatch):
    payloads = []
    def reply(config, system, payload):
        payloads.append(payload)
        return {"queries": ["independent default models"]} if len(payloads) == 1 else {"indices": []}
    monkeypatch.setattr(research.providers, "json_reply", reply)
    monkeypatch.setattr(retrieval, "search", lambda query: {"results": [{"doi": "x", "title": "Candidate"}], "errors": [], "query": query})
    result = research.research({}, {}, {}, settings=SETTINGS)
    assert all(x["research_policy"] == SETTINGS for x in payloads)
    assert result["sources"] == [] and result["searches"]


def test_workspace_preserves_text_log_and_review_but_no_credentials(monkeypatch):
    app = create_app(); app.config["TESTING"] = True
    client = app.test_client()
    headers = {"X-CSRF-Token": client.get("/api/state").json["csrf"]}
    monkeypatch.setattr("zeroshot.providers.complete", lambda *a, **k: "ok")
    client.post("/api/connect", headers=headers, json={"provider": "gemini", "model": "test", "token": "PRIVATE-KEY", "share_summaries": True})
    bundle = reviewed_bundle()
    bundle["research_log"] = [{"errors": ["Example retrieval error"]}]
    assert client.post("/api/bundle", headers=headers, json=bundle).status_code == 200
    exported = client.get("/api/download/workspace")
    assert "PRIVATE-KEY" not in exported.get_data(as_text=True)
    full = exported.json
    assert full["sources"][0]["text"] == bundle["sources"][0]["text"]
    assert full["research_log"] == bundle["research_log"]
    assert client.post("/api/bundle", headers=headers, json=full).status_code == 200
    # Old-version state backups can be imported without losing paper text.
    assert client.post("/api/bundle", headers=headers, json={"bundle": full, "research_log": bundle["research_log"]}).status_code == 200
    excerpts = client.get("/api/download/evidence").json
    assert excerpts["sources"][0]["original_text_characters"] > 0
    assert "dataset_review" not in excerpts["sources"][0]
    assert excerpts["research_log"] == bundle["research_log"]
    assert client.post("/api/research-policy", headers=headers, json={"excluded_datasets": ["New excluded population"]}).status_code == 200
    current = client.get("/api/state").json["bundle"]
    assert "dataset_review" not in current["sources"][0]
    assert all(not x["reviewed"] for x in current["models"])
