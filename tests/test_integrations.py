"""Provider/retrieval contracts without live credentials or network use."""
import json

import numpy as np
import pytest

from zeroshot import providers, research, retrieval
from zeroshot.demo import demo_bundle
from zeroshot.evidence import profile
from zeroshot.network import validate_url


@pytest.mark.parametrize("provider,response",[
    ("compatible",{"choices":[{"message":{"content":"ok"}}]}),
    ("anthropic",{"content":[{"type":"text","text":"ok"}]}),
    ("gemini",{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}),
])
def test_provider_wire_contract(provider,response,monkeypatch):
    seen={}
    def send(url,headers,payload,**kwargs):
        seen.update(url=url,headers=headers,payload=payload);return response
    monkeypatch.setattr(providers,"post_json",send)
    answer=providers.complete({"provider":provider,"model":"test-model","token":"secret","endpoint":"https://example.com/v1"},[{"role":"user","content":"hi"}],"system")
    assert answer=="ok"
    assert "secret" not in seen["url"]
    assert any("secret" in x for x in seen["headers"].values())


def test_search_really_calls_two_indexes_and_deduplicates(monkeypatch):
    urls=[]
    def get(url):
        urls.append(url)
        result={"message":{"items":[{"DOI":"10.1/a","title":["Study"]}]}} if "crossref" in url else {"resultList":{"result":[{"id":"123","source":"MED","doi":"10.1/a","title":"Study","pmcid":"PMC123","isOpenAccess":"Y"}]}}
        return json.dumps(result).encode(),"application/json",url
    monkeypatch.setattr(retrieval,"get_bytes",get)
    result=retrieval.search("process regression")
    assert len(urls)==2 and len(result["results"])==1
    assert "fullTextXML" in result["results"][0]["fulltext_url"]


def test_extraction_cannot_approve_itself(monkeypatch):
    frame,bundle=demo_bundle();draft=bundle["models"][0];draft["reviewed"]=True
    monkeypatch.setattr(providers,"json_reply",lambda *a,**k:{"models":[draft],"issues":[]})
    out=research.extract({},bundle["target"],profile(frame),bundle["sources"])
    assert out["models"][0]["reviewed"] is False


def test_private_source_urls_are_blocked(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo",lambda *a,**k:[(2,1,6,"",("127.0.0.1",443))])
    with pytest.raises(ValueError,match="public"):
        validate_url("https://internal.example/document")
    assert validate_url("http://127.0.0.1:11434/v1",allow_local=True)


def test_json_reply_rejects_prose_without_fabricated_fallback(monkeypatch):
    monkeypatch.setattr(providers,"complete",lambda *a,**k:"I cannot find any coefficients.")
    with pytest.raises(ValueError,match="invalid JSON"):
        providers.json_reply({},"system",{})
