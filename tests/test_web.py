"""End-to-end Flask routes with real direct prediction; paid/network paths mocked."""
import io
import json
import time
import zipfile

import pytest

from zeroshot.web import create_app


@pytest.fixture
def client():
    app=create_app();app.config["TESTING"]=True
    return app.test_client()


def bootstrap(client):
    data=client.get("/api/state").get_json()
    return {"X-CSRF-Token":data["csrf"]}


def test_csrf_host_and_session_isolation(client):
    h=bootstrap(client)
    assert client.post("/api/demo",json={}).status_code==403
    assert client.post("/api/demo",headers={**h,"Origin":"https://evil.example"},json={}).status_code==403
    assert client.get("/",headers={"Host":"evil.example"}).status_code==400
    assert client.post("/api/demo",headers=h,json={}).status_code==200
    other=client.application.test_client()
    assert other.get("/api/state").get_json()["profile"] is None


def test_complete_demo_export_and_reimport(client):
    h=bootstrap(client);client.post("/api/demo",json={},headers=h)
    s=client.get("/api/state").get_json();s["bundle"]["models"][0]["reviewed"]=True
    assert client.post("/api/bundle",headers=h,json=s["bundle"]).status_code==200
    cfg={"method":"direct","worlds":8,"samples_per_world":16,"assumptions_approved":True,"allow_synthetic":True}
    assert client.post("/api/validate",headers=h,json=cfg).status_code==200
    assert client.post("/api/run",headers=h,json=cfg).status_code==200
    for _ in range(200):
        job=client.get("/api/job").get_json()
        if not job["busy"]:break
        time.sleep(.01)
    assert job["job"]["status"]=="complete",job
    result=client.get("/api/results").get_json()
    assert result["total_rows"]==60
    assert len(client.get("/api/distribution/0").get_json()["edges"])==41
    download=client.get("/api/download/run")
    z=zipfile.ZipFile(io.BytesIO(download.data))
    assert {"predictions.csv","audit.json","evidence.json","predictive_distributions.npz"}<=set(z.namelist())
    evidence=json.loads(z.read("evidence.json"))
    assert "token" not in json.dumps(evidence)
    assert client.post("/api/bundle",headers=h,json=evidence).status_code==200
    assert client.post("/api/validate",headers=h,json=cfg).status_code==200


def test_upload_duplicate_headers_and_target_missing(client):
    h=bootstrap(client)
    data={"file":(io.BytesIO(b"a,a\n1,2\n"),"bad.csv"),"role":"context"}
    response=client.post("/api/upload",headers=h,data=data)
    assert response.status_code==400 and "unique" in response.get_json()["error"]


def test_llm_credentials_not_exposed_or_exported(client,monkeypatch):
    h=bootstrap(client)
    monkeypatch.setattr("zeroshot.providers.complete",lambda *a,**k:"Connected")
    payload={"provider":"compatible","endpoint":"https://example.com/v1","model":"test","token":"VERY_SECRET_TOKEN","share_summaries":True,"send_max_tokens":True}
    assert client.post("/api/connect",headers=h,json=payload).status_code==200
    assert "VERY_SECRET_TOKEN" not in client.get("/api/state").get_data(as_text=True)
    assert "VERY_SECRET_TOKEN" not in client.get("/api/download/evidence").get_data(as_text=True)
    assert client.post("/api/disconnect",headers=h,json={}).status_code==200
    assert not client.get("/api/state").get_json()["connected"]

