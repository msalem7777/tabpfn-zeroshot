"""Local-only Flask application. Browser sessions isolate data and credentials."""
import argparse
import copy
import io
import json
import secrets
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file, session
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from . import engine, providers, research, retrieval, screening, automation, preparation, model_agents
from .demo import demo_bundle
from .evidence import profile, source_record, validate_models, validate_draft_shape
from .schema import RunConfig, Target


def read_frame(upload):
    name = upload.filename.lower()
    raw = upload.read()
    if name.endswith(".csv"):
        import csv
        text = raw.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])
        if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
            raise ValueError("CSV headers must be nonempty and unique.")
        frame = pd.read_csv(io.StringIO(text))
    elif name.endswith(".xlsx"):
        header_frame = pd.read_excel(io.BytesIO(raw), header=None, nrows=1)
        header = header_frame.iloc[0].tolist()
        if any(pd.isna(c) or not str(c).strip() for c in header) or len(set(map(str, header))) != len(header):
            raise ValueError("Spreadsheet headers must be nonempty and unique.")
        frame = pd.read_excel(io.BytesIO(raw))
    else:
        raise ValueError("Upload a UTF-8 CSV or XLSX file.")
    frame.columns = frame.columns.astype(str)
    if len(frame) > 50000 or len(frame.columns) > 500 or not len(frame):
        raise ValueError("Use 1–50,000 rows and at most 500 columns. Prediction sample budgets apply separately.")
    return frame


def export_bundle(bundle):
    """Export reviewed excerpts, not entire retrieved papers or API tokens."""
    result = copy.deepcopy(bundle)
    for source in result["sources"]:
        quotes = list(dict.fromkeys(a["quote"] for m in result["models"] for a in [*m.get("anchors", []), *(a for d in m.get("missing_distributions", []) for a in d.get("anchors", []))]
                                   if a.get("source_id") == source["id"] and isinstance(a.get("quote"), str)))
        quote = source.get("screening_report", {}).get("quote", "")
        if quote and quote in source["text"]:
            quotes.append(quote)
        source["original_text_characters"] = len(source["text"])
        source["text"] = "\n\n".join(quotes)
        source["export_scope"] = "supporting excerpts only; sha256 refers to original source text"
        source.pop("dataset_review", None)  # Excerpt imports require a fresh review.
    return result


def create_app():
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.secret_key = secrets.token_bytes(32)
    app.config.update(MAX_CONTENT_LENGTH=12*1024*1024, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Strict", TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    states, lock = {}, threading.RLock()

    def state():
        with lock:
            now = time.time()
            # Bounded local memory; active jobs are not expired.
            for sid in list(states):
                if now-states[sid]["touched"] > 8*3600 and not states[sid]["busy"]:
                    del states[sid]
            sid = session.get("sid")
            if sid not in states:
                if len(states) >= 32:
                    raise ValueError("Too many local sessions. Restart the app to clear them.")
                sid = secrets.token_urlsafe(24)
                session["sid"] = sid
                states[sid] = {"csrf": secrets.token_urlsafe(24), "provider": None,
                    "context": None, "test": None, "bundle": {"target": {}, "models": [], "sources": []},
                    "research_log": [], "result": None, "busy": False, "cancel": threading.Event(),
                    "job": {"status": "idle", "message": "Ready", "progress": 0}, "touched": now}
            states[sid]["touched"] = now
            return states[sid]

    @app.before_request
    def protect():
        if request.method in {"POST", "PUT", "DELETE"}:
            s = state()
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                return jsonify(error="Cross-origin requests are not allowed."), 403
            if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), s["csrf"]):
                return jsonify(error="Session token missing. Reload this page."), 403

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.errorhandler(Exception)
    def errors(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, ValidationError):
            # Pydantic's normal repr includes input values. Return field locations/messages only.
            message = "; ".join(".".join(map(str, e["loc"])) + ": " + e["msg"] for e in exc.errors())
        elif isinstance(exc, ValueError):
            message = str(exc)
        else:
            message = f"Operation failed ({type(exc).__name__}). No results were substituted."
        return jsonify(error=message), 400

    def idle(s):
        if s["busy"]:
            raise ValueError("Wait for the active operation, or cancel it before editing.")

    def llm(s):
        if not s["provider"]:
            raise ValueError("Connect an LLM first.")
        if s["context"] is None:
            raise ValueError("Upload context rows first.")
        return s["provider"]

    def launch(s, fn):
        with lock:
            idle(s)
            if any(x["busy"] for x in states.values()):
                raise ValueError("Another session is using this local worker. Wait for it to finish.")
            s["busy"] = True
            s["cancel"].clear()
            s["job"] = {"status": "running", "message": "Starting", "progress": 0}
        def progress(message, fraction=0):
            if s["cancel"].is_set():
                raise engine.Cancelled("Cancelled between operations.")
            with lock:
                s["job"].update(message=message, progress=fraction)
        def worker():
            try:
                fn(progress)
                if s["cancel"].is_set():
                    raise engine.Cancelled("Cancelled between operations.")
                with lock:
                    s["job"].update(status="complete", message="Complete", progress=1)
            except Exception as exc:
                if isinstance(exc, ValidationError):
                    msg = "; ".join(str(e["loc"]) + ": " + e["msg"] for e in exc.errors())
                elif isinstance(exc, ValueError):
                    msg = str(exc)
                else:
                    msg = f"Operation failed ({type(exc).__name__}). Check installation, model access, or available memory. No fallback predictions were returned."
                with lock:
                    s["job"].update(status="cancelled" if isinstance(exc, engine.Cancelled) else "error", message=msg)
            finally:
                with lock:
                    s["busy"] = False
        threading.Thread(target=worker, daemon=True).start()
        return jsonify(started=True)

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/state")
    def get_state():
        s = state()
        displayed_bundle = copy.deepcopy(s["bundle"])
        for record in displayed_bundle["sources"]:
            record["screening_flags"] = screening.source_flags(record, displayed_bundle.get("research_policy"))
        return jsonify(csrf=s["csrf"], connected=bool(s["provider"]),
                       profile=profile(s["context"]) if s["context"] is not None else None,
                       test_profile=profile(s["test"]) if s["test"] is not None else None,
                       bundle=displayed_bundle, job=s["job"], has_result=s["result"] is not None,
                       research_log=s["research_log"])

    @app.post("/api/connect")
    def connect():
        s = state(); idle(s)
        body = request.get_json()
        if not body.get("share_summaries"):
            raise ValueError("Approve sending column summaries, target descriptions and source excerpts to this provider.")
        config = {k: body.get(k) for k in ["provider", "endpoint", "model", "token", "send_max_tokens"]}
        providers.complete(config, [{"role": "user", "content": "Reply with Connected."}], "Connection test.", 2048)
        s["provider"] = config
        return jsonify(connected=True)

    @app.get("/api/job")
    def job_status():
        s = state()
        return jsonify(job=s["job"], busy=s["busy"])

    @app.post("/api/disconnect")
    def disconnect():
        s = state(); idle(s); s["provider"] = None
        return jsonify(connected=False)

    @app.post("/api/upload")
    def upload():
        s = state(); idle(s)
        role = request.form.get("role", "context")
        if role not in {"context", "test"}:
            raise ValueError("Unknown dataset role.")
        s[role] = read_frame(request.files["file"])
        s["result"] = None
        for m in s["bundle"]["models"]:
            m["reviewed"] = False
        return jsonify(profile=profile(s[role]))

    @app.post("/api/clear-test")
    def clear_test():
        s = state(); idle(s); s["test"] = None; s["result"] = None
        return jsonify(ok=True)

    @app.post("/api/demo")
    def demo():
        s = state(); idle(s)
        s["context"], s["bundle"] = demo_bundle()
        s["test"] = None; s["result"] = None; s["research_log"] = []
        return jsonify(ok=True)

    @app.post("/api/bundle")
    def bundle():
        s = state(); idle(s)
        b = request.get_json()
        # Migration from v0.1: a browser-saved /api/state response contains
        # full source text, unlike the old excerpt-only evidence download.
        if isinstance(b, dict) and isinstance(b.get("bundle"), dict):
            b = {**b["bundle"], "research_log": b.get("research_log", [])}
        if not isinstance(b, dict) or not isinstance(b.get("models"), list) or not isinstance(b.get("sources"), list):
            raise ValueError("Evidence bundle requires target, models and sources.")
        Target.model_validate(b["target"])
        if len(b["models"]) > 100 or len(b["sources"]) > 100:
            raise ValueError("At most 100 sources/models per bundle.")
        for model in b["models"]:
            validate_draft_shape(model)
        for source in b["sources"]:
            if not all(isinstance(source.get(k), str) for k in ["id", "title", "text", "url", "kind"]):
                raise ValueError("Every source needs string id, title, text, url and kind.")
        if len({source["id"] for source in b["sources"]}) != len(b["sources"]):
            raise ValueError("Source IDs must be unique.")
        s["bundle"] = {k: b[k] for k in ["target", "models", "sources"]}
        s["bundle"]["research_policy"] = screening.policy(b.get("research_policy"))
        s['bundle']['model_policy'] = model_agents.policy(b.get('model_policy'))
        for key in ('agent_log', 'preparation'):
            if key in b:
                s['bundle'][key] = copy.deepcopy(b[key])
        # Imported automated workspaces must be rechecked against current rows.
        if b.get('workflow') == 'automatic':
            s['bundle']['workflow'] = 'automatic'
        if isinstance(b.get("research_log"), list):
            s["research_log"] = b["research_log"]
        s["result"] = None
        return jsonify(ok=True)

    @app.post('/api/model-policy')
    def model_policy():
        s = state(); idle(s)
        s['bundle']['model_policy'] = model_agents.policy(request.get_json())
        s['result'] = None
        return jsonify(ok=True)

    @app.post('/api/suggest-models')
    def suggest_models():
        s = state(); config = llm(s)
        if s['context'] is None:
            raise ValueError('Upload context inputs first.')
        def work(progress):
            updated = model_agents.build(config, s['bundle'], s['context'], s['test'],
                                         profile(s['context']), progress, suggest_only=True)
            with lock:
                s['bundle'] = updated
                s['result'] = None
            if not updated['models']:
                raise ValueError('No model passed review yet. The agent report lists the remaining problems.')
        return launch(s, work)

    @app.post("/api/research-policy")
    def research_policy():
        s = state(); idle(s)
        settings = screening.policy(request.get_json())
        if settings != screening.policy(s["bundle"].get("research_policy")):
            for record in s["bundle"]["sources"]:
                record.pop("dataset_review", None)
                record.pop("screening_report", None)
            for model in s["bundle"]["models"]:
                model["reviewed"] = False
        s["bundle"]["research_policy"] = settings
        s["result"] = None
        return jsonify(ok=True)

    @app.post("/api/source-review")
    def source_review():
        s = state(); idle(s); b = request.get_json()
        record = next((x for x in s["bundle"]["sources"] if x["id"] == b.get("source_id")), None)
        if record is None or b.get("decision") not in ("approve", "independent", "background", "exclude"):
            raise ValueError("Choose a source and a review decision.")
        reason = b.get("reason", "")
        if not isinstance(reason, str) or len(reason) > 4000:
            raise ValueError("Optional notes must be at most 4,000 characters.")
        record["dataset_review"] = {"decision": b["decision"], "reason": reason.strip(),
            "key": screening.review_key(record, s["bundle"].get("research_policy"))}
        for model in s["bundle"]["models"]:
            model["reviewed"] = False
        s["result"] = None
        return jsonify(ok=True)

    @app.post('/api/source-remove')
    def source_remove():
        s = state(); idle(s)
        source_id = request.get_json().get('source_id')
        s['bundle']['sources'] = [x for x in s['bundle']['sources'] if x['id'] != source_id]
        s['bundle']['models'] = [m for m in s['bundle']['models'] if source_id not in
            {a['source_id'] for a in [*(m.get('anchors') or []), *(a for d in m.get('missing_distributions') or [] for a in d.get('anchors') or [])]}]
        s['bundle'].pop('preparation', None)
        s['result'] = None
        return jsonify(ok=True)

    @app.post("/api/screen")
    def screen():
        s = state(); config = llm(s)
        ids = request.get_json().get("source_ids", [])
        selected = [x for x in s["bundle"]["sources"] if x["id"] in ids]
        if not 1 <= len(selected) <= 8:
            raise ValueError("Select 1–8 sources to screen.")
        def work(progress):
            progress("Checking study populations and missing model information")
            reviewed = research.screen_sources(config, copy.deepcopy(selected), s["bundle"].get("research_policy"))
            with lock:
                by_id = {x["id"]: x for x in reviewed}
                s["bundle"]["sources"] = [by_id.get(x["id"], x) for x in s["bundle"]["sources"]]
            progress("Screening complete: inspect exact passages and record your decision")
        return launch(s, work)

    @app.post("/api/source")
    def source():
        s = state(); idle(s)
        if request.files:
            f = request.files["file"]
            text = retrieval.text_from_bytes(f.read(), filename=f.filename)
            record = source_record(f.filename, text, kind="uploaded")
        else:
            b = request.get_json()
            if b.get("url"):
                record = retrieval.retrieve(b["url"])
            else:
                if not str(b.get("text", "")).strip():
                    raise ValueError("Paste source text or enter a public HTTPS URL.")
                record = source_record(b.get("title") or "Manually supplied source", b["text"], kind="manual")
        if record["id"] not in {x["id"] for x in s["bundle"]["sources"]}:
            s["bundle"]["sources"].append(record)
        return jsonify(source=record)

    @app.post("/api/research")
    def do_research():
        s = state(); config = llm(s)
        target = Target.model_validate(s["bundle"]["target"]).model_dump()
        def work(progress):
            report = research.research(config, target, profile(s["context"]), progress, s["bundle"].get("research_policy"))
            for source in report['sources']:
                progress('Screening candidate: ' + source['title'][:90])
                try:
                    research.screen_sources(config, [source], s['bundle'].get('research_policy'))
                except ValueError as error:
                    source['screening_report'] = {'status': 'unknown', 'reason': str(error), 'quote': ''}
            progress("Papers selected. Approve or reject them, then build predictions.")
            with lock:
                known = {x["id"]: x for x in s["bundle"]["sources"]}
                for record in report["sources"]:
                    # Recover a shortened evidence export when the same source
                    # is retrieved again. Never discard a longer local copy.
                    if record["id"] not in known or len(record["text"]) > len(known[record["id"]]["text"]):
                        known[record["id"]] = record
                s["bundle"]["sources"] = list(known.values())
                s["research_log"].append({k:v for k,v in report.items() if k != "sources"})
        return launch(s, work)

    @app.post("/api/extract")
    def do_extract():
        s = state(); config = llm(s)
        if not any(x.get('dataset_review', {}).get('decision') in ('approve', 'independent') for x in s['bundle']['sources']):
            raise ValueError("Approve at least one paper first. No written explanation is needed.")
        def work(progress):
            updated = automation.extract_approved(config, s['bundle'], s['context'], s['test'], profile(s['context']), progress)
            updated['workflow'] = 'automatic'
            with lock:
                s['bundle'] = updated
                s["research_log"].append({"extraction_issues": updated['preparation']['source_issues']})
                s["result"] = None
            if not updated['models']:
                raise ValueError('No usable equations found. See skipped-model and source reasons; add another paper to continue.')
        return launch(s, work)

    @app.post('/api/pipeline')
    def pipeline():
        """Approve sources, then one action performs extraction through output."""
        s = state(); config = llm(s)
        run_config = RunConfig.model_validate(request.get_json()).model_dump()
        if s['context'] is None:
            raise ValueError('Upload context inputs first.')
        def work(progress):
            updated = model_agents.build(config, s['bundle'], s['context'], s['test'], profile(s['context']), progress)
            updated['workflow'] = 'automatic'
            with lock:
                s['bundle'] = updated
            if not updated['models']:
                raise ValueError('No model passed the delegated checks. See the agent report for specific problems.')
            result = engine.run(s['context'], s['test'], copy.deepcopy(updated), run_config, progress, s['cancel'].is_set)
            with lock:
                s['result'] = result
        idle(s); s['result'] = None
        return launch(s, work)

    @app.post("/api/chat")
    def chat():
        s = state(); idle(s)
        message = request.get_json().get("message", "")
        if not isinstance(message, str) or not message.strip() or len(message) > 10000:
            raise ValueError("Enter a question up to 10,000 characters.")
        return jsonify(reply=research.helper(llm(s), message, s["bundle"]["target"],
                                            profile(s["context"]), s["bundle"]["models"]))

    @app.post("/api/validate")
    def validate():
        s = state(); idle(s)
        if s["context"] is None:
            raise ValueError("Upload context rows first.")
        target = Target.model_validate(s["bundle"]["target"])
        config = RunConfig.model_validate(request.get_json())
        if s['bundle'].get('workflow') == 'automatic':
            accepted, report = preparation.prepare(s['bundle'], s['context'], s['test'], config.allow_synthetic, require_query=config.method != 'tabpfn')
            if not accepted:
                raise ValueError('No usable equations remain. Rebuild from approved papers.')
            s['bundle']['models'] = accepted
            s['bundle']['preparation'] = report
            return jsonify(valid=True, models=len(accepted), preparation=report)
        screening.check_models(s['bundle'])
        models = validate_models(s["bundle"]["models"], s["bundle"]["sources"], target,
                                 s["context"], config.allow_synthetic)
        if s["test"] is not None and config.method != "tabpfn":
            validate_models(s["bundle"]["models"], s["bundle"]["sources"], target,
                            s["test"], config.allow_synthetic)
        return jsonify(valid=True, models=len(models))

    @app.post("/api/run")
    def run():
        s = state()
        config = RunConfig.model_validate(request.get_json()).model_dump()
        if s["context"] is None:
            raise ValueError("Upload context rows first.")
        def work(progress):
            if s['bundle'].get('workflow') == 'automatic':
                accepted, report = preparation.prepare(s['bundle'], s['context'], s['test'], config['allow_synthetic'], require_query=config['method'] != 'tabpfn')
                s['bundle']['models'] = accepted
                s['bundle']['preparation'] = report
            result = engine.run(s["context"], s["test"], copy.deepcopy(s["bundle"]), config,
                                progress, s["cancel"].is_set)
            progress("Finalizing results", 1)
            with lock:
                s["result"] = result
        idle(s); s["result"] = None
        return launch(s, work)

    @app.post("/api/cancel")
    def cancel():
        s = state(); s["cancel"].set()
        return jsonify(message="Cancellation requested; the current network/inference call must finish first.")

    @app.get("/api/results")
    def results():
        s = state()
        if s["result"] is None:
            raise ValueError("No completed predictions.")
        r = s["result"]
        return jsonify(rows=r["predictions"].head(200).to_dict(orient="records"), total_rows=len(r["predictions"]), audit=r["audit"])

    @app.get("/api/distribution/<int:row>")
    def distribution(row):
        r = state()["result"]
        if r is None or row not in r['predictions']['row_position'].values:
            raise ValueError("Unknown prediction row.")
        position = int(np.flatnonzero(r['predictions']['row_position'].to_numpy() == row)[0])
        selected = {k: v[:, :, position].ravel() for k,v in r["arrays"].items() if k.endswith("_samples")}
        joined = np.concatenate(list(selected.values()))
        edges = np.histogram_bin_edges(joined, bins=40)
        return jsonify(edges=edges.tolist(), series={k: np.histogram(v, bins=edges)[0].tolist() for k,v in selected.items()})

    @app.get("/api/download/evidence")
    def evidence_download():
        s = state()
        exported = export_bundle(s["bundle"])
        exported["research_log"] = s["research_log"]
        data = json.dumps(exported, indent=2, allow_nan=False).encode()
        return send_file(io.BytesIO(data), mimetype="application/json", as_attachment=True, download_name="evidence.json")

    @app.get("/api/download/workspace")
    def workspace_download():
        """User-controlled local backup, including source text but never tokens/rows."""
        s = state()
        exported = copy.deepcopy(s["bundle"])
        exported["research_log"] = s["research_log"]
        exported["export_scope"] = "Full research workspace: source text included; no API credentials or dataset rows."
        data = json.dumps(exported, indent=2, allow_nan=False).encode()
        return send_file(io.BytesIO(data), mimetype="application/json", as_attachment=True, download_name="research-workspace.json")

    @app.get("/api/download/run")
    def run_download():
        s = state(); r = s["result"]
        if r is None:
            raise ValueError("No completed run to export.")
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("predictions.csv", r["predictions"].to_csv(index=False))
            z.writestr("audit.json", json.dumps(r["audit"], indent=2, allow_nan=False))
            z.writestr("evidence.json", json.dumps(export_bundle(s["bundle"]), indent=2, allow_nan=False))
            z.writestr("research_log.json", json.dumps(s["research_log"], indent=2))
            samples = io.BytesIO(); np.savez_compressed(samples, **r["arrays"])
            z.writestr("predictive_distributions.npz", samples.getvalue())
        data.seek(0)
        return send_file(data, mimetype="application/zip", as_attachment=True, download_name="zeroshot-run.zip")

    return app


def main():
    parser = argparse.ArgumentParser(description="ZeroShot — local research prototype")
    parser.add_argument("--port", type=int, default=7432)
    args = parser.parse_args()
    # Intentionally loopback-only; no debug mode or permissive CORS.
    print(f"Open http://127.0.0.1:{args.port} · Ctrl+C stops the local app")
    create_app().run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
