"""Optional real-browser smoke test. Starts/stops only its own temporary test server.

Install: python -m pip install playwright
Browser: python -m playwright install chromium
Run: python tests/browser_smoke.py
"""
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright, expect
from werkzeug.serving import make_server

from zeroshot.web import create_app
from zeroshot import engine
import numpy as np


def main():
    # Test only: exercise the actual web pipeline with a recording stand-in.
    # This does not claim a real pretrained TabPFN inference succeeded.
    original_run = engine.run
    class BrowserAdapter:
        def __init__(self, *args): pass
        def close(self): pass
        def predict(self, x, y, query, count, rng):
            mean = np.full(len(query), np.mean(y))
            return mean, np.ones(len(query)), rng.normal(mean, 1, (count, len(query)))
    def test_run(*args, **kwargs):
        return original_run(*args, **kwargs, adapter_factory=BrowserAdapter)
    engine.run = test_run
    server=make_server("127.0.0.1",0,create_app(),threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
            page=browser.new_page(viewport={"width":1512,"height":1100})
            page.on("pageerror",lambda exc:errors.append(str(exc)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            expect(page.locator("#workspaceSummary")).to_contain_text("Upload")
            page.locator("#demo").click()
            page.get_by_text("Optional: inspect or edit extracted equations",exact=True).click()
            page.locator(".model-card").wait_for()
            page.locator(".model-card .check input").check()
            page.locator(".model-card.approved").wait_for()
            page.locator('[data-tab="run"]').click()
            page.locator("#worlds").fill("8")
            page.locator("#samples").fill("32")
            page.locator("#validate").click()
            expect(page.locator("#notice")).to_contain_text("passed structural")
            page.locator("#run").click()
            page.locator("#resultContent").wait_for(state="visible",timeout=30000)
            page.locator("#distributionChart svg").wait_for()
            assert page.locator("#resultTable tbody tr").count()==60
            page.locator("#resultTable tbody tr").nth(3).click()
            expect(page.locator("#distTitle")).to_contain_text("row 3")
            with page.expect_download() as download:
                page.locator("#downloadRun").click()
            assert download.value.suggested_filename=="zeroshot-run.zip"
            output=Path(__file__).resolve().parents[1]/"qa"
            output.mkdir(exist_ok=True)
            page.screenshot(path=str(output/"results-desktop.png"),full_page=True)
            page.locator('[data-tab="run"]').click()
            page.locator('#method').select_option('tabpfn')
            page.locator('#run').click()
            expect(page.locator('#resultTable')).to_contain_text('Zero-shot mean (TabPFN)',timeout=30000)
            assert 'Direct mean' not in page.locator('#resultTable').inner_text()
            page.locator('#distributionChart svg').wait_for()
            page.screenshot(path=str(output/'tabpfn-mixture-desktop.png'),full_page=True)
            page.locator('[data-tab="evidence"]').click()
            page.get_by_text("Advanced: dataset exclusions and source screening",exact=True).click()
            # Exercise the new benchmark-review controls without a paid LLM.
            page.locator("#taiwanPolicy").click()
            page.locator("#savePolicy").click()
            expect(page.locator("#notice")).to_contain_text("Exclusions saved")
            page.get_by_role("button",name="Approve",exact=True).first.click()
            expect(page.locator("#sources")).to_contain_text("Approved")
            with page.expect_download() as backup:
                page.get_by_role("link",name="Download research workspace (includes source text)").click()
            assert backup.value.suggested_filename=="research-workspace.json"
            page.screenshot(path=str(output/"evidence-desktop.png"),full_page=True)
            page.set_viewport_size({"width":390,"height":844})
            page.locator('[data-tab="setup"]').click()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
            page.screenshot(path=str(output/"setup-mobile.png"),full_page=True)
            # A new workspace must not show the previous workspace's predictions.
            page.locator("#demo").click()
            page.locator(".model-card").wait_for()
            page.locator('[data-tab="results"]').click()
            assert page.locator("#resultContent").is_hidden()
            assert page.locator("#resultEmpty").is_visible()
            assert not errors,errors
            browser.close()
        print("Browser smoke passed: demo, approval, validation, run, chart, row selection, download, mobile layout.")
    finally:
        engine.run = original_run
        server.shutdown();server.server_close();thread.join(timeout=5)


if __name__ == "__main__":
    main()
