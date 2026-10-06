"""Browser check of delegated UI with explicit offline LLM/TabPFN doubles.

Run: python tests/agent_browser_smoke.py (Playwright/Chromium required).
Starts and stops only its own temporary test server. No real API/inference calls.
"""
import copy
import threading
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright, expect
from werkzeug.serving import make_server
from zeroshot import engine, providers
from zeroshot.demo import demo_bundle
from zeroshot.web import create_app


def main():
    frame,bundle=demo_bundle()
    draft=copy.deepcopy(bundle['models'][0]);draft.update(anchors=[],synthetic=False,reviewed=False,
        proposal_basis='Fictional numbers for software testing, not scientific evidence.')
    bundle['models']=[]
    bundle['sources'][0]['kind']='manual'
    bundle['model_policy']={'mode':'delegate','instructions':'Use the illustrative test equation.'}
    originals=(providers.complete,providers.json_reply,engine.run)
    providers.complete=lambda *a,**k:'Connected'
    def json_reply(config,system,payload):
        if 'model proposer' in system:return {'models':[copy.deepcopy(draft)],'issues':[]}
        return {'reviews':[{'id':m['id'],'accept':True,'reason':'Test reviewer checked the illustrative assumptions.'} for m in payload['models']]}
    providers.json_reply=json_reply
    class Adapter:
        def __init__(self,*a):pass
        def close(self):pass
        def predict(self,x,y,q,n,rng):
            mean=np.full(len(q),y.mean());return mean,np.ones(len(q)),rng.normal(mean,1,(n,len(q)))
    engine.run=lambda *a,**k:originals[2](*a,**k,adapter_factory=Adapter)
    server=make_server('127.0.0.1',0,create_app(),threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1400,'height':1000})
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}')
            csrf=page.request.get(f'http://127.0.0.1:{server.server_port}/api/state').json()['csrf']
            response=page.request.post(f'http://127.0.0.1:{server.server_port}/api/bundle',
                data=bundle,headers={'X-CSRF-Token':csrf})
            assert response.ok
            page.reload()
            page.locator('#contextFile').set_input_files({'name':'context.csv','mimeType':'text/csv','buffer':frame.to_csv(index=False).encode()})
            expect(page.locator('#contextInfo')).to_contain_text('60 rows')
            page.locator('#showConnect').click();page.locator('#llmModel').fill('offline-test')
            page.locator('#shareSummaries').check();page.get_by_role('button',name='Test & connect',exact=True).click()
            expect(page.locator('#connection')).to_have_text('LLM connected')
            page.locator('[data-tab="run"]').click();page.locator('#worlds').fill('2');page.locator('#samples').fill('16')
            page.locator('[data-tab="evidence"]').click()
            expect(page.locator('#modelInstructions')).to_have_value('Use the illustrative test equation.')
            page.locator('#pipeline').click()
            expect(page.locator('#resultTable')).to_contain_text('Zero-shot mean (TabPFN)',timeout=30000)
            expect(page.locator('#resultWarnings')).to_contain_text('ASSUMPTION-BASED MODELS')
            page.locator('[data-tab="evidence"]').click()
            page.get_by_text('Optional: inspect or edit extracted equations',exact=True).click()
            expect(page.locator('#models')).to_contain_text('LLM-PROPOSED MODEL')
            out=Path(__file__).resolve().parents[1]/'qa';out.mkdir(exist_ok=True)
            page.screenshot(path=str(out/'delegated-models-desktop.png'))
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.screenshot(path=str(out/'delegated-models-mobile.png'))
            assert not errors,errors
            browser.close()
        print('Delegated browser workflow passed (mock LLM and TabPFN).')
    finally:
        providers.complete,providers.json_reply,engine.run=originals
        server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':main()
