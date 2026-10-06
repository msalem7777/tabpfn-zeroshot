"""Offline tests for delegated modeling; no paid calls or real TabPFN jobs."""
import copy
import numpy as np
import pytest

from zeroshot import model_agents, screening, providers, research
from zeroshot.demo import demo_bundle
from zeroshot.evidence import profile
from zeroshot.engine import run
from zeroshot.web import create_app


def fixture():
    frame, bundle = demo_bundle()
    draft = copy.deepcopy(bundle['models'][0])
    draft.update(anchors=[], synthetic=False, reviewed=False,
                 proposal_basis='Proposed illustrative coefficient means and standard errors; not study estimates.')
    bundle['sources'] = []
    bundle['models'] = []
    bundle['model_policy'] = {'mode':'delegate', 'instructions':'Use the supplied illustrative model.'}
    return frame, bundle, draft


def mock_roles(monkeypatch, draft, decisions=(True,)):
    calls = []
    monkeypatch.setattr(research, 'research', lambda *a: {'sources':[], 'errors':[]})
    def reply(config, system, payload):
        calls.append((system, copy.deepcopy(payload)))
        if 'model proposer' in system:
            return {'models':[copy.deepcopy(draft)], 'issues':[]}
        n = sum('critical model reviewer' in s for s,_ in calls)-1
        accept = decisions[min(n,len(decisions)-1)]
        return {'reviews':[{'id':m['id'], 'accept':accept,
                            'reason':'Checked instructions, units and explicit assumptions.'}
                           for m in payload['models']]}
    monkeypatch.setattr(providers, 'json_reply', reply)
    return calls


def test_propose_review_prepare_and_audit_assumptions(monkeypatch):
    frame, bundle, draft = fixture(); calls = mock_roles(monkeypatch, draft)
    updated = model_agents.build({}, bundle, frame, None, profile(frame))
    assert len(calls) == 2
    assert updated['preparation']['accepted'] == 1
    model = updated['models'][0]
    assert model['model_origin'] == 'llm_proposed'
    assert 'UNSOURCED LLM PRIOR' in model['proposal_basis']
    assert calls[0][1]['user_model_instructions'] == bundle['model_policy']['instructions']
    assert 'rows' not in calls[0][1]  # Profile has counts, no row records.
    class Adapter:
        def __init__(self,*a): pass
        def close(self): pass
        def predict(self,x,y,q,count,rng):
            return np.zeros(len(q)),np.ones(len(q)),rng.normal(size=(count,len(q)))
    result=run(frame,frame.iloc[:2],updated,{'worlds':2,'samples_per_world':16,'assumptions_approved':True},adapter_factory=Adapter)
    assert result['audit']['warnings'][0].startswith('ASSUMPTION-BASED')
    assert result['audit']['agent_log']
    assert 'direct_samples' not in result['arrays']


def test_reviewer_rejection_triggers_one_bounded_repair(monkeypatch):
    frame,bundle,draft=fixture();calls=mock_roles(monkeypatch,draft,(False,True))
    updated=model_agents.build({},bundle,frame,None,profile(frame))
    assert len(calls)==4
    assert calls[2][1]['previous_problems']
    assert updated['models'][0]['id']=='proposal-2-1'


def test_reviewer_rejection_never_runs_unreviewed_proposals(monkeypatch):
    frame,bundle,draft=fixture();calls=mock_roles(monkeypatch,draft,(False,))
    updated=model_agents.build({},bundle,frame,None,profile(frame))
    assert len(calls)==4
    assert updated['models']==[]
    assert updated['preparation']['source_issues']


def test_rejected_sources_never_reach_proposer_and_no_repeat_search(monkeypatch):
    frame,bundle,draft=fixture();calls=mock_roles(monkeypatch,draft)
    bundle['sources']=[{'id':'reject','title':'Rejected','text':'DO NOT SEND THIS',
                        'url':'','kind':'manual','dataset_review':{'decision':'exclude'}}]
    monkeypatch.setattr(research,'research',lambda *a:pytest.fail('Must reuse saved papers'))
    model_agents.build({},bundle,frame,None,profile(frame))
    assert all(payload['sources']==[] for _,payload in calls)


def test_invalid_uncertainty_is_not_saved_as_runnable(monkeypatch):
    frame,bundle,draft=fixture();draft['standard_errors']=[-1,-1,-1]
    mock_roles(monkeypatch,draft)
    updated=model_agents.build({},bundle,frame,None,profile(frame))
    assert updated['models']==[]
    assert updated['preparation']['skipped']


def test_published_mode_cannot_execute_assumed_models(monkeypatch):
    frame,bundle,draft=fixture();mock_roles(monkeypatch,draft)
    updated=model_agents.build({},bundle,frame,None,profile(frame))
    updated['model_policy']['mode']='published'
    with pytest.raises(ValueError,match='delegated'):
        screening.check_models(updated)


def test_agent_log_and_instructions_survive_workspace_roundtrip(monkeypatch):
    frame,bundle,draft=fixture();mock_roles(monkeypatch,draft)
    saved=model_agents.build({},bundle,frame,None,profile(frame))
    client=create_app().test_client()
    headers={'X-CSRF-Token':client.get('/api/state').json['csrf']}
    assert client.post('/api/bundle',json=saved,headers=headers).status_code==200
    restored=client.get('/api/download/workspace').json
    for key in ('models','sources','model_policy','agent_log','preparation'):
        assert restored[key]==saved[key]


def test_exact_user_model_survives_automatic_build(monkeypatch):
    frame,bundle,draft=fixture();mock_roles(monkeypatch,draft)
    manual=copy.deepcopy(draft)
    manual.update(id='my-equation',model_origin='user_specified',
                  proposal_basis='Exact user inputs for a test scenario.',coefficients=[9,2,4])
    bundle['models']=[manual]
    updated=model_agents.build({},bundle,frame,None,profile(frame))
    kept=next(m for m in updated['models'] if m['id']=='my-equation')
    assert kept['coefficients']==[9,2,4]
