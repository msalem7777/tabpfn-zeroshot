"""Analytic and workflow checks; no network, credentials or real TabPFN jobs."""
import copy
import io
import time

import numpy as np
import pytest

from zeroshot import automation, providers, research, screening
from zeroshot.demo import demo_bundle
from zeroshot.engine import run
from zeroshot.marginalization import eligibility, sample_frame
from zeroshot.preparation import prepare
from zeroshot.schema import EvidenceModel
from zeroshot.web import create_app


def fixture():
    frame, bundle = demo_bundle()
    model = bundle['models'][0]
    model.update(reviewed=True, standard_errors=[0, 0, 0])
    quote = 'Fictional pressure is normally distributed with mean 2 and standard deviation 0.5.'
    bundle['sources'][0]['text'] += ' ' + quote
    model['missing_distributions'] = [{'column': 'pressure', 'family': 'normal', 'mean': 2,
        'sd': .5, 'basis': 'Fictional normal distribution for test only',
        'anchors': [{'source_id': bundle['sources'][0]['id'], 'quote': quote, 'supports': 'Pressure distribution'}]}]
    return frame, bundle


def test_missing_predictor_moments_follow_complete_equation():
    frame, bundle = fixture()
    frame = frame.drop(columns='pressure')
    result = run(frame, frame.iloc[:1], bundle, {'method': 'direct', 'worlds': 1000,
        'samples_per_world': 16, 'allow_synthetic': True, 'assumptions_approved': True})
    row = result['predictions'].iloc[0]
    # y=20+1.5*temperature+4*pressure+noise; Var(y)=16*.25+9=13.
    expected = 20 + 1.5 * frame.iloc[0].temperature + 4 * 2
    assert row.direct_mean == pytest.approx(expected, abs=.2)
    assert row.direct_total_variance == pytest.approx(13, abs=.6)
    assert row.direct_within_variance == 9


def test_conditional_distribution_uses_observed_row_and_reuses_terms():
    frame, bundle = fixture()
    distribution = bundle['models'][0]['missing_distributions'][0]
    distribution.update(mean=1, sd=0, conditional_mean_coefficients={'temperature': .2})
    parsed = EvidenceModel.model_validate(bundle['models'][0])
    sampled = sample_frame(frame.drop(columns='pressure'), parsed, np.random.default_rng(1))
    np.testing.assert_allclose(sampled.pressure, 1 + .2 * frame.temperature)
    missing_both = frame.drop(columns=['pressure', 'temperature'])
    mask, reasons = eligibility(missing_both, parsed)
    assert not mask.any()
    assert 'conditioning variable' in str(reasons)


def test_unavailable_joint_distribution_is_not_assumed_independent():
    frame, bundle = fixture()
    d = copy.deepcopy(bundle['models'][0]['missing_distributions'][0]); d['column'] = 'temperature'
    bundle['models'][0]['missing_distributions'].append(d)
    mask, reasons = eligibility(frame.drop(columns=['pressure', 'temperature']), EvidenceModel.model_validate(bundle['models'][0]))
    assert not mask.any() and 'independence not established' in str(reasons)


def test_missing_rows_are_reported_without_renumbering():
    frame, bundle = fixture()
    bundle['models'][0]['missing_distributions'] = []
    query = frame.iloc[:3].copy(); query.iloc[0, 1] = np.nan
    result = run(frame, query, bundle, {'method': 'direct', 'worlds': 2,
        'samples_per_world': 16, 'allow_synthetic': True, 'assumptions_approved': True})
    assert result['predictions'].row_position.tolist() == [1, 2]
    assert result['audit']['skipped_rows']['query'] == [0]


def test_source_provided_reduced_equation_covers_rows_missing_full_predictor():
    frame, bundle = fixture(); full = bundle['models'][0]; full['missing_distributions'] = []
    reduced = copy.deepcopy(full)
    reduced.update(id='published-reduced', predictors=full['predictors'][:1], coefficients=[10, 2], standard_errors=[0, 0])
    bundle['models'].append(reduced)
    query = frame.iloc[:2].copy(); query.iloc[0, 1] = np.nan
    result = run(frame, query, bundle, {'method': 'direct', 'worlds': 8,
        'samples_per_world': 16, 'allow_synthetic': True, 'assumptions_approved': True})
    assert result['predictions'].row_position.tolist() == [0, 1]
    assert result['predictions'].iloc[0].direct_mean == pytest.approx(10 + 2 * query.iloc[0].temperature)
    assert all(w['query_model_indices'][0] == 1 for w in result['audit']['worlds'])


def test_unquoted_missing_distribution_is_skipped():
    frame, bundle = fixture()
    bundle['models'][0]['missing_distributions'][0]['anchors'][0]['quote'] = 'Invented unsupported distribution passage'
    models, report = prepare(bundle, frame.drop(columns='pressure'), allow_synthetic=True)
    assert not models and 'quote is absent' in report['skipped'][0]['reason']


def test_extra_observed_columns_reach_tabpfn_without_changing_direct_predictions():
    frame, bundle = fixture(); frame['extra'] = np.arange(len(frame))
    seen = []
    class Adapter:
        def __init__(self, *a): pass
        def close(self): pass
        def predict(self, x, y, q, count, rng):
            seen.append(list(x.columns)); n = len(q)
            return np.zeros(n), np.ones(n), rng.normal(size=(count, n))
    config = {'worlds': 2, 'samples_per_world': 16, 'allow_synthetic': True, 'assumptions_approved': True}
    result = run(frame, frame.iloc[:2], bundle, {**config, 'method':'both'}, adapter_factory=Adapter)
    direct = run(frame.drop(columns='extra'), frame.iloc[:2].drop(columns='extra'), bundle, {**config, 'method': 'direct'})
    assert all('extra' in c for c in seen)
    np.testing.assert_array_equal(result['arrays']['direct_samples'], direct['arrays']['direct_samples'])


def test_json_recovery_and_bounded_retry(monkeypatch):
    assert providers.parse_object('Here is the result:\n```json\n{"models": []}\n```') == {'models': []}
    with pytest.raises(ValueError):
        providers.parse_object('{"models": [{"id": "inner"}')
    calls = []
    def reply(*a, **k):
        calls.append(1)
        return '{"models":' if len(calls) < 3 else '{"models": []}'
    monkeypatch.setattr(providers, 'complete', reply)
    assert providers.json_reply({}, 'system', {}) == {'models': []}
    assert len(calls) == 3


def test_simple_approval_and_full_pipeline_without_model_review(monkeypatch):
    frame, bundle = fixture()
    # This fixture stands in for a source response; no real scientific claim.
    source = bundle['sources'][0]; source['kind'] = 'manual'
    draft = bundle['models'].pop(); draft['synthetic'] = False; draft['reviewed'] = False
    app = create_app(); app.config['TESTING'] = True; client = app.test_client()
    headers = {'X-CSRF-Token': client.get('/api/state').json['csrf']}
    monkeypatch.setattr(providers, 'complete', lambda *a, **k: 'Connected')
    client.post('/api/connect', headers=headers, json={'provider':'compatible','model':'test','share_summaries':True})
    client.post('/api/upload', headers=headers, data={'role':'context','file':(io.BytesIO(frame.to_csv(index=False).encode()), 'context.csv')})
    client.post('/api/bundle', headers=headers, json=bundle)
    response = client.post('/api/source-review', headers=headers, json={'source_id':source['id'],'decision':'approve'})
    assert response.status_code == 200
    monkeypatch.setattr(providers, 'json_reply', lambda *a, **k: {'models':[copy.deepcopy(draft)],'issues':[]})
    response = client.post('/api/pipeline', headers=headers, json={'method':'direct','worlds':2,'samples_per_world':16,'assumptions_approved':True})
    assert response.status_code == 200
    for _ in range(100):
        if not client.get('/api/job').json['busy']: break
        time.sleep(.01)
    state = client.get('/api/state').json
    assert state['has_result'], state['job']
    assert state['bundle']['workflow'] == 'automatic'
    assert state['bundle']['preparation']['accepted'] == 1
    assert client.get('/api/download/run').status_code == 200


def test_approved_benchmark_source_still_needs_supported_provenance():
    _, bundle = fixture(); source = bundle['sources'][0]
    settings = {'excluded_datasets':['Taiwan credit card']}
    source['dataset_review'] = {'decision':'approve','key':screening.review_key(source, settings)}
    with pytest.raises(ValueError, match='Automatic screening'):
        screening.require_source_review(source, settings)


def test_direct_path_does_not_require_unrelated_query_columns():
    frame, bundle = fixture()
    context = frame.assign(unrelated=1)
    result = run(context, frame.iloc[:2], bundle, {'method':'direct', 'worlds':2,
        'samples_per_world':16, 'allow_synthetic':True, 'assumptions_approved':True})
    assert len(result['predictions']) == 2


def test_fixed_missing_distribution_domain_is_checked_before_running():
    frame, bundle = fixture()
    raw = bundle['models'][0]
    raw['predictors'][1]['transform'] = 'log'
    raw['missing_distributions'][0].update(mean=-1, sd=0)
    mask, reasons = eligibility(frame.drop(columns='pressure'), EvidenceModel.model_validate(raw))
    assert not mask.any()
    assert 'invalid transformation' in reasons[0][0]
