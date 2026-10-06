"""Regression checks for literature-context → fixed-weight TabPFN → mixture.

The recording adapter is a test double, never a substitute in real runs.
Run with python smoke_test.py; no credentials or pretrained model required.
"""
import copy
import numpy as np
import pytest

from zeroshot import engine
from zeroshot.demo import demo_bundle
from zeroshot.preparation import prepare
from zeroshot.schema import EvidenceModel, RunConfig


def inputs():
    frame, bundle = demo_bundle()
    bundle['models'][0]['reviewed'] = True
    config = dict(worlds=5, samples_per_world=16, allow_synthetic=True,
                  assumptions_approved=True)
    return frame, bundle, config


class RecordingAdapter:
    calls = []
    closed = False

    def __init__(self, family, config):
        type(self).calls = []
        type(self).closed = False

    def predict(self, x, y, query, count, rng):
        self.calls.append((x.copy(), y.copy(), query.copy()))
        # A deterministic function of the context labels makes averaging testable.
        mean = np.full(len(query), np.mean(y))
        return mean, np.full(len(query), 4.), rng.normal(mean, 2., (count, len(query)))

    def close(self):
        type(self).closed = True


def test_primary_path_only_evaluates_literature_on_context_and_averages_tabpfn(monkeypatch):
    frame, bundle, config = inputs()
    bundle['models'][0]['predictors'][0]['transform'] = 'log'
    query = frame.iloc[:3].copy()
    query['temperature'] = -100  # Invalid for the equation, valid numeric TabPFN input.
    original = engine.design_matrix
    evaluated = []
    def context_only(data, model):
        assert (data.temperature > 0).all(), 'Literature was applied to test rows'
        evaluated.append(len(data))
        return original(data, model)
    monkeypatch.setattr(engine, 'design_matrix', context_only)
    result = engine.run(frame, query, bundle, config, adapter_factory=RecordingAdapter)
    assert RunConfig().method == 'tabpfn'
    assert evaluated == [len(frame)] * config['worlds']
    assert len(RecordingAdapter.calls) == config['worlds']
    assert RecordingAdapter.closed
    assert not any(c.startswith('direct_') for c in result['predictions'])
    assert 'direct_samples' not in result['arrays']
    means = np.array([y.mean() for _, y, _ in RecordingAdapter.calls])
    assert np.ptp(means) > 0  # Different literature draws generate different labels.
    np.testing.assert_allclose(result['predictions'].tabpfn_mean, means.mean())
    np.testing.assert_allclose(result['predictions'].tabpfn_within_variance, 4.)
    np.testing.assert_allclose(result['predictions'].tabpfn_between_variance, means.var())
    np.testing.assert_allclose(result['predictions'].tabpfn_total_variance, 4. + means.var())
    assert result['audit']['literature_evaluated_on_test'] is False
    assert result['audit']['skipped_rows']['query'] == []
    for (_, labels, _), world in zip(RecordingAdapter.calls, result['audit']['worlds']):
        np.testing.assert_allclose(labels, world['context_labels'])


def test_preparation_keeps_context_usable_model_despite_incompatible_query():
    frame, bundle, _ = inputs()
    bundle['models'][0]['predictors'][0]['transform'] = 'log'
    query = frame.iloc[:2].copy(); query['temperature'] = -1
    models, report = prepare(bundle, frame, query, allow_synthetic=True)
    assert len(models) == 1
    assert 'query' not in report['coverage']
    models, _ = prepare(bundle, frame, query, allow_synthetic=True, require_query=True)
    assert not models


def test_cross_fitting_retains_unsupported_query_and_never_uses_its_label():
    frame, bundle, config = inputs()
    frame.loc[0, 'pressure'] = np.nan
    result = engine.run(frame, None, bundle, config, adapter_factory=RecordingAdapter)
    assert len(result['predictions']) == len(frame)
    assert result['audit']['skipped_rows'] == {'context':[0], 'query':[]}
    for split in result['audit']['splits']:
        assert not set(split['context_positions']) & set(split['prediction_positions'])
        assert 0 not in split['context_positions']
    assert len(RecordingAdapter.calls) == config['worlds'] * 3


def test_empirical_draws_preserve_joint_coefficients_and_paired_residuals():
    _, bundle, _ = inputs()
    raw = bundle['models'][0]
    raw.update(parameter_distribution='empirical', parameter_draws=[[10,1,2],[30,3,6]],
               residual_draws=[1,3], residual_log_sd=0, standard_errors=None,
               covariance=None, transfer_intercept_sd=0)
    model = EvidenceModel.model_validate(raw)
    seen = set()
    rng = np.random.default_rng(42)
    for _ in range(30):
        beta, sigma = engine.world_parameters(model, rng)
        seen.add(tuple(beta))
        assert beta[0] == 10 * beta[1]
        assert beta[2] == 2 * beta[1]
        assert sigma == beta[1]
    assert len(seen) == 2
    bad = copy.deepcopy(raw); bad['parameter_draws'] = [[1,2]]
    with pytest.raises(ValueError):
        EvidenceModel.model_validate(bad)


def test_binary_contexts_are_sampled_labels_not_soft_probabilities():
    frame, bundle, config = inputs()
    target = bundle['target']; target.update(family='logistic', unit='0/1', positive_class='default')
    m = bundle['models'][0]
    m.update(family='logistic', target_unit='0/1', positive_class='default',
             residual_sd=0, residual_log_sd=0, coefficients=[0,0,0], standard_errors=[0,0,0])
    engine.run(frame, frame.iloc[:2], bundle, config, adapter_factory=RecordingAdapter)
    assert all(set(y) == {0,1} for _, y, _ in RecordingAdapter.calls)


def test_regression_uses_sampled_equation_mean_without_extra_residual_noise():
    frame, bundle, config = inputs()
    m = bundle['models'][0]; m.update(standard_errors=[0,0,0], transfer_intercept_sd=0)
    result = engine.run(frame, frame.iloc[:2], bundle, config, adapter_factory=RecordingAdapter)
    expected = 20 + 1.5 * frame.temperature + 4 * frame.pressure
    for _, y, _ in RecordingAdapter.calls:
        np.testing.assert_allclose(y, expected)
    np.testing.assert_allclose(result['predictions'].tabpfn_between_variance, 0, atol=1e-20)
