"""Check statistical invariants against analytic results, not mirrored implementation."""
import copy

import numpy as np
import pytest

from zeroshot.demo import demo_bundle
from zeroshot.distributions import summarize
from zeroshot.engine import Cancelled, run, world_parameters, world_prediction
from zeroshot.evidence import design_matrix, validate_models
from zeroshot.schema import EvidenceModel, Target


def reviewed():
    frame, bundle = demo_bundle()
    bundle["models"][0]["reviewed"] = True
    return frame, bundle


def test_total_variance_and_no_division_by_number_of_worlds():
    means = np.array([[10.], [14.]])
    variances = np.array([[9.], [9.]])
    samples = means[:,None,:]
    a = summarize(means, variances, samples, .9)
    b = summarize(np.repeat(means,20,axis=0),np.repeat(variances,20,axis=0),np.repeat(samples,20,axis=0),.9)
    assert a["mean"][0] == 12
    assert a["within_variance"][0] == 9
    assert a["between_variance"][0] == 4
    assert a["total_variance"][0] == 13
    assert b["total_variance"][0] == 13


def test_source_scale_and_unit_conversion():
    frame,bundle=reviewed(); raw=bundle["models"][0]
    raw["predictors"][0].update(multiplier=2,offset=3,source_center=5,source_scale=4)
    m=EvidenceModel.model_validate(raw)
    d=design_matrix(frame,m)
    np.testing.assert_allclose(d[:,1],(frame.temperature*2+3-5)/4)
    assert not np.isclose(d[:,1].mean(),0)  # no target-population recentering


def test_parameter_draw_is_shared_across_rows_and_joint():
    frame,bundle=reviewed();raw=bundle["models"][0]
    raw.update(standard_errors=None,covariance=[[1,.5,0],[.5,.25,0],[0,0,0]],residual_sd=0)
    m=EvidenceModel.model_validate(raw);rng=np.random.default_rng(9)
    draws=np.array([world_parameters(m,rng)[0] for _ in range(1000)])
    np.testing.assert_allclose(draws[:,1]-1.5,.5*(draws[:,0]-20),atol=1e-7)
    beta,sigma=world_parameters(m,rng)
    repeated=frame.iloc[[0,0,0]]
    _,_,ys=world_prediction(m,design_matrix(repeated,m),beta,sigma,1,rng)
    assert len(set(ys[0])) == 1


def test_linear_moments_match_analytic_evidence_prediction():
    frame,bundle=reviewed()
    config={"method":"direct","worlds":1000,"samples_per_world":16,"assumptions_approved":True,"allow_synthetic":True}
    query=frame.iloc[:1]
    result=run(frame,query,bundle,config)
    row=result["predictions"].iloc[0]
    x=np.r_[1,query.iloc[0].to_numpy()]
    analytic_mean=x @ np.array([20,1.5,4])
    analytic_variance=np.sum((x*np.array([1,.1,.4]))**2)+9
    assert abs(row.direct_mean-analytic_mean)<.5
    assert abs(row.direct_total_variance-analytic_variance)<1
    assert row.direct_within_variance == 9


def test_lognormal_moments_include_jensen_correction():
    frame,bundle=reviewed();raw=bundle["models"][0]
    raw.update(coefficients=[1,0,0],standard_errors=[0,0,0],residual_sd=.3,target_transform="log")
    m=EvidenceModel.model_validate(raw)
    mean,var,_=world_prediction(m,design_matrix(frame.iloc[:1],m),np.array([1,0,0]),.3,16,np.random.default_rng(2))
    assert mean[0] == pytest.approx(np.exp(1+.09/2))
    assert var[0] == pytest.approx(np.expm1(.09)*np.exp(2+.09))


def test_binary_total_variance_and_probability_intervals():
    frame,bundle=reviewed();raw=bundle["models"][0]
    raw.update(family="logistic",positive_class="success",coefficients=[-2,.04,.1],residual_sd=0)
    bundle["target"].update(family="logistic",positive_class="success")
    r=run(frame,None,bundle,{"method":"direct","worlds":16,"allow_synthetic":True,"assumptions_approved":True})
    p=r["predictions"].direct_mean
    np.testing.assert_allclose(r["predictions"].direct_total_variance,p*(1-p),atol=1e-12)
    assert "direct_probability_lower" in r["predictions"]


@pytest.mark.parametrize("mutation,match",[
    (lambda m:m.update(reviewed=False),"review"),
    (lambda m:m.update(covariance=[[1,2,0],[2,1,0],[0,0,1]],standard_errors=None),"positive semidefinite"),
    (lambda m:m.update(standard_errors=None),"covariance"),
    (lambda m:m.update(coefficients=[1,2]),"intercept"),
    (lambda m:m["anchors"][0].update(quote="This quote is invented and absent."),"absent"),
    (lambda m:m.update(target_unit="different"),"differs"),
])
def test_bad_evidence_is_blocked(mutation,match):
    frame,bundle=reviewed();mutation(bundle["models"][0])
    with pytest.raises(ValueError,match=match):
        validate_models(bundle["models"],bundle["sources"],Target.model_validate(bundle["target"]),frame,True)


def test_missing_predictor_and_missing_values_blocked():
    frame,bundle=reviewed();m=EvidenceModel.model_validate(bundle["models"][0])
    with pytest.raises(ValueError,match="missing predictor"):
        design_matrix(frame.drop(columns="pressure"),m)
    frame.loc[0,"temperature"]=np.nan
    with pytest.raises(ValueError,match="missing, nonnumeric"):
        design_matrix(frame,m)


class RecordingAdapter:
    calls=[]
    def __init__(self,family,config): pass
    def predict(self,xc,yc,xt,count,rng):
        self.calls.append((xc.index.tolist(),xt.index.tolist()))
        n=len(xt)
        return np.ones(n),np.ones(n),rng.normal(1,1,(count,n))
    def close(self): pass


def test_cross_fitting_and_direct_path_are_independent_of_adapter():
    frame,bundle=reviewed();config={"worlds":3,"samples_per_world":16,"allow_synthetic":True,"assumptions_approved":True}
    RecordingAdapter.calls=[]
    both=run(frame,None,bundle,{**config,"method":"both"},adapter_factory=RecordingAdapter)
    direct=run(frame,None,bundle,{**config,"method":"direct"})
    np.testing.assert_array_equal(both["arrays"]["direct_samples"],direct["arrays"]["direct_samples"])
    assert len(RecordingAdapter.calls)==9
    assert all(not set(a)&set(b) for a,b in RecordingAdapter.calls)
    assert both["audit"]["prediction_mode"]=="cross_fitted"


def test_no_target_labels_and_no_unapproved_run():
    frame,bundle=reviewed();frame["yield"]=1
    with pytest.raises(ValueError,match="target must be absent"):
        run(frame,None,bundle,{"method":"direct","assumptions_approved":True})
    with pytest.raises(ValueError,match="Approve"):
        run(frame.drop(columns="yield"),None,bundle,{"method":"direct"})


def test_cancellation_and_memory_budget():
    frame,bundle=reviewed();config={"method":"direct","allow_synthetic":True,"assumptions_approved":True}
    with pytest.raises(Cancelled):
        run(frame,None,bundle,config,cancelled=lambda:True)
    with pytest.raises(ValueError,match="budget"):
        run(frame,None,bundle,{**config,"worlds":1000,"samples_per_world":2048})

