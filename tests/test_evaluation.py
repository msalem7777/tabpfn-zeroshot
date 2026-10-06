import numpy as np
from zeroshot.evaluate import empirical_crps


def test_distribution_score_matches_pairwise_definition():
    samples=np.array([[1.,4.],[2.,3.],[5.,1.]])
    truth=np.array([3.,2.])
    expected=np.mean(np.abs(samples-truth),axis=0)-.5*np.mean(np.abs(samples[:,None,:]-samples[None,:,:]),axis=(0,1))
    np.testing.assert_allclose(empirical_crps(samples,truth),expected)
