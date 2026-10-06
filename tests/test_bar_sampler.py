"""Exercise tail sampling without installing Torch or downloading model weights."""
import numpy as np
from scipy.stats import norm

from zeroshot.distributions import bar_samples


class ArrayTensor:
    def __init__(self,array): self.value=np.asarray(array)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def __getitem__(self,key): return ArrayTensor(self.value[key])
    def __float__(self): return float(self.value)
    def softmax(self,axis):
        ex=np.exp(self.value-self.value.max(axis=axis,keepdims=True))
        return ArrayTensor(ex/ex.sum(axis=axis,keepdims=True))


class Tail:
    def __init__(self,width): self.scale=ArrayTensor(float(width)/norm.ppf(.75))


class Criterion:
    borders=ArrayTensor([-1.,0.,1.,2.])
    bucket_widths=ArrayTensor([1.,1.,1.])
    halfnormal_with_p_weight_before=staticmethod(Tail)


def test_full_support_sampler_keeps_tails_and_matches_known_moments():
    logits=ArrayTensor(np.log([[.25,.5,.25]]))
    draws=bar_samples(Criterion(),logits,200000,np.random.default_rng(10))[:,0]
    sigma=1/norm.ppf(.75)
    half_mean=sigma*np.sqrt(2/np.pi)
    # Left tail is -|N(0,sigma)|; right tail is 1+|N(0,sigma)|.
    mean=.25*(-half_mean)+.5*.5+.25*(1+half_mean)
    second=.25*sigma**2+.5/3+.25*(1+2*half_mean+sigma**2)
    assert draws.min() < -1 and draws.max() > 2
    assert abs(draws.mean()-mean)<.015
    assert abs(draws.var()-(second-mean**2))<.04
