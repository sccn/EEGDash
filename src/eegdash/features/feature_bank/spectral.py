import numpy as np
import numba as nb
from scipy.signal import welch

from ..extractors import FeatureExtractor
from ..decorators import FeaturePredecessor, univariate_feature


__all__ = [
    "SpectralFeatureExtractor",
    "NormalizedSpectralFeatureExtractor",
    "DBSpectralFeatureExtractor",
    "spectral_root_total_power",
    "spectral_moment",
    "spectral_entropy",
    "spectral_edge",
    "spectral_slope",
    "spectral_bands_power",
    "spectral_hjorth_activity",
    "spectral_hjorth_mobility",
    "spectral_hjorth_complexity",
]


class SpectralFeatureExtractor(FeatureExtractor):
    def preprocess(self, x, **kwargs):
        f_min = kwargs.pop("f_min") if "f_min" in kwargs else None
        f_max = kwargs.pop("f_max") if "f_max" in kwargs else None
        kwargs["axis"] = -1
        f, p = welch(x, **kwargs)
        if f_min is not None or f_max is not None:
            f_min_idx = f > f_min if f_min is not None else True
            f_max_idx = f < f_max if f_max is not None else True
            idx = np.logical_and(f_min_idx, f_max_idx)
            f = f[idx]
            p = p[..., idx]
        return f, p


@FeaturePredecessor(SpectralFeatureExtractor)
class NormalizedSpectralFeatureExtractor(FeatureExtractor):
    def preprocess(self, *x):
        return (*x[:-1], x[-1] / x[-1].sum(axis=-1, keepdims=True))


@FeaturePredecessor(SpectralFeatureExtractor)
class DBSpectralFeatureExtractor(FeatureExtractor):
    def preprocess(self, *x, eps=1e-15):
        return (*x[:-1], 10 * np.log10(x[-1] + eps))


@FeaturePredecessor(SpectralFeatureExtractor)
@univariate_feature
def spectral_root_total_power(f, p):
    return np.sqrt(p.sum(axis=-1))


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_moment(f, p):
    return np.sum(f * p, axis=-1)


@FeaturePredecessor(SpectralFeatureExtractor)
@univariate_feature
def spectral_hjorth_activity(f, p):
    return np.sum(p, axis=-1)


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_hjorth_mobility(f, p):
    return np.sqrt(np.sum(np.power(f, 2) * p, axis=-1))


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_hjorth_complexity(f, p):
    return np.sqrt(np.sum(np.power(f, 4) * p, axis=-1))


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
def spectral_entropy(f, p):
    idx = p > 0
    plogp = np.zeros_like(p)
    plogp[idx] = p[idx] * np.log(p[idx])
    return -np.sum(plogp, axis=-1)


@FeaturePredecessor(NormalizedSpectralFeatureExtractor)
@univariate_feature
@nb.njit(cache=True, fastmath=True)
def spectral_edge(f, p, edge=0.9):
    se = np.empty(p.shape[:-1])
    for i in np.ndindex(p.shape[:-1]):
        se[i] = f[np.searchsorted(np.cumsum(p[i]), edge)]
    return se


@FeaturePredecessor(DBSpectralFeatureExtractor)
@univariate_feature
def spectral_slope(f, p):
    log_f = np.vstack((np.log(f), np.ones(f.shape[0]))).T
    r = np.linalg.lstsq(log_f, p.reshape(-1, p.shape[-1]).T)[0]
    r = r.reshape(2, *p.shape[:-1])
    return {"exp": r[0], "int": r[1]}


@FeaturePredecessor(
    SpectralFeatureExtractor,
    NormalizedSpectralFeatureExtractor,
    DBSpectralFeatureExtractor,
)
@univariate_feature
def spectral_bands_power(
    f,
    p,
    bands={
        "delta": (1, 4.5),
        "theta": (4.5, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
    },
):
    bands_power = dict()
    for k, v in bands.items():
        assert isinstance(k, str)
        assert isinstance(v, tuple)
        assert len(v) == 2
        mask = np.logical_and(f > v[0], f < v[1])
        power = p[..., mask].sum(axis=-1)
        bands_power[k] = power
    return bands_power
