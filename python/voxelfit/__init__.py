"""Python front end for VoxelFit.jl: fast voxel-wise nonlinear least-squares fitting of MRI data.

    import numpy as np
    import voxelfit

    te = np.arange(10, 330, 10.0)                       # echo times (ms)
    res = voxelfit.fit("MonoExp", te, data, mask=mask)  # data: (..., n_echoes)
    t2 = res["T2"]                                      # same spatial shape as data[..., 0]

Julia runs in-process through juliacall. Data and outputs are shared with Julia without
copying: inputs are read in place (after at most one conversion to a C-contiguous array with
the observation axis last) and Julia writes the parameter maps straight into NumPy arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["fit", "Expr", "FitResult", "models", "STATUS", "julia"]

STATUS = {
    0: "not fitted (outside mask)",
    1: "converged (step)",
    2: "converged (cost)",
    3: "converged (gradient)",
    4: "max iterations",
    5: "failed",
    6: "invalid data",
}

_JL = None


def julia():
    """Return juliacall's Main module with VoxelFit loaded (starts Julia on first use)."""
    global _JL
    if _JL is None:
        # Must be set before juliacall starts Julia. Signal handling is required for threads.
        os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
        os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
        from juliacall import Main as jl

        jl.seval("using VoxelFit")
        _JL = jl
    return _JL


@dataclass
class Expr:
    """A custom model written as a Julia expression.

    Expr("S0 * exp(-TE / T2) + C", params=["S0", "T2", "C"], x="TE",
         p0=[1000, 50, 0], lower=[0, 0, 0])

    `x` names the acquisition variable (a list of names if each observation has several
    settings and `x` is passed to `fit` as an (n_obs, k) array). `covariates` names per-voxel
    maps passed to `fit(..., covariates=...)`. The expression is compiled as Julia code.
    """

    expr: str
    params: Sequence[str]
    x: Union[str, Sequence[str]] = "x"
    covariates: Sequence[str] = ()
    p0: Optional[Sequence[float]] = None
    lower: Optional[Sequence[float]] = None
    upper: Optional[Sequence[float]] = None


@dataclass
class FitResult:
    names: list
    params: dict
    sse: np.ndarray
    iterations: np.ndarray
    status: np.ndarray
    stderr: Optional[dict] = None
    _stack: np.ndarray = field(default=None, repr=False)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.params[name]

    def keys(self):
        return list(self.names)

    @property
    def converged(self) -> np.ndarray:
        return (self.status >= 1) & (self.status <= 3)

    def __repr__(self):
        fitted = int(np.count_nonzero((self.status != 0) & (self.status != 6)))
        return (f"FitResult(shape={self.sse.shape}, params={self.names}, fitted={fitted}, "
                f"converged={int(self.converged.sum())})")


def _wrap(jl, a: np.ndarray):
    """Julia Array viewing a C-contiguous NumPy array (axes reversed), no copy."""
    assert a.flags.c_contiguous
    return jl.VoxelFit.wrap_array(a.ctypes.data, a.dtype.str, tuple(int(s) for s in a.shape))


def _strings(jl, xs):
    import juliacall

    return juliacall.convert(jl.Vector[jl.String], [str(s) for s in xs])


def _model(jl, model):
    if isinstance(model, Expr):
        x = model.x if isinstance(model.x, str) else jl.Tuple(_strings(jl, model.x))
        f = lambda v: None if v is None else np.asarray(v, dtype=np.float64)
        return jl.VoxelFit.expr_model(model.expr, params=_strings(jl, model.params), x=x,
                                      covariates=_strings(jl, model.covariates),
                                      p0=f(model.p0), lower=f(model.lower), upper=f(model.upper))
    if isinstance(model, str):
        return jl.VoxelFit.parse_model(model)
    return model  # assume an already-constructed Julia model


def fit(model, x, data, *, mask=None, axis: int = -1, init=None, lower=None, upper=None,
        covariates=None, weights=None, precision: str = "float64", stderr: bool = False,
        maxiter: int = 100, xtol=None, ftol=None, gtol=None, chunksize: int = 128) -> FitResult:
    """Fit `model` to every voxel of `data`.

    model      built-in model name ("MonoExp", "SPGR(TR=15)", "IVIM", ...; see `models()`) or
               an `Expr`.
    x          acquisition values, shape (n_obs,) or (n_obs, k).
    data       array with the observations along `axis` (default: last).
    mask       boolean/int array with the spatial shape; zero voxels are skipped.
    init       None (data-driven per voxel), P values for all voxels, or an array of shape
               (*spatial, P).
    lower, upper  bounds, P values each (use np.inf for none). Defaults come from the model.
    covariates per-voxel known maps passed to the model, shape spatial or (*spatial, K)
               (e.g. a relative B1 map for SPGR).
    weights    per-observation weights, shape (n_obs,).
    precision  "float64" or "float32" (half the memory, about twice as fast).
    stderr     also return standard-error maps.

    CPU threads come from PYTHON_JULIACALL_THREADS (default "auto") at the first call.
    """
    jl = julia()
    dt = np.dtype(np.float32 if precision in ("float32", "single", np.float32) else np.float64)
    data = np.asarray(data)
    if data.dtype.byteorder == ">" or data.dtype.kind not in "fiub":
        data = data.astype(dt)
    data = np.ascontiguousarray(np.moveaxis(data, axis, -1))
    spatial = data.shape[:-1]
    nobs = data.shape[-1]
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] != nobs:
        raise ValueError(f"x has {x.shape[0]} entries but data have {nobs} observations")

    jmodel = _model(jl, model)
    names = [str(n) for n in jl.VoxelFit.paramnames(jmodel)]
    P = len(names)

    def spatial_map(a, name, dtype=None):
        a = np.asarray(a, dtype=dtype)
        if a.shape != spatial:
            raise ValueError(f"{name} has shape {a.shape}, expected {spatial}")
        return np.ascontiguousarray(a)

    def stacked_map(a, name, k=None):
        # (*spatial, K) -> (K, *spatial) so that Julia sees (spatial..., K) after axis reversal.
        a = np.asarray(a, dtype=dt)
        if a.shape == spatial:
            return np.ascontiguousarray(a)
        if a.shape[:-1] != spatial or (k is not None and a.shape[-1] != k):
            raise ValueError(f"{name} has shape {a.shape}, expected {spatial} + (K,)")
        return np.ascontiguousarray(np.moveaxis(a, -1, 0))

    keep = []  # keep converted inputs alive during the call
    kw = dict(maxiter=int(maxiter), xtol=xtol, ftol=ftol, gtol=gtol, chunksize=int(chunksize))
    if mask is not None:
        m = spatial_map(mask, "mask")
        if m.dtype.kind not in "biuf":
            m = m.astype(np.uint8)
        keep.append(m)
        kw["mask"] = _wrap(jl, m)
    if init is not None:
        a = np.asarray(init, dtype=np.float64)
        if a.ndim == 1:
            kw["init"] = a
        else:
            a = stacked_map(a, "init", P)
            keep.append(a)
            kw["init"] = _wrap(jl, a)
    if covariates is not None:
        c = stacked_map(covariates, "covariates")
        keep.append(c)
        kw["covariates"] = _wrap(jl, c)
    for k, v in (("lower", lower), ("upper", upper), ("weights", weights)):
        if v is not None:
            kw[k] = np.asarray(v, dtype=np.float64)

    stack = np.full((P,) + spatial, np.nan, dtype=dt)
    sse = np.full(spatial, np.nan, dtype=dt)
    iterations = np.zeros(spatial, dtype=np.int32)
    status = np.zeros(spatial, dtype=np.uint8)
    se = np.full((P,) + spatial, np.nan, dtype=dt) if stderr else None

    jl.VoxelFit.fit_into_b(jmodel, x, _wrap(jl, data), _wrap(jl, stack), _wrap(jl, sse),
                           _wrap(jl, iterations), _wrap(jl, status),
                           None if se is None else _wrap(jl, se), **kw)
    del keep
    return FitResult(names=names, params={n: stack[j] for j, n in enumerate(names)}, sse=sse,
                     iterations=iterations, status=status,
                     stderr=None if se is None else {n: se[j] for j, n in enumerate(names)},
                     _stack=stack)


def models() -> dict:
    """Built-in model names and their parameters."""
    jl = julia()
    d = jl.seval(
        "Dict(k => [string(p) for p in VoxelFit.paramnames(k == \"SPGR\" ? v(TR=1.0) : v())]"
        " for (k, v) in VoxelFit.BUILTIN_MODELS)")
    return {str(k): [str(p) for p in v] for k, v in d.items()}
