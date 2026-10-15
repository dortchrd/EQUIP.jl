"""Python front end for VoxelFit.jl: fast voxel-wise nonlinear least-squares fitting of MRI data.

    import numpy as np
    import voxelfit

    te = np.arange(10, 330, 10.0)                       # echo times (ms)
    res = voxelfit.fit("MonoExp", te, data, mask=mask)  # data: (..., n_echoes)
    t2 = res["T2"]                                      # same spatial shape as data[..., 0]

    acq = {"TI": ti, "TD": 2.5}                         # protocol: arrays per observation, scalars constant
    res = voxelfit.fit("SIRFSE", acq, data, fixed={"kmf": 12.5, "Sm": sm_map}, free=["R1m"])

Julia runs in-process through juliacall. Data, masks and maps are shared with Julia without
copying (after at most one conversion to a C-contiguous array with the observation axis last),
and Julia writes the parameter maps straight into NumPy arrays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["fit", "Expr", "Estimate", "FitResult", "models", "STATUS", "julia"]

STATUS = {
    0: "not fitted (outside mask)",
    1: "converged (step)",
    2: "converged (cost)",
    3: "converged (gradient)",
    4: "max iterations",
    5: "failed",
    6: "invalid data or fixed-map value",
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
        jl.seval("_vf_estimate(n, s, l, u) = Symbol(n) => VoxelFit.Estimate(s === nothing ? :auto : Float64(s); lower = l, upper = u)")
        jl.seval("_vf_tuple(xs...) = xs")
        _JL = jl
    return _JL


@dataclass
class Estimate:
    """Estimate a parameter, with a starting value (None: from the model's guess) and bounds."""

    start: Optional[float] = None
    lower: float = -np.inf
    upper: float = np.inf


@dataclass
class Expr:
    """A custom model written as a Julia expression for the signal at one observation.

    Expr("S0 * exp(-TE / T2) + C", params=["S0", "T2", "C"], protocol=["TE"],
         p0=[1000, 50, 0], lower=[0, 0, 0])

    `protocol` names the acquisition settings used in the expression. The expression is
    compiled as Julia code.
    """

    expr: str
    params: Sequence[str]
    protocol: Sequence[str] = ("x",)
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
        f = lambda v: None if v is None else np.asarray(v, dtype=np.float64)
        return jl.VoxelFit.expr_model(model.expr, params=_strings(jl, model.params),
                                      protocol=jl._vf_tuple(*[str(p) for p in model.protocol]),
                                      p0=f(model.p0), lower=f(model.lower), upper=f(model.upper))
    if isinstance(model, str):
        return jl.VoxelFit.parse_model(model)
    return model  # assume an already-constructed Julia model


def _free(jl, free):
    if free is None:
        return jl._vf_tuple()
    if isinstance(free, str):
        free = [free]
    if isinstance(free, dict):
        items = []
        for name, est in free.items():
            if est is None:
                items.append(name)
            else:
                est = est if isinstance(est, Estimate) else Estimate(*est)
                items.append(jl._vf_estimate(name, est.start, float(est.lower), float(est.upper)))
        free = items
    return jl._vf_tuple(*free)


def fit(model, acq, data, *, mask=None, axis: int = -1, fixed=None, free=None, start=None,
        weights=None, precision: str = "float64", stderr: bool = False, maxiter: int = 100,
        xtol=None, ftol=None, gtol=None, chunksize: int = 128) -> FitResult:
    """Fit `model` to every voxel of `data`.

    model      built-in model name ("MonoExp", "SIRFSE", "SIRFSE(magnitude=false)", ...; see
               `models()`) or an `Expr`.
    acq        protocol: a dict of acquisition settings (arrays with one value per observation,
               scalars constant), e.g. {"TI": ti, "TD": 2.5}, or one array for one-setting models.
    data       array with the observations along `axis` (default: last).
    mask       array with the spatial shape; zero voxels are skipped.
    fixed      dict of parameters to hold fixed: numbers, or arrays with the spatial shape
               (per-voxel maps such as {"B1": b1map}). A non-finite map value marks the voxel invalid.
    free       parameters to also estimate: a list of names, or a dict name -> Estimate(start,
               lower, upper) (or a (start, lower, upper) tuple).
    start      dict of starting values (numbers or spatial maps) for estimated parameters, or a
               previous FitResult.
    weights    per-observation weights.
    precision  "float64" or "float32" (half the memory, faster).
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

    jmodel = _model(jl, model)
    keep = []  # keep converted inputs alive during the call

    def spatial_map(a, name):
        a = np.asarray(a)
        if a.shape != spatial:
            raise ValueError(f"{name} has shape {a.shape}, expected {spatial}")
        if a.dtype.kind not in "biuf" or a.dtype.byteorder == ">":
            a = a.astype(np.float64)
        a = np.ascontiguousarray(a)
        keep.append(a)
        return _wrap(jl, a)

    def value_or_map(v, name):
        return float(v) if np.ndim(v) == 0 else spatial_map(v, name)

    if isinstance(acq, dict):
        jacq = jl.VoxelFit.named(**{k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=np.float64))
                                    for k, v in acq.items()})
    else:
        jacq = np.asarray(acq, dtype=np.float64)

    jfixed = jl.VoxelFit.named(**{k: value_or_map(v, f"fixed[{k!r}]") for k, v in (fixed or {}).items()})
    jfree = _free(jl, free)
    if isinstance(start, FitResult):
        start = start.params
    jstart = jl.VoxelFit.named(**{k: value_or_map(v, f"start[{k!r}]") for k, v in (start or {}).items()})

    names = [str(n) for n in jl.VoxelFit.free_names(jmodel, fixed=jfixed, free=jfree)]
    P = len(names)
    stack = np.full((P,) + spatial, np.nan, dtype=dt)
    sse = np.full(spatial, np.nan, dtype=dt)
    iterations = np.zeros(spatial, dtype=np.int32)
    status = np.zeros(spatial, dtype=np.uint8)
    se = np.full((P,) + spatial, np.nan, dtype=dt) if stderr else None

    kw = dict(fixed=jfixed, free=jfree, start=jstart, maxiter=int(maxiter), xtol=xtol, ftol=ftol,
              gtol=gtol, chunksize=int(chunksize))
    if mask is not None:
        kw["mask"] = spatial_map(np.asarray(mask) != 0, "mask")
    if weights is not None:
        kw["weights"] = np.asarray(weights, dtype=np.float64)

    jl.VoxelFit.fit_into_b(jmodel, jacq, _wrap(jl, data), _wrap(jl, stack), _wrap(jl, sse),
                           _wrap(jl, iterations), _wrap(jl, status),
                           None if se is None else _wrap(jl, se), **kw)
    del keep
    return FitResult(names=names, params={n: stack[j] for j, n in enumerate(names)}, sse=sse,
                     iterations=iterations, status=status,
                     stderr=None if se is None else {n: se[j] for j, n in enumerate(names)},
                     _stack=stack)


def models() -> dict:
    """Built-in models: protocol settings and parameters with their default roles."""
    jl = julia()
    out = {}
    for name in jl.seval("sort!(collect(keys(VoxelFit.BUILTIN_MODELS)))"):
        m = jl.VoxelFit.parse_model(str(name))
        roles = jl.seval("m -> [string(k) => VoxelFit._roledesc(v) for (k, v) in pairs(VoxelFit.parameters(m))]")(m)
        out[str(name)] = {
            "protocol": [str(p) for p in jl.VoxelFit.protocol(m)],
            "parameters": {str(k): str(v) for k, v in roles},
        }
    return out
