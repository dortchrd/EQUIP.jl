"""Tests for the Python front end (run: python -m pytest python/tests or python python/tests/test_voxelfit.py)."""
import time

import numpy as np

import voxelfit


def monoexp_data(shape=(40, 30, 6), seed=0, sigma=5.0):
    rng = np.random.default_rng(seed)
    te = np.arange(10, 330, 10.0)
    t2 = rng.uniform(40, 120, shape)
    s0 = rng.uniform(800, 1200, shape)
    data = s0[..., None] * np.exp(-te / t2[..., None]) + sigma * rng.standard_normal(shape + (te.size,))
    return te, data, s0, t2


def test_monoexp_mask_precision_stderr():
    te, data, s0, t2 = monoexp_data()
    mask = np.random.default_rng(1).random(t2.shape) > 0.25
    for precision in ("float64", "float32"):
        res = voxelfit.fit("MonoExp", te, data.astype(np.float32), mask=mask, precision=precision,
                           stderr=True)
        assert res["T2"].shape == t2.shape
        assert res["T2"].dtype == np.dtype(precision)
        rel = np.abs(res["T2"][mask] - t2[mask]) / t2[mask]
        assert np.median(rel) < 0.01
        assert np.all(np.isnan(res["T2"][~mask])) and np.all(res.status[~mask] == 0)
        assert np.all(res.converged[mask])
        # standard errors should roughly match the actual error spread
        z = (res["T2"][mask] - t2[mask]) / res.stderr["T2"][mask]
        assert 0.8 < np.std(z) < 1.2, np.std(z)


def test_axis_and_layouts_agree():
    te, data, _, _ = monoexp_data(shape=(10, 12, 3))
    a = voxelfit.fit("MonoExp", te, data)
    b = voxelfit.fit("MonoExp", te, np.moveaxis(data, -1, 0), axis=0)
    c = voxelfit.fit("MonoExp", te, np.asfortranarray(data))          # non-contiguous input
    assert np.array_equal(a["T2"], b["T2"]) and np.array_equal(a["T2"], c["T2"])


def test_spgr_b1_covariate_and_expr():
    rng = np.random.default_rng(2)
    shape = (20, 20, 4)
    fa = np.array([2, 5, 10, 15, 20, 30.0])
    tr = 15.0
    t1 = rng.uniform(600, 1600, shape)
    m0 = rng.uniform(4000, 6000, shape)
    b1 = rng.uniform(0.8, 1.2, shape)
    a = np.deg2rad(fa) * b1[..., None]
    e1 = np.exp(-tr / t1)[..., None]
    data = m0[..., None] * np.sin(a) * (1 - e1) / (1 - np.cos(a) * e1) + 2 * rng.standard_normal(shape + (fa.size,))
    with_b1 = voxelfit.fit("SPGR(TR=15)", fa, data, covariates=b1)
    without = voxelfit.fit("SPGR(TR=15)", fa, data)
    err_b1 = np.median(np.abs(with_b1["T1"] - t1) / t1)
    err_no = np.median(np.abs(without["T1"] - t1) / t1)
    assert err_b1 < 0.03 < err_no
    # Same model written as an expression with a named covariate.
    expr = voxelfit.Expr(
        "M0 * sin(deg2rad(fa) * B1) * (1 - exp(-15 / T1)) / (1 - cos(deg2rad(fa) * B1) * exp(-15 / T1))",
        params=["M0", "T1"], x="fa", covariates=["B1"], lower=[0, 1e-6])
    res = voxelfit.fit(expr, fa, data, covariates=b1, init=[5000, 1000])
    assert np.allclose(res["T1"], with_b1["T1"], rtol=1e-4)
    # Per-voxel init map from a previous fit converges immediately.
    again = voxelfit.fit("SPGR(TR=15)", fa, data, covariates=b1,
                         init=np.stack([with_b1["M0"], with_b1["T1"]], axis=-1))
    assert again.iterations.max() <= 2


def test_multicolumn_x():
    # Two settings per observation: S = S0 * exp(-TE/T2) * (1 - exp(-TR/T1))
    rng = np.random.default_rng(3)
    x = np.array([[te, tr] for te in (10, 30, 60, 90) for tr in (300, 800, 2000, 5000)], float)
    shape = (8, 8)
    s0, t2, t1 = rng.uniform(900, 1100, shape), rng.uniform(50, 90, shape), rng.uniform(800, 1500, shape)
    data = (s0[..., None] * np.exp(-x[:, 0] / t2[..., None]) * (1 - np.exp(-x[:, 1] / t1[..., None])))
    m = voxelfit.Expr("S0 * exp(-TE / T2) * (1 - exp(-TR / T1))", params=["S0", "T2", "T1"],
                      x=["TE", "TR"], p0=[1000, 60, 1000], lower=[0, 1, 1])
    res = voxelfit.fit(m, x, data)
    assert np.allclose(res["T1"], t1, rtol=1e-6) and np.allclose(res["T2"], t2, rtol=1e-6)


def test_throughput():
    te, data, _, _ = monoexp_data(shape=(128, 128, 30), sigma=5.0)
    data = data.astype(np.float32)
    voxelfit.fit("MonoExp", te, data[:2], precision="float32")  # warm-up
    t = time.perf_counter()
    res = voxelfit.fit("MonoExp", te, data, precision="float32")
    dt = time.perf_counter() - t
    print(f"\n{data.shape[:-1]} = {res.sse.size} voxels in {dt:.3f} s ({1e6 * dt / res.sse.size:.2f} µs/voxel, "
          f"{voxelfit.julia().Threads.nthreads()} Julia threads)")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
