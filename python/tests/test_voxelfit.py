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


def expm2(A):
    w, V = np.linalg.eig(A)
    return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real


def sir_eq1(ti, td, pmf, R1f, Sf, M0f, kmf, R1m, Sm):
    kfm = kmf * pmf
    A = np.array([[-(R1f + kfm), kmf], [kfm, -(R1m + kmf)]])
    M0 = np.array([M0f, pmf * M0f])
    S = np.diag([Sf, Sm])
    out = []
    for t1, t2 in zip(ti, td):
        Mz = (expm2(A * t1) @ S @ (np.eye(2) - expm2(A * t2)) + (np.eye(2) - expm2(A * t1))) @ M0
        out.append(abs(Mz[0]))
    return np.array(out)


def test_monoexp_mask_precision_stderr():
    te, data, s0, t2 = monoexp_data()
    mask = np.random.default_rng(1).random(t2.shape) > 0.25
    for precision in ("float64", "float32"):
        res = voxelfit.fit("MonoExp", te, data.astype(np.float32), mask=mask, precision=precision,
                           stderr=True)
        assert res.names == ["S0", "T2"]
        assert res["T2"].shape == t2.shape
        assert res["T2"].dtype == np.dtype(precision)
        rel = np.abs(res["T2"][mask] - t2[mask]) / t2[mask]
        assert np.median(rel) < 0.01
        assert np.all(np.isnan(res["T2"][~mask])) and np.all(res.status[~mask] == 0)
        assert np.all(res.converged[mask])
        z = (res["T2"][mask] - t2[mask]) / res.stderr["T2"][mask]
        assert 0.8 < np.std(z) < 1.2, np.std(z)


def test_axis_and_layouts_agree():
    te, data, _, _ = monoexp_data(shape=(10, 12, 3))
    a = voxelfit.fit("MonoExp", te, data)
    b = voxelfit.fit("MonoExp", te, np.moveaxis(data, -1, 0), axis=0)
    c = voxelfit.fit("MonoExp", te, np.asfortranarray(data))
    d = voxelfit.fit("MonoExp", {"TE": te}, data)
    for r in (b, c, d):
        assert np.array_equal(a["T2"], r["T2"])


def test_spgr_b1_map_and_start():
    rng = np.random.default_rng(2)
    shape = (20, 20, 4)
    acq = {"FA": np.array([2, 5, 10, 15, 20, 30.0]), "TR": 15.0}
    t1 = rng.uniform(600, 1600, shape)
    m0 = rng.uniform(4000, 6000, shape)
    b1 = rng.uniform(0.8, 1.2, shape)
    a = np.deg2rad(acq["FA"]) * b1[..., None]
    e1 = np.exp(-acq["TR"] / t1)[..., None]
    data = m0[..., None] * np.sin(a) * (1 - e1) / (1 - np.cos(a) * e1) + 2 * rng.standard_normal(shape + (6,))
    with_b1 = voxelfit.fit("SPGR", acq, data, fixed={"B1": b1})
    without = voxelfit.fit("SPGR", acq, data)
    err_b1 = np.median(np.abs(with_b1["T1"] - t1) / t1)
    err_no = np.median(np.abs(without["T1"] - t1) / t1)
    assert err_b1 < 0.03 < err_no
    again = voxelfit.fit("SPGR", acq, data, fixed={"B1": b1}, start=with_b1)   # warm start
    assert again.iterations.max() <= 2
    b1bad = b1.copy()
    b1bad[0, 0, 0] = np.nan
    r = voxelfit.fit("SPGR", acq, data, fixed={"B1": b1bad})
    assert r.status[0, 0, 0] == 6 and np.isnan(r["T1"][0, 0, 0])


def test_sirfse_fixed_free():
    acq = {"TI": np.array([0.010, 0.050, 0.056, 0.277, 0.843]),
           "TD": np.array([3.270, 4.489, 1.652, 2.922, 0.010])}
    truth = dict(pmf=0.12, R1f=1.0, Sf=-0.95, M0f=1000.0, kmf=15.0, R1m=1.0, Sm=0.83)
    y = sir_eq1(acq["TI"], acq["TD"], **truth)
    data = np.tile(y, (3, 4, 1))
    r = voxelfit.fit("SIRFSE", acq, data, free=["kmf"])
    assert r.names == ["pmf", "R1f", "Sf", "M0f", "kmf"]
    assert np.allclose(r["kmf"], 15.0, rtol=1e-5) and np.allclose(r["pmf"], 0.12, rtol=1e-5)
    r = voxelfit.fit("SIRFSE", acq, data, fixed={"kmf": 15.0, "Sm": np.full((3, 4), 0.83)})
    assert r.names == ["pmf", "R1f", "Sf", "M0f"] and np.allclose(r["pmf"], 0.12, rtol=1e-5)
    r = voxelfit.fit("SIRFSE", acq, data, free={"kmf": voxelfit.Estimate(20.0, 5.0, 50.0)})
    assert np.allclose(r["kmf"], 15.0, rtol=1e-5)
    models = voxelfit.models()
    assert models["SIRFSE"]["protocol"] == ["TI", "TD"] and "kmf" in models["SIRFSE"]["parameters"]


def test_expr_model_two_settings():
    rng = np.random.default_rng(3)
    te = np.repeat([10.0, 30.0, 60.0, 90.0], 4)
    tr = np.tile([300.0, 800.0, 2000.0, 5000.0], 4)
    shape = (8, 8)
    s0, t2, t1 = rng.uniform(900, 1100, shape), rng.uniform(50, 90, shape), rng.uniform(800, 1500, shape)
    data = s0[..., None] * np.exp(-te / t2[..., None]) * (1 - np.exp(-tr / t1[..., None]))
    m = voxelfit.Expr("S0 * exp(-TE / T2) * (1 - exp(-TR / T1))", params=["S0", "T2", "T1"],
                      protocol=["TE", "TR"], p0=[1000, 60, 1000], lower=[0, 1, 1])
    res = voxelfit.fit(m, {"TE": te, "TR": tr}, data)
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
