"""End-to-end self test: every emitted backend must agree with the reference.

1. numerics  : VF recovers a known rational system to machine precision
2. touchstone: write -> read round trip, and the 2-port column-order gotcha
3. emitters  : the generated SystemVerilog is PARSED BACK into matrices and
               compared to the model, and the generated C and Python are run
               against sp2ss's own simulator on the same stimulus
4. physics   : the fitted DC resistance and inductance match the demo network
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp2ss.touchstone import read_touchstone, write_touchstone, Network
from sp2ss.convert import to_param, upper_tri_index, flatten, unflatten
from sp2ss.vectfit import vector_fit, model_response, fit_errors
from sp2ss.statespace import (from_pole_residue, balanced_truncate, to_modal,
                              discretize)
from sp2ss.passivity import check_passivity
from sp2ss import emit
import tools.make_demo_pdn as demo

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------ 1
def test_vf():
    print("\n[1] vector fitting on a known rational system")
    p = np.array([-4500+0j, -41000+0j, -100+5000j, -100-5000j])
    r = np.array([[-3000+0j, -83000+0j, -5+7000j, -5-7000j],
                  [-1500+0j,  20000+0j, 30+900j,  30-900j]])
    d = np.array([0.2, -0.1]); e = np.zeros(2)
    f = np.logspace(1, 5, 400); s = 2j*np.pi*f
    F = model_response(s, p, r, d, e)
    vf = vector_fit(F, s, order=4, n_iter=10, asymp='d', weight='inv', prune=False)
    err = fit_errors(F, model_response(s, vf.poles, vf.residues, vf.d, vf.e))
    check("VF recovers poles", np.allclose(np.sort_complex(vf.poles),
                                           np.sort_complex(p), rtol=1e-8),
          f"max rel err {err['max_rel']:.2e}")
    check("VF residual is machine precision", err['max_rel'] < 1e-10)


# ------------------------------------------------------------------ 2
def test_touchstone():
    print("\n[2] touchstone round trip")
    f = np.logspace(6, 9, 51)
    rng = np.random.default_rng(3)
    d = rng.normal(size=(51, 2, 2)) + 1j*rng.normal(size=(51, 2, 2))
    d = 0.5*(d + np.swapaxes(d, 1, 2))
    net = Network(f, d, "S", 50.0)
    with tempfile.TemporaryDirectory() as td:
        for fmt in ("RI", "MA", "DB"):
            fp = os.path.join(td, "t.s2p")
            write_touchstone(fp, net, fmt=fmt)
            back = read_touchstone(fp)
            check(f"round trip {fmt}", np.abs(back.data-net.data).max() < 1e-7,
                  f"max diff {np.abs(back.data-net.data).max():.2e}")
        # the classic 2-port ordering trap: S21 must land in row 1 col 0
        fp = os.path.join(td, "o.s2p")
        with open(fp, "w") as fh:
            fh.write("# HZ S RI R 50\n1e9 0.1 0 0.2 0 0.3 0 0.4 0\n")
        n2 = read_touchstone(fp)
        check("2-port order S11,S21,S12,S22",
              np.allclose(n2.data[0], [[0.1, 0.3], [0.2, 0.4]]),
              f"got {n2.data[0].real.tolist()}")
        # noise data after the network block must be ignored
        with open(fp, "w") as fh:
            fh.write("# HZ S MA R 50\n1e9 1 0 1 0 1 0 1 0\n2e9 1 0 1 0 1 0 1 0\n"
                     "1e9 1.5 0.2 0 50\n2e9 1.6 0.3 0 50\n")
        n3 = read_touchstone(fp)
        check("v1 noise block skipped", n3.n_freq == 2, f"n_freq={n3.n_freq}")


# ------------------------------------------------------------------ 3+4
def build_model():
    net = read_touchstone("out/demo_pdn.s2p")
    Z = to_param(net, "Z").symmetrized()
    elems = upper_tri_index(2)
    F = flatten(Z, elems)
    vf = vector_fit(F, Z.s, order=40, n_iter=12, asymp="de", weight="inv")
    R = np.transpose(unflatten(vf.residues, elems, 2), (2, 0, 1))
    D = unflatten(vf.d[:, None], elems, 2)[:, :, 0]
    E = unflatten(vf.e[:, None], elems, 2)[:, :, 0]
    ss = from_pole_residue(vf.poles, R, D, E)
    ss = balanced_truncate(ss, order=8)
    ss = to_modal(ss)
    return net, Z, ss, discretize(ss, 5e-12, "zoh")


def parse_sv(text, n, npt, ny):
    """Rebuild Ad/Bd/Cd/Dd/Ee from generated SystemVerilog."""
    body = re.sub(r"//[^\n]*", "", text)
    body = " ".join(body.split())
    Ad = np.zeros((n, n)); Bd = np.zeros((n, npt))
    Cd = np.zeros((ny, n)); Dd = np.zeros((ny, npt)); Ee = np.zeros((ny, npt))
    num = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"

    pat = re.compile(rf"([-+]?)\s*({num})\s*\*\s*"
                     rf"(?:\((i_p\d+)\s*-\s*i_p\d+_d\)|([A-Za-z_]\w*))")

    def terms(rhs):
        out = []
        for m in pat.finditer(rhs):
            sign = -1.0 if m.group(1) == "-" else 1.0
            if m.group(3):                       # E branch: (i_pJ - i_pJ_d)
                out.append((sign*float(m.group(2)), "E" + m.group(3)[3:]))
            else:
                out.append((sign*float(m.group(2)), m.group(4)))
        return out

    for m in re.finditer(rf"assign v_p(\d+) = (.+?);", body):
        i = int(m.group(1))
        for c, sym in terms(m.group(2)):
            if sym.startswith("x"):
                Cd[i, int(sym[1:])] = c
            elif sym.startswith("E"):
                Ee[i, int(sym[1:])] = c
            elif sym.startswith("i_p"):
                Dd[i, int(sym[3:])] = c
    for m in re.finditer(rf"x(\d+) <= (.+?);", body):
        i = int(m.group(1))
        rhs = m.group(2)
        if rhs.strip() == "0.0":
            continue
        for c, sym in terms(rhs):
            if sym.startswith("x"):
                Ad[i, int(sym[1:])] = c
            elif sym.startswith("i_p"):
                Bd[i, int(sym[3:])] = c
    return Ad, Bd, Cd, Dd, Ee


def test_emitters():
    print("\n[3] emitted backends vs the reference model")
    net, Z, ss, dss = build_model()
    n, npt, ny = dss.n_states, dss.n_ports, dss.Cd.shape[0]

    # --- SystemVerilog: parse the generated text back into matrices ------
    sv = emit.emit_systemverilog(dss, "pdn_ss", {})
    Ad, Bd, Cd, Dd, Ee = parse_sv(sv, n, npt, ny)
    for nm, got, want in [("Ad", Ad, dss.Ad), ("Bd", Bd, dss.Bd),
                          ("Cd", Cd, dss.Cd), ("Dd", Dd, dss.Dd),
                          ("E/Ts", Ee, dss.Ee/dss.Ts)]:
        rel = np.abs(got-want).max()/max(np.abs(want).max(), 1e-300)
        check(f"SV {nm} matches", rel < 1e-14, f"max rel {rel:.2e}")
    check("SV has no malformed signs", not re.search(r"\+-|\+ \+|- -", sv))

    # --- stimulus shared by every backend --------------------------------
    N = 3000
    rng = np.random.default_rng(7)
    u = np.zeros((N, npt))
    u[:, 0] = np.clip((np.arange(N)-100)/1.0, 0, 1)*0.8
    u[:, 1] = 0.05*np.sin(2*np.pi*np.arange(N)/37.0)
    u += 0.01*rng.normal(size=u.shape)
    y_ref = dss.simulate(u)

    # --- emitted Python ---------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        pyf = os.path.join(td, "m.py")
        open(pyf, "w").write(emit.emit_python(dss, "pdn_ss", {}))
        sys.path.insert(0, td)
        import importlib
        mod = importlib.import_module("m")
        y_py = mod.PdnSs().run(u)
        sys.path.pop(0)
        rel = np.abs(y_py-y_ref).max()/max(np.abs(y_ref).max(), 1e-300)
        check("emitted Python matches simulator", rel < 1e-12, f"max rel {rel:.2e}")

        # --- emitted C ----------------------------------------------------
        cf = os.path.join(td, "m.c")
        open(cf, "w").write(emit.emit_c(dss, "pdn_ss", {}) + f"""
#include <stdio.h>
int main(void) {{
    pdn_ss_state st; pdn_ss_reset(&st);
    double u[{npt}], y[{ny}];
    for (int k = 0; k < {N}; ++k) {{
        if (scanf("%lf %lf", &u[0], &u[1]) != 2) return 1;
        pdn_ss_step(&st, u, y);
        printf("%.17g %.17g\\n", y[0], y[1]);
    }}
    return 0;
}}
""")
        exe = os.path.join(td, "m")
        r = subprocess.run(["cc", "-O2", "-o", exe, cf], capture_output=True, text=True)
        if r.returncode != 0:
            check("emitted C compiles", False, r.stderr.strip()[:200])
        else:
            check("emitted C compiles", True)
            stdin = "\n".join(f"{a:.17g} {b:.17g}" for a, b in u)
            r = subprocess.run([exe], input=stdin, capture_output=True, text=True)
            y_c = np.array([[float(v) for v in ln.split()]
                            for ln in r.stdout.strip().splitlines()])
            rel = np.abs(y_c-y_ref).max()/max(np.abs(y_ref).max(), 1e-300)
            check("emitted C matches simulator", rel < 1e-12, f"max rel {rel:.2e}")

    return net, Z, ss, dss


def test_physics(net, Z, ss, dss):
    print("\n[4] physical sanity against the known demo network")
    ztrue = demo.pdn_z(np.array([1e-6, net.f[0]]))
    dc_true = ztrue[0, 0, 0].real
    dc_fit = ss.dc_gain()[0, 0]
    check("DC resistance", abs(dc_fit-dc_true)/dc_true < 1e-3,
          f"fit {dc_fit*1e3:.6f} mohm vs true {dc_true*1e3:.6f} mohm")
    if ss.E is not None:
        check("port-0 inductance ~ die ESL/4 = 0.25 pH",
              abs(ss.E[0, 0]*1e12 - 0.25) < 0.05,
              f"got {ss.E[0,0]*1e12:.4f} pH")
    ok, viol, dmin = check_passivity(ss, verbose=False)
    check("model is passive", ok, f"min eig Herm(D) = {dmin:.3e}")
    check("discrete model is stable", dss.is_stable(),
          f"rho = {dss.spectral_radius():.9f}")
    H = ss.freqresp(net.f)
    rel = (np.abs(H-Z.data)/np.abs(Z.data)).max()
    check("frequency fit under 0.1%", rel < 1e-3, f"max rel {rel:.2e}")


if __name__ == "__main__":
    if not os.path.exists("out/demo_pdn.s2p"):
        subprocess.run([sys.executable, "tools/make_demo_pdn.py"], check=True)
    test_vf()
    test_touchstone()
    net, Z, ss, dss = test_emitters()
    test_physics(net, Z, ss, dss)
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURE(S): {FAILS}"))
    sys.exit(1 if FAILS else 0)
