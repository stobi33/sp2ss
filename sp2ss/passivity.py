"""Passivity assessment and enforcement for immittance (Z / Y) state spaces.

A PDN macromodel that is not passive can extract energy from the simulation.
In an RNM loop that is closed around a regulator or a decap network, an active
macromodel shows up as a slowly growing oscillation that looks like a "real"
instability -- so this pass is not optional.

For an immittance model, passive == POSITIVE REAL:
    (a) H(s) analytic in Re(s) > 0            -> guaranteed by stable poles
    (b) H(s*) = H(s)*                         -> guaranteed by a real realization
    (c) Herm(H(jw)) = (H + H^H)/2  >= 0  for all w

(c) is checked exactly with the positive-real Hamiltonian: its purely imaginary
eigenvalues are precisely the frequencies where an eigenvalue of Herm(H) crosses
zero (Grivet-Talocia, IEEE TCAS-I 51(9), 2004).  Between consecutive crossings
the sign cannot change, so testing one interior point per interval is exhaustive.

Enforcement perturbs C only (D is handled separately), which keeps the poles --
and therefore stability -- untouched.  ΔZ(jw) = ΔC (jwI - A)^-1 B is LINEAR in
ΔC, so each violated mode becomes one linear inequality and the whole step is a
small convex QP: minimise the change to the fitted response subject to pushing
every negative eigenvalue up to +margin.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .statespace import StateSpace

__all__ = ["herm_eigs", "passivity_violations", "check_passivity",
           "enforce_passivity", "make_D_passive"]


def herm_eigs(ss, f):
    """Eigenvalues of the Hermitian part of H(j2 pi f), shape (Nf, n)."""
    H = ss.freqresp(f)
    Hh = 0.5 * (H + np.conj(np.swapaxes(H, 1, 2)))
    return np.linalg.eigvalsh(Hh)


def make_D_passive(D, margin=0.0):
    """Clip the symmetric part of D so D + D^T >= 2*margin*I."""
    Dsym = 0.5 * (D + D.T)
    Dskew = 0.5 * (D - D.T)
    w, V = np.linalg.eigh(Dsym)
    if w.min() >= margin:
        return D, 0.0
    wc = np.clip(w, margin, None)
    Dn = V @ np.diag(wc) @ V.T + Dskew
    return Dn, float(np.abs(Dn - D).max())


def _pr_hamiltonian(ss):
    """Positive-real Hamiltonian; None if D + D^T is singular."""
    Dh = ss.D + ss.D.T
    if np.linalg.cond(Dh) > 1e12:
        return None
    Di = np.linalg.inv(Dh)
    K = ss.A - ss.B @ Di @ ss.C
    return np.block([[K, -ss.B @ Di @ ss.B.T],
                     [ss.C.T @ Di @ ss.C, -K.T]])


def crossover_freqs(ss, tol=1e-8):
    """Frequencies (Hz) where an eigenvalue of Herm(H) crosses zero."""
    M = _pr_hamiltonian(ss)
    if M is None:
        return None
    ev = np.linalg.eigvals(M)
    scale = max(np.abs(ev).max(), 1e-300)
    imag_axis = np.abs(ev.real) < tol * scale
    w = np.abs(ev[imag_axis].imag)
    w = np.unique(np.round(w[w > 0], 6))
    return w / (2 * np.pi)


def passivity_violations(ss, fmin=None, fmax=None, n_scan=4000):
    """Locate passivity violations.

    Returns a list of dicts: {'f': Hz, 'lam': most negative eigenvalue}.
    Uses the exact Hamiltonian crossings when available and falls back to a
    dense logarithmic scan otherwise; the scan is always run too, because it
    also catches the (numerically common) shallow violations.
    """
    fx = crossover_freqs(ss)
    probes = []

    if fx is not None and len(fx):
        edges = np.concatenate(([0.0], np.sort(fx), [np.sort(fx)[-1] * 10 + 1.0]))
        for a, b in zip(edges[:-1], edges[1:]):
            probes.append(0.5 * (a + b) if a > 0 else max(b * 0.5, 1e-30))

    fp = ss.max_pole_freq()
    lo = fmin if fmin else max(fp * 1e-6, 1e-3)
    hi = fmax if fmax else fp * 20
    probes += list(np.logspace(np.log10(lo), np.log10(hi), n_scan))
    probes = np.unique(np.array([p for p in probes if p > 0]))

    lam = herm_eigs(ss, probes)
    worst = lam.min(axis=1)
    bad = worst < 0
    out = []
    if bad.any():
        idx = np.where(bad)[0]
        # group contiguous violating samples into bands, report the worst of each
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for grp in np.split(idx, splits):
            j = grp[np.argmin(worst[grp])]
            out.append({"f": float(probes[j]), "lam": float(worst[j]),
                        "f_lo": float(probes[grp[0]]), "f_hi": float(probes[grp[-1]])})
    return out


def check_passivity(ss, verbose=True, **kw):
    """Convenience wrapper -> (is_passive, violations, worst_eig)."""
    Dh = ss.D + ss.D.T
    dmin = float(np.linalg.eigvalsh(0.5 * Dh).min())
    v = passivity_violations(ss, **kw)
    ok = (len(v) == 0) and dmin >= 0
    if verbose:
        if ok:
            print(f"  passivity: OK (min eig of Herm(D) = {dmin:.4e})")
        else:
            print(f"  passivity: {len(v)} violating band(s), "
                  f"min eig of Herm(D) = {dmin:.4e}")
            for d in v[:8]:
                print(f"    {d['f_lo']:.4g} .. {d['f_hi']:.4g} Hz  "
                      f"worst eig = {d['lam']:.4e}")
            if len(v) > 8:
                print(f"    ... and {len(v)-8} more")
    return ok, v, dmin


# ------------------------------------------------------------- enforcement

def _solve_qp(P, Aineq, b, ridge=1e-12):
    """min 1/2 x'Px  s.t.  Aineq x >= b, via the bound-constrained dual.

    Dual:  max_{lam >= 0}  -1/2 lam' (A P^-1 A') lam + b' lam,   x = P^-1 A' lam
    """
    from scipy.optimize import minimize

    n = P.shape[0]
    Pr = P + ridge * np.trace(P) / max(n, 1) * np.eye(n)
    cho = sla.cho_factor(Pr)
    PiAt = sla.cho_solve(cho, Aineq.T)          # P^-1 A'
    M = Aineq @ PiAt
    M = 0.5 * (M + M.T)
    M += 1e-12 * max(np.trace(M) / max(M.shape[0], 1), 1e-300) * np.eye(M.shape[0])

    def fun(l):
        return 0.5 * l @ M @ l - b @ l

    def jac(l):
        return M @ l - b

    l0 = np.maximum(b, 0.0)
    r = minimize(fun, l0, jac=jac, method="L-BFGS-B",
                 bounds=[(0.0, None)] * len(b),
                 options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12})
    return PiAt @ np.maximum(r.x, 0.0)


def enforce_passivity(ss, f_band, margin_rel=1e-3, n_iter=20, n_scan=3000,
                      verbose=True):
    """Make an immittance state space passive by perturbing C.

    Parameters
    ----------
    f_band     : frequencies used to weight "don't change the fitted response"
    margin_rel : target eigenvalue floor, relative to the mean of Herm(D)
    n_iter     : outer iterations (each solves one QP)

    Returns a new StateSpace and a report dict.
    """
    ss = StateSpace(ss.A, ss.B, ss.C.copy(), ss.D.copy(), ss.E, ss.meta)
    n, nu = ss.n_states, ss.n_ports
    ny = ss.C.shape[0]

    Dnew, dchg = make_D_passive(ss.D, margin=0.0)
    if dchg > 0:
        if verbose:
            print(f"  D was not passive; symmetric part clipped (max |dD| = {dchg:.3e})")
        ss.D = Dnew

    scaleD = max(np.abs(np.linalg.eigvalsh(0.5 * (ss.D + ss.D.T))).max(), 1e-30)
    margin = margin_rel * scaleD

    # --- objective: preserve the frequency response over the fitted band -----
    fw = np.asarray(f_band, float)
    fw = fw[fw > 0]
    if len(fw) > 400:
        fw = np.exp(np.linspace(np.log(fw.min()), np.log(fw.max()), 400))
    Gcols = []
    I = np.eye(n)
    for fk in fw:
        G = np.linalg.solve(2j * np.pi * fk * I - ss.A, ss.B)   # n x nu
        for j in range(nu):
            Gcols.append(G[:, j].real)
            Gcols.append(G[:, j].imag)
    Phi = np.array(Gcols)                                       # rows x n
    # weight each frequency by 1/|H| so the objective is a RELATIVE change
    Hmag = np.abs(ss.freqresp(fw)).max(axis=(1, 2))
    wrep = np.repeat(1.0 / np.maximum(Hmag, 1e-300), 2 * nu)
    Phi = Phi * wrep[:, None]
    Pblk = Phi.T @ Phi
    Pblk = 0.5 * (Pblk + Pblk.T)
    P = sla.block_diag(*([Pblk] * ny))                          # x = vec rows of dC

    report = {"iterations": 0, "initial_worst": None, "final_worst": None,
              "dC_norm": 0.0, "converged": False}

    for it in range(n_iter):
        viol = passivity_violations(ss, fmin=fw.min() * 1e-3,
                                    fmax=fw.max() * 1e3, n_scan=n_scan)
        worst = min([d["lam"] for d in viol], default=0.0)
        if report["initial_worst"] is None:
            report["initial_worst"] = worst
        if not viol:
            report["converged"] = True
            report["final_worst"] = worst
            report["iterations"] = it
            if verbose and it:
                print(f"  passivity enforced after {it} iteration(s)")
            break

        # one constraint per violating (frequency, negative eigenvector)
        rows, rhs = [], []
        for d in viol:
            fs = np.unique(np.linspace(d["f_lo"], d["f_hi"], 7))
            for fk in fs:
                H = ss.freqresp([fk])[0]
                Hh = 0.5 * (H + H.conj().T)
                w, V = np.linalg.eigh(Hh)
                G = np.linalg.solve(2j * np.pi * fk * np.eye(n) - ss.A, ss.B)
                for q in np.where(w < margin)[0]:
                    v = V[:, q]
                    u = G @ v                       # n-vector
                    row = np.zeros(ny * n)
                    for i in range(ny):
                        row[i * n:(i + 1) * n] = (np.conj(v[i]) * u).real
                    rows.append(row)
                    rhs.append(margin - w[q])
        if not rows:
            report["converged"] = True
            break

        Aineq = np.array(rows)
        b = np.array(rhs)
        x = _solve_qp(P, Aineq, b)
        dC = x.reshape(ny, n)

        step = 1.0
        for _ in range(6):                      # backtrack if we overshoot
            trial = StateSpace(ss.A, ss.B, ss.C + step * dC, ss.D, ss.E, ss.meta)
            tv = passivity_violations(trial, fmin=fw.min() * 1e-3,
                                      fmax=fw.max() * 1e3, n_scan=n_scan)
            tw = min([d["lam"] for d in tv], default=0.0)
            if tw > worst or not tv:
                break
            step *= 0.5
        ss = StateSpace(ss.A, ss.B, ss.C + step * dC, ss.D, ss.E, ss.meta)
        report["dC_norm"] += float(np.linalg.norm(step * dC))
        report["iterations"] = it + 1
        if verbose:
            print(f"  passivity iter {it+1:2d}: worst eig {worst:.3e} -> {tw:.3e} "
                  f"(step {step:g}, {len(b)} constraints)")

    v = passivity_violations(ss, fmin=fw.min() * 1e-3, fmax=fw.max() * 1e3,
                             n_scan=n_scan)
    report["final_worst"] = min([d["lam"] for d in v], default=0.0)
    report["converged"] = len(v) == 0
    return ss, report
