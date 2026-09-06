"""Pole-residue -> real state space, model reduction, and discretization."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

__all__ = ["StateSpace", "from_pole_residue", "to_pole_residue", "absorb_e",
           "balanced_truncate", "BalancedBasis", "reduce_to_target",
           "to_modal", "block_structure", "discretize", "DiscreteSS",
           "hankel_sv", "suggest_timestep"]


class StateSpace:
    """Continuous descriptor form  y = C x + D u + E du/dt,  dx/dt = A x + B u.

    E is normally zero (asymp='d'); it is only populated when the fit needed an
    s-proportional term.
    """

    def __init__(self, A, B, C, D, E=None, meta=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.E = None if E is None or not np.any(E) else np.asarray(E, float)
        self.meta = dict(meta or {})

    @property
    def n_states(self):
        return self.A.shape[0]

    @property
    def n_ports(self):
        return self.B.shape[1]

    def __repr__(self):
        return (f"<StateSpace n={self.n_states} ports={self.n_ports} "
                f"E={'yes' if self.E is not None else 'no'}>")

    def freqresp(self, f):
        """H(j2 pi f), shape (Nf, ny, nu)."""
        s = 1j * 2 * np.pi * np.asarray(f, dtype=float)
        I = np.eye(self.n_states)
        out = np.empty((len(s), self.C.shape[0], self.B.shape[1]), dtype=complex)
        for k, sk in enumerate(s):
            out[k] = self.C @ np.linalg.solve(sk * I - self.A, self.B) + self.D
            if self.E is not None:
                out[k] += sk * self.E
        return out

    def is_stable(self, margin=0.0):
        return bool(np.max(np.linalg.eigvals(self.A).real) < -margin)

    def max_pole_freq(self):
        ev = np.linalg.eigvals(self.A)
        return float(np.max(np.abs(ev)) / (2 * np.pi))

    def dc_gain(self):
        return self.D - self.C @ np.linalg.solve(self.A, self.B)


def from_pole_residue(poles, R, D, E=None):
    """Build a real state space from  H(s) = D + sE + sum_m R_m/(s-p_m).

    R : (M, ny, nu) residue matrices with conjugate pairs adjacent.

    Realization (Gustavsen): for each pole, replicate the input dimension so
    C (sI-A)^-1 B reproduces the residue matrix exactly.  Conjugate pairs use
    the real 2x2 block  [[sigma, omega], [-omega, sigma]]  with B = [2I; 0]
    and C = [Re R, Im R], which is an exact similarity of the complex pair.
    """
    poles = np.asarray(poles, dtype=complex)
    R = np.asarray(R, dtype=complex)
    M = len(poles)
    ny, nu = R.shape[1], R.shape[2]
    if nu != ny:
        raise ValueError("square residue matrices expected")
    n = nu

    Ab, Bb, Cb = [], [], []
    m = 0
    while m < M:
        if abs(poles[m].imag) < 1e-12:
            Ab.append(poles[m].real * np.eye(n))
            Bb.append(np.eye(n))
            Cb.append(R[m].real)
            m += 1
        else:
            if m + 1 >= M or not np.isclose(poles[m + 1], np.conj(poles[m])):
                raise ValueError("complex poles must come in adjacent conj pairs")
            sr, si = poles[m].real, poles[m].imag
            blk = np.block([[sr * np.eye(n), si * np.eye(n)],
                            [-si * np.eye(n), sr * np.eye(n)]])
            Ab.append(blk)
            Bb.append(np.vstack([2 * np.eye(n), np.zeros((n, n))]))
            Cb.append(np.hstack([R[m].real, R[m].imag]))
            m += 2

    A = sla.block_diag(*Ab)
    B = np.vstack(Bb)
    C = np.hstack(Cb)
    return StateSpace(A, B, C, np.asarray(D, float).real,
                      None if E is None else np.asarray(E, float).real)


def absorb_e(poles, R, D, E, wc):
    """Fold an s*E term into the pole/residue set as a band-limited inductor.

    A raw s*E feedthrough is a pure differentiator: with the piecewise-constant
    current an RNM driver produces, it is impulsive, and any discrete realization
    of it either rings at Nyquist (bilinear, pole at z = -1) or needs
    special-casing in every emitter.  Band-limiting it at wc instead,

        s*E  ->  E*s/(1 + s/wc)  =  E*wc  -  E*wc^2 / (s + wc)

    turns it into an ordinary real pole at -wc with residue -E*wc^2 plus a bump
    of E*wc in D.  Everything downstream (realization, truncation, passivity,
    discretization, emission) then works unchanged, and the branch stays passive
    for any symmetric PSD E because

        Re{ E*jw/(1 + jw/wc) } = E*w^2/wc / (1 + (w/wc)^2)  >=  0.

    Physically E is the series port inductance and wc is the frequency above
    which you have declared the model invalid -- put it near Nyquist.
    """
    E = np.asarray(E, dtype=float)
    E = 0.5 * (E + E.T)
    w, V = np.linalg.eigh(E)
    if w.min() < 0:                       # keep the branch passive
        E = V @ np.diag(np.clip(w, 0.0, None)) @ V.T
    poles = np.concatenate([np.asarray(poles, dtype=complex),
                            [complex(-wc, 0.0)]])
    R = np.concatenate([np.asarray(R, dtype=complex),
                        (-E * wc ** 2).astype(complex)[None, :, :]], axis=0)
    D = np.asarray(D, dtype=float) + E * wc
    return poles, R, D, E


def to_pole_residue(ss):
    """State space -> pole/residue (rank-1 residues from the eigenbasis)."""
    lam, V = np.linalg.eig(ss.A)
    Vi = np.linalg.inv(V)
    Cv = ss.C @ V
    Vb = Vi @ ss.B
    R = np.einsum("im,mj->mij", Cv, Vb)
    order = np.argsort(lam.imag + 1e-9 * lam.real)
    return lam[order], R[order], ss.D


# ------------------------------------------------------- model reduction

def gramians(ss):
    P = sla.solve_continuous_lyapunov(ss.A, -ss.B @ ss.B.T)
    Q = sla.solve_continuous_lyapunov(ss.A.T, -ss.C.T @ ss.C)
    return P, Q


def hankel_sv(ss):
    P, Q = gramians(ss)
    P = 0.5 * (P + P.T)
    Q = 0.5 * (Q + Q.T)
    Lp = _psd_chol(P)
    Lq = _psd_chol(Q)
    return np.linalg.svd(Lq.T @ Lp, compute_uv=False), Lp, Lq


def _psd_chol(X):
    """Cholesky-like factor tolerant of tiny negative eigenvalues."""
    try:
        return np.linalg.cholesky(X)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(X)
        w = np.clip(w, 0.0, None)
        return V @ np.diag(np.sqrt(w))


def balanced_truncate(ss, order=None, tol=None, verbose=False):
    """Square-root balanced truncation of a stable continuous system.

    order : keep this many states
    tol   : keep states with hankel_sv > tol * hankel_sv[0]

    Balanced truncation preserves stability but NOT passivity -- run the
    passivity pass afterwards.
    """
    if not ss.is_stable():
        raise ValueError("balanced truncation needs a stable system")
    hsv, Lp, Lq = hankel_sv(ss)
    U, S, Vt = np.linalg.svd(Lq.T @ Lp)
    if order is None:
        if tol is None:
            tol = 1e-6
        order = int(np.sum(S > tol * S[0]))
        order = max(order, 2)
    order = min(order, ss.n_states)
    if order == ss.n_states:
        out = StateSpace(ss.A, ss.B, ss.C, ss.D, ss.E, ss.meta)
        out.meta["hsv"] = S
        return out

    S1 = S[:order]
    si = np.diag(1.0 / np.sqrt(S1))
    T = Lp @ Vt[:order].T @ si            # n x r
    Ti = si @ U[:, :order].T @ Lq.T       # r x n
    A = Ti @ ss.A @ T
    B = Ti @ ss.B
    C = ss.C @ T
    if verbose:
        drop = S[order:]
        bound = 2 * drop.sum() if drop.size else 0.0
        print(f"  balanced truncation {ss.n_states} -> {order} states, "
              f"H-inf error bound <= {bound:.3e}")
    out = StateSpace(A, B, C, ss.D, ss.E, ss.meta)
    out.meta["hsv"] = S
    out.meta["trunc_bound"] = float(2 * S[order:].sum()) if order < len(S) else 0.0
    return out


# ------------------------------------------------------- modal (sparse) form

class BalancedBasis:
    """Cached square-root balancing of a stable system.

    The expensive part (two Lyapunov solves, two Choleskys, one SVD) depends
    only on the system, not on the retained order, so it is computed once and
    reused for every candidate order during an order search.
    """

    def __init__(self, ss):
        if not ss.is_stable():
            raise ValueError("balanced truncation needs a stable system")
        P, Q = gramians(ss)
        self.Lp = _psd_chol(0.5 * (P + P.T))
        self.Lq = _psd_chol(0.5 * (Q + Q.T))
        self.U, self.S, self.Vt = np.linalg.svd(self.Lq.T @ self.Lp)
        self.ss = ss

    @property
    def hsv(self):
        return self.S

    def degenerate_split(self, order, rtol=1e-3):
        """True if truncating here splits a near-degenerate HSV pair.

        Cutting between two nearly equal Hankel singular values makes the
        balancing transform ill-conditioned and the truncation numerically
        poor, which is the main reason the reduced-order error is not monotone
        in the retained order.
        """
        if order <= 0 or order >= len(self.S):
            return False
        a, b = self.S[order - 1], self.S[order]
        return abs(a - b) <= rtol * max(a, 1e-300)

    def truncate(self, order):
        ss = self.ss
        order = int(min(max(order, 1), ss.n_states))
        if order == ss.n_states:
            out = StateSpace(ss.A, ss.B, ss.C, ss.D, ss.E, ss.meta)
        else:
            si = np.diag(1.0 / np.sqrt(self.S[:order]))
            T = self.Lp @ self.Vt[:order].T @ si
            Ti = si @ self.U[:, :order].T @ self.Lq.T
            out = StateSpace(Ti @ ss.A @ T, Ti @ ss.B, ss.C @ T, ss.D, ss.E,
                             ss.meta)
        out.meta = dict(ss.meta)
        out.meta["hsv"] = self.S
        out.meta["trunc_bound"] = (float(2 * self.S[order:].sum())
                                   if order < len(self.S) else 0.0)
        return out


def reduce_to_target(ss, err_fn, target, verbose=False, max_evals=40):
    """Smallest retained order whose error meets `target`.

    `err_fn(candidate) -> float`.  The search is a coarse geometric ladder
    followed by a downward linear refinement rather than a bisection: the
    reduced-order error is NOT monotone in the retained order (near-degenerate
    Hankel singular values make some cuts much worse than deeper ones), so
    bisection can and does miss far smaller models that meet the target.
    """
    basis = BalancedBasis(ss)
    n = ss.n_states
    ladder = sorted(set(int(round(v)) for v in
                        np.geomspace(2, n, min(max_evals, n))) | {n})
    cache = {}

    def ev(k):
        if k not in cache:
            if basis.degenerate_split(k):
                cache[k] = np.inf
            else:
                try:
                    cache[k] = err_fn(basis.truncate(k))
                except Exception:
                    cache[k] = np.inf
        return cache[k]

    best = None
    for k in ladder:
        if ev(k) <= target:
            best = k
            break
    if best is None:
        return None, basis, cache

    lower = [k for k in ladder if k < best]
    floor_k = max(lower) if lower else 2
    for k in range(best - 1, floor_k - 1, -1):
        if ev(k) <= target:
            best = k
        elif best - k > 6:
            break
    if verbose:
        print(f"  order search: {len(cache)} candidates evaluated, "
              f"chose {best} (err {cache[best]:.3e})")
    return basis.truncate(best), basis, cache


def to_modal(ss, tol=1e-9):
    """Similarity transform to real block-diagonal (modal) A.

    A becomes 1x1 blocks for real eigenvalues and 2x2 [[sig, om], [-om, sig]]
    blocks for conjugate pairs.  This is what makes the emitted RNM code cheap:
    the state update costs ~2 multiplies per state instead of n_states.

    For a conjugate pair with eigenvector v = vr + j*vi,
        A vr = sig*vr - om*vi ,  A vi = om*vr + sig*vi
    so [vr | vi] is the real invariant basis for the 2x2 block above.
    """
    lam, V = np.linalg.eig(ss.A)
    n = ss.n_states
    used = np.zeros(n, dtype=bool)
    cols, blocks = [], []
    for i in range(n):
        if used[i]:
            continue
        li = lam[i]
        if abs(li.imag) <= tol * max(abs(li), 1.0):
            used[i] = True
            v = V[:, i]
            v = v.real if np.linalg.norm(v.real) >= np.linalg.norm(v.imag) else v.imag
            cols.append(v / max(np.linalg.norm(v), 1e-300))
            blocks.append(np.array([[li.real]]))
        else:
            j = next((k for k in range(i + 1, n)
                      if not used[k] and abs(lam[k] - np.conj(li))
                      <= tol * max(abs(li), 1.0)), None)
            used[i] = True
            if j is None:
                cols.append(V[:, i].real / max(np.linalg.norm(V[:, i].real), 1e-300))
                blocks.append(np.array([[li.real]]))
                continue
            used[j] = True
            v = V[:, i]
            scale = max(np.linalg.norm(v), 1e-300)
            cols += [v.real / scale, v.imag / scale]
            blocks.append(np.array([[li.real, li.imag], [-li.imag, li.real]]))

    T = np.column_stack(cols)
    if np.linalg.cond(T) > 1e12:
        raise np.linalg.LinAlgError(
            "eigenbasis is ill-conditioned (near-defective A); "
            "use --form dense or reduce the model order")
    Ti = np.linalg.inv(T)
    A = sla.block_diag(*blocks)
    out = StateSpace(A, Ti @ ss.B, ss.C @ T, ss.D, ss.E, ss.meta)
    out.meta["modal"] = True
    out.meta["block_sizes"] = [b.shape[0] for b in blocks]
    return out


def block_structure(A, tol=1e-10):
    """Detect 1x1 / 2x2 block-diagonal structure in A (for sparse emission)."""
    n = A.shape[0]
    sizes = []
    i = 0
    scale = max(np.abs(A).max(), 1e-300)
    while i < n:
        if i + 1 < n and abs(A[i, i + 1]) > tol * scale:
            sizes.append(2)
            i += 2
        else:
            sizes.append(1)
            i += 1
    # verify everything off the blocks is negligible
    mask = np.ones((n, n), dtype=bool)
    k = 0
    for sz in sizes:
        mask[k:k + sz, k:k + sz] = False
        k += sz
    ok = np.abs(A[mask]).max() <= tol * scale if mask.any() else True
    return (sizes if ok else [n]), bool(ok)


# ------------------------------------------------------- discretization

class DiscreteSS:
    """x[n+1] = Ad x[n] + Bd u[n];  y[n] = Cd x[n] + Dd u[n] + (E/Ts)(u[n]-u[n-1])

    The optional last term realizes the continuous s*E (series port inductance)
    as a one-tap FIR.  That is EXACT whenever the input is piecewise linear over
    a sample -- which is the honest reading of an RNM current that is only
    defined at sample instants -- it is unconditionally stable (no poles), and
    at low frequency (E/Ts)(1 - z^-1) -> jw*E as required.  The bilinear
    alternative would place a pole at z = -1 and ring at Nyquist forever.

    A symmetric real E is lossless (Re{jw*E} = 0), so it does not enter the
    passivity test at all.
    """

    def __init__(self, Ad, Bd, Cd, Dd, Ts, method, Ee=None, meta=None):
        self.Ad = np.asarray(Ad, float)
        self.Bd = np.asarray(Bd, float)
        self.Cd = np.asarray(Cd, float)
        self.Dd = np.asarray(Dd, float)
        self.Ts = float(Ts)
        self.method = method
        self.Ee = Ee              # (gain, ) tuple for the bilinear s*E branch
        self.meta = dict(meta or {})

    @property
    def n_states(self):
        return self.Ad.shape[0]

    @property
    def n_ports(self):
        return self.Bd.shape[1]

    def spectral_radius(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.Ad))))

    def is_stable(self):
        return self.spectral_radius() < 1.0

    def __repr__(self):
        return (f"<DiscreteSS n={self.n_states} ports={self.n_ports} "
                f"Ts={self.Ts:.4g}s rho={self.spectral_radius():.6f} "
                f"{self.method}>")

    def freqresp(self, f):
        """Discrete-time response at frequency f (Hz)."""
        z = np.exp(2j * np.pi * np.asarray(f, float) * self.Ts)
        I = np.eye(self.n_states)
        out = np.empty((len(z), self.Cd.shape[0], self.Bd.shape[1]), dtype=complex)
        for k, zk in enumerate(z):
            out[k] = self.Cd @ np.linalg.solve(zk * I - self.Ad, self.Bd) + self.Dd
            if self.Ee is not None:
                out[k] += self.Ee * (1.0 - 1.0 / zk) / self.Ts
        return out

    def simulate(self, u, x0=None):
        """u : (Nt, n_ports) -> y : (Nt, n_ports)."""
        u = np.atleast_2d(np.asarray(u, float))
        if u.shape[1] != self.n_ports:
            u = u.T
        nt = u.shape[0]
        x = np.zeros(self.n_states) if x0 is None else np.array(x0, float)
        y = np.zeros((nt, self.Cd.shape[0]))
        ye = np.zeros(self.Cd.shape[0])
        uprev = np.zeros(self.n_ports)
        for k in range(nt):
            y[k] = self.Cd @ x + self.Dd @ u[k]
            if self.Ee is not None:
                y[k] += self.Ee @ (u[k] - uprev) / self.Ts
                uprev = u[k].copy()
            x = self.Ad @ x + self.Bd @ u[k]
        return y


def discretize(ss, Ts, method="zoh"):
    """Continuous -> discrete.

    'zoh'     : exact for piecewise-constant input.  This is what an RNM
                current source actually produces, so it is the default.
    'bilinear': Tustin, preserves passivity exactly (maps LHP -> unit disc and
                the jw axis onto the unit circle), but warps frequency.
    'foh'     : first-order hold, better for ramped stimulus.
    """
    n, m = ss.n_states, ss.n_ports
    A, B, C, D = ss.A, ss.B, ss.C, ss.D
    Ee = None

    if ss.E is not None:
        Ee = 0.5 * (ss.E + ss.E.T)         # symmetric part = the inductance

    if method == "zoh":
        Mx = np.zeros((n + m, n + m))
        Mx[:n, :n] = A * Ts
        Mx[:n, n:] = B * Ts
        Ex = sla.expm(Mx)
        Ad, Bd = Ex[:n, :n], Ex[:n, n:]
        Cd, Dd = C, D
    elif method in ("bilinear", "tustin"):
        I = np.eye(n)
        M1 = I - A * (Ts / 2)
        M2 = I + A * (Ts / 2)
        Ad = np.linalg.solve(M1, M2)
        Bd = np.linalg.solve(M1, B) * Ts
        Cd = np.linalg.solve(M1.T, C.T).T
        Dd = D + 0.5 * (C @ Bd)
    elif method == "foh":
        Mx = np.zeros((n + 2 * m, n + 2 * m))
        Mx[:n, :n] = A * Ts
        Mx[:n, n:n + m] = B * Ts
        Mx[n:n + m, n + m:] = np.eye(m)
        Ex = sla.expm(Mx)
        Ad = Ex[:n, :n]
        G1 = Ex[:n, n:n + m]
        G2 = Ex[:n, n + m:]
        Bd = G1 - G2 + Ad @ G2
        Cd, Dd = C, D + C @ G2
    else:
        raise ValueError(f"unknown discretization method {method!r}")

    return DiscreteSS(Ad, Bd, Cd, Dd, Ts, method, Ee, ss.meta)


def suggest_timestep(ss, ppc=20, fmax=None):
    """Timestep guidance.

    ZOH is exact for the state part under a staircase input, so the binding
    constraint is resolving the highest frequency you care about: ppc samples
    per cycle at `fmax` (defaulting to the fastest retained pole).  If the model
    carries a series inductance (E), the FIR branch adds a half-sample lag whose
    relative error is about pi*f*Ts = pi/ppc, so ppc also sets that accuracy:
    ppc = 20 -> ~16%% on the inductive part alone at f = fs/20, ppc = 100 -> ~3%%.
    In practice pick ppc from the bandwidth that actually matters for your
    droop, not from the top of the Touchstone file.
    """
    fp = ss.max_pole_freq() if fmax is None else fmax
    return 1.0 / (ppc * max(fp, 1e-30))
