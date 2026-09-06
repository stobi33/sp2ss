"""Vector fitting (Gustavsen & Semlyen) with relaxed non-triviality.

Fits a set of K frequency responses sharing a COMMON pole set:

    f_k(s) ~= sum_m  c_km / (s - p_m)  +  d_k  +  s * e_k

References
----------
B. Gustavsen, A. Semlyen, "Rational approximation of frequency domain
responses by vector fitting", IEEE Trans. Power Delivery, 14(3), 1999.
B. Gustavsen, "Improving the pole relocating properties of vector fitting",
IEEE Trans. Power Delivery, 21(3), 2006.  (the 'relaxed' variant used here)

Everything is done in real arithmetic: complex-conjugate pole pairs are
represented by the two real basis functions

    1/(s-p) + 1/(s-p*)      and      j/(s-p) - j/(s-p*)

so the unknown residues stay real and conjugacy is structurally exact.
"""
from __future__ import annotations

import numpy as np

__all__ = ["start_poles", "vector_fit", "model_response", "VFResult",
           "prune_poles", "make_weight", "fit_errors"]


# ---------------------------------------------------------------- helpers

def _pole_index(poles):
    """Classify poles: 0 = real, 1 = first of a conj pair, 2 = second."""
    idx = np.zeros(len(poles), dtype=int)
    m = 0
    while m < len(poles):
        if abs(poles[m].imag) > 0:
            if m + 1 < len(poles) and np.isclose(poles[m + 1], np.conj(poles[m])):
                idx[m], idx[m + 1] = 1, 2
                m += 2
                continue
            raise ValueError("complex poles must appear as adjacent conjugate pairs")
        idx[m] = 0
        m += 1
    return idx


def _order_poles(poles):
    """Sort into real poles first, then conjugate pairs stored (p, p*)."""
    poles = np.asarray(poles, dtype=complex)
    real = np.sort(poles[np.abs(poles.imag) < 1e-12].real)
    cplx = poles[np.abs(poles.imag) >= 1e-12]
    cplx = cplx[cplx.imag > 0]
    cplx = cplx[np.argsort(cplx.imag)]
    out = [complex(r, 0.0) for r in real]
    for p in cplx:
        out += [p, np.conj(p)]
    return np.array(out, dtype=complex)


def _basis(s, poles, cidx):
    """Real-residue partial-fraction basis, shape (Ns, M) complex."""
    D = np.empty((len(s), len(poles)), dtype=complex)
    m = 0
    while m < len(poles):
        if cidx[m] == 0:
            D[:, m] = 1.0 / (s - poles[m])
            m += 1
        else:
            a = 1.0 / (s - poles[m])
            b = 1.0 / (s - poles[m + 1])
            D[:, m] = a + b
            D[:, m + 1] = 1j * (a - b)
            m += 2
    return D


def _augment(D, s, asymp):
    """Append the d (constant) and e (s*) columns."""
    cols = [D]
    if asymp >= 1:
        cols.append(np.ones((len(s), 1), dtype=complex))
    if asymp >= 2:
        cols.append(s.reshape(-1, 1))
    return np.hstack(cols)


def _realify(M, rhs):
    """Stack real and imaginary parts so the LS solution is real."""
    A = np.vstack([M.real, M.imag])
    b = np.concatenate([rhs.real, rhs.imag])
    return A, b


def _scaled_lstsq(A, b, rcond=1e-14):
    """Column-equilibrated least squares -- essential for VF conditioning."""
    nrm = np.linalg.norm(A, axis=0)
    nrm[nrm == 0] = 1.0
    x, *_ = np.linalg.lstsq(A / nrm, b, rcond=rcond)
    return x / nrm


def _asymp_code(asymp):
    a = str(asymp).lower()
    return {"none": 0, "": 0, "0": 0, "d": 1, "1": 1, "de": 2, "2": 2}[a]


# ---------------------------------------------------------------- startup

def start_poles(fmin, fmax, order, kind="log", alpha=0.01, n_real=0):
    """Standard VF starting poles: conjugate pairs spread over the band.

    Real part is -alpha * imag part (Gustavsen recommends alpha = 1/100), which
    gives lightly damped starting poles that relocate readily.
    """
    n_pairs = (order - n_real) // 2
    if n_pairs < 1:
        raise ValueError("order too small")
    lo, hi = max(fmin, 1e-30), fmax
    if kind == "log":
        w = 2 * np.pi * np.logspace(np.log10(lo), np.log10(hi), n_pairs)
    else:
        w = 2 * np.pi * np.linspace(lo, hi, n_pairs)
    poles = []
    if n_real:
        wr = 2 * np.pi * np.logspace(np.log10(lo), np.log10(hi), n_real)
        poles += [complex(-x, 0.0) for x in wr]
    for wi in w:
        p = complex(-alpha * wi, wi)
        poles += [p, np.conj(p)]
    return _order_poles(np.array(poles))


def make_weight(f, mode="inv", floor=1e-12, abs_floor=0.0):
    """Sample weights, shape broadcastable to (K, Ns).

    'inv'  : 1/|f|      -> minimise RELATIVE error (use this for PDN Z)
    'sqrt' : 1/sqrt|f|  -> compromise
    'none' : 1

    `abs_floor` clamps the weight denominator to an absolute value (ohms for a
    Z fit).  Relative weighting is what makes a PDN fit usable across four
    decades of impedance, but taken literally it also chases the VNA noise
    floor: below roughly 1 mohm a 50-ohm-referenced S-parameter measurement is
    mostly noise, and an unclamped 1/|Z| weight will spend the model order
    fitting it.  Set abs_floor to the impedance you actually trust.
    """
    f = np.atleast_2d(f)
    mag = np.abs(f)
    # floor PER ELEMENT, not against the global maximum: a transfer impedance
    # can sit four decades below the driving-point one, and a global floor
    # gives its near-zero samples a near-infinite weight.
    scale = np.maximum(mag, floor * mag.max(axis=1, keepdims=True))
    if abs_floor > 0:
        scale = np.maximum(scale, abs_floor)
    if mode == "inv":
        return 1.0 / scale
    if mode == "sqrt":
        return 1.0 / np.sqrt(scale)
    if mode == "none":
        return np.ones_like(mag)
    raise ValueError(f"unknown weight mode {mode!r}")


# ---------------------------------------------------------------- the fit

class VFResult:
    def __init__(self, poles, residues, d, e, rms, history):
        self.poles = poles            # (M,) complex, conj pairs adjacent
        self.residues = residues      # (K, M) complex, conj-consistent
        self.d = d                    # (K,) real
        self.e = e                    # (K,) real
        self.rms = rms                # weighted rms residual of the last pass
        self.history = history        # rms per iteration

    @property
    def order(self):
        return len(self.poles)

    def __repr__(self):
        nr = int(np.sum(np.abs(self.poles.imag) < 1e-12))
        return (f"<VFResult order={self.order} ({nr} real, "
                f"{(self.order-nr)//2} pairs) rms={self.rms:.3e}>")


def _pole_relocation(F, s, poles, W, asymp, relax):
    """One VF iteration: returns the relocated pole set."""
    K, Ns = F.shape
    cidx = _pole_index(poles)
    M = len(poles)
    D = _basis(s, poles, cidx)
    Aloc_cols = M + asymp
    Aglb_cols = M + (1 if relax else 0)

    # Per-element QR compression (Gustavsen's "fast VF"): only the rows that
    # couple to the shared sigma-unknowns need to reach the global system.
    blocks = []
    for k in range(K):
        w = W[k][:, None]
        A1 = _augment(D, s, asymp) * w
        A2 = -(D * F[k][:, None]) * w
        if relax:
            A2 = np.hstack([A2, (-F[k][:, None]) * w])
        Ak, _ = _realify(np.hstack([A1, A2]), np.zeros(Ns))
        # economy QR; keep the trailing block of R
        _, R = np.linalg.qr(Ak, mode="reduced")
        ncols = Ak.shape[1]
        keep = min(R.shape[0], ncols)
        R = R[:keep, :]
        blocks.append(R[Aloc_cols:, Aloc_cols:])
    AA = np.vstack(blocks)
    bb = np.zeros(AA.shape[0])

    if relax:
        # Non-triviality constraint: Re{ sum_n sigma(s_n) } = Ns  (relaxed form)
        scale = np.linalg.norm(np.abs(F).ravel()) / max(F.size, 1)
        row = np.concatenate([(scale * D.sum(axis=0)).real, [scale * Ns]])
        AA = np.vstack([AA, row])
        bb = np.concatenate([bb, [scale * Ns]])

    x = _scaled_lstsq(AA, bb)
    if relax:
        csig, dsig = x[:M], x[M]
        if abs(dsig) < 1e-8 * max(np.abs(csig).max(), 1e-30):
            dsig = 1.0                      # degenerate -> fall back to classic VF
    else:
        csig, dsig = x[:M], 1.0
    csig = csig / dsig

    # Relocated poles = eigenvalues of (A - b c^T) in the real modal basis
    Am = np.zeros((M, M))
    bvec = np.zeros(M)
    m = 0
    while m < M:
        if cidx[m] == 0:
            Am[m, m] = poles[m].real
            bvec[m] = 1.0
            m += 1
        else:
            sr, si = poles[m].real, poles[m].imag
            Am[m, m] = Am[m + 1, m + 1] = sr
            Am[m, m + 1] = si
            Am[m + 1, m] = -si
            bvec[m], bvec[m + 1] = 2.0, 0.0
            m += 2
    H = Am - np.outer(bvec, csig)
    new = np.linalg.eigvals(H)
    return _order_poles(new)


def _clip_poles(poles, w_max, w_min, clip_hi=10.0, clip_lo=1e-3):
    """Keep relocated poles inside a sane magnitude window.

    A pole far above the data band is numerically DEGENERATE with the constant
    term d, because R/(s-p) -> -R/p over the whole band.  Left unchecked, VF
    happily runs off into that direction and produces enormous cancelling
    residue/d pairs (seen in practice: p ~ 1e14 Hz, R ~ 1e20, d ~ 1e5 for a
    milliohm PDN).  The fit error looks fine but the realization has a ~1e8
    internal dynamic range, which destroys balanced truncation and costs real
    precision in the emitted fixed-period HDL.  Reflecting such poles back to
    the band edge keeps them useful instead of degenerate.
    """
    out = poles.copy()
    mag = np.abs(out)
    hi = clip_hi * w_max
    lo = clip_lo * w_min
    too_big = mag > hi
    if too_big.any():
        out[too_big] = out[too_big] / mag[too_big] * hi
    too_small = (mag < lo) & (mag > 0)
    if too_small.any():
        out[too_small] = out[too_small] / mag[too_small] * lo
    # clipping can push a pole onto/over the jw axis; re-damp it
    bad = out.real >= 0
    if bad.any():
        out[bad] = -np.abs(out[bad].real) - 1e-12 * np.abs(out[bad]) + 1j * out[bad].imag
        out[bad] = out[bad] - 2 * np.maximum(out[bad].real, 0)
    return _order_poles(out)


def prune_poles(F, s, poles, W, asymp, w_max, K=10.0, res_tol=1e-12,
                verbose=False):
    """Drop poles that contribute nothing but a constant over the fitted band.

    Any pole with |p| > K*w_max is effectively a constant in band, so it is
    removed and the residues/d are re-solved -- d absorbs its contribution and
    the model loses a spurious cancellation pair.  Poles whose residues are
    negligible relative to the largest are dropped too.
    """
    poles = _order_poles(np.asarray(poles, dtype=complex))
    R, d, e = _residue_solve(F, s, poles, W, asymp)

    keep = np.ones(len(poles), dtype=bool)
    keep &= np.abs(poles) <= K * w_max
    rmag = np.abs(R).max(axis=0)
    keep &= rmag > res_tol * max(rmag.max(), 1e-300)
    # never split a conjugate pair
    cidx = _pole_index(poles)
    m = 0
    while m < len(poles):
        if cidx[m] == 0:
            m += 1
        else:
            both = keep[m] and keep[m + 1]
            keep[m] = keep[m + 1] = both
            m += 2

    if keep.all() or keep.sum() < 2:
        return poles, R, d, e, 0
    dropped = int((~keep).sum())
    poles = _order_poles(poles[keep])
    R, d, e = _residue_solve(F, s, poles, W, asymp)
    if verbose:
        print(f"  pruned {dropped} out-of-band / negligible pole(s) -> "
              f"order {len(poles)}")
    return poles, R, d, e, dropped


def _residue_solve(F, s, poles, W, asymp):
    """Final linear stage: residues, d, e for the fixed pole set."""
    K, Ns = F.shape
    cidx = _pole_index(poles)
    M = len(poles)
    D = _augment(_basis(s, poles, cidx), s, asymp)

    C = np.zeros((K, M))
    d = np.zeros(K)
    e = np.zeros(K)
    for k in range(K):
        w = W[k][:, None]
        A, b = _realify(D * w, F[k] * W[k])
        x = _scaled_lstsq(A, b)
        C[k] = x[:M]
        if asymp >= 1:
            d[k] = x[M]
        if asymp >= 2:
            e[k] = x[M + 1]

    # real coefficients -> complex conjugate residues
    R = np.zeros((K, M), dtype=complex)
    m = 0
    while m < M:
        if cidx[m] == 0:
            R[:, m] = C[:, m]
            m += 1
        else:
            R[:, m] = C[:, m] + 1j * C[:, m + 1]
            R[:, m + 1] = np.conj(R[:, m])
            m += 2
    return R, d, e


def model_response(s, poles, residues, d, e):
    """Evaluate sum_m R/(s-p) + d + s e. Returns (K, Ns)."""
    R = np.atleast_2d(residues)
    d = np.atleast_1d(d)
    e = np.atleast_1d(e)
    out = R @ (1.0 / (s[None, :] - poles[:, None]))
    return out + d[:, None] + np.outer(e, s)


def vector_fit(F, s, order=None, poles=None, n_iter=8, asymp="d",
               weight="inv", relax=True, stable=True, tol=1e-12, verbose=False,
               scale=True, clip=0.0, prune=True, weight_floor=0.0):
    """Fit K responses with a common pole set.

    Parameters
    ----------
    F      : (K, Ns) complex responses
    s      : (Ns,) complex, = j*2*pi*f  (an s == 0 sample is allowed)
    order  : model order (number of poles) if `poles` is not given
    poles  : (M,) starting poles; overrides `order`
    n_iter : pole relocation iterations
    asymp  : 'none' | 'd' | 'de'  -- include the constant / s-proportional terms
    weight : 'inv' | 'sqrt' | 'none', or an explicit (K,Ns) array
    weight_floor : absolute floor for the relative-weight denominator
             (ohms for Z) -- stops the fit chasing the noise floor
    stable : reflect any right-half-plane pole back into the LHP
    scale  : internally normalise s by the geometric-mean band frequency
             (conditioning; results are unscaled on the way out)
    clip   : cap |pole| at clip * w_max during relocation (0 = off, the
             default -- clipping piles poles up at the clip radius; use
             asymp='de' instead if the fit runs off to huge poles)
    prune  : drop out-of-band / negligible poles after the last iteration

    Returns
    -------
    VFResult
    """
    F = np.atleast_2d(np.asarray(F, dtype=complex))
    s = np.asarray(s, dtype=complex)
    K, Ns = F.shape
    if s.shape[0] != Ns:
        raise ValueError("s and F have mismatched lengths")
    a = _asymp_code(asymp)

    if isinstance(weight, str):
        W = make_weight(F, weight, abs_floor=weight_floor)
    else:
        W = np.broadcast_to(np.asarray(weight, dtype=float), (K, Ns)).copy()

    if poles is None:
        if order is None:
            raise ValueError("give either order or poles")
        fpos = np.abs(s.imag) / (2 * np.pi)
        fpos = fpos[fpos > 0]
        poles = start_poles(fpos.min(), fpos.max(), order)
    poles = _order_poles(np.asarray(poles, dtype=complex))

    # ---- internal frequency normalisation --------------------------------
    wmag = np.abs(s.imag)
    wnz = wmag[wmag > 0]
    w_max_true = float(wnz.max())
    w_min_true = float(wnz.min())
    w0 = float(np.sqrt(w_min_true * w_max_true)) if scale else 1.0
    sN = s / w0
    polesN = poles / w0
    w_max, w_min = w_max_true / w0, w_min_true / w0

    history = []
    prev = np.inf
    for it in range(max(n_iter, 0)):
        polesN = _pole_relocation(F, sN, polesN, W, a, relax)
        if stable:
            bad = polesN.real > 0
            if bad.any():
                polesN = _order_poles(polesN - 2 * polesN.real * bad)
        if clip:
            polesN = _clip_poles(polesN, w_max, w_min, clip_hi=clip)
        R, d, e = _residue_solve(F, sN, polesN, W, a)
        err = model_response(sN, polesN, R, d, e) - F
        rms = float(np.sqrt(np.mean(np.abs(err * W) ** 2)))
        history.append(rms)
        if verbose:
            print(f"  VF iter {it+1:2d}/{n_iter}  weighted rms = {rms:.4e}")
        if abs(prev - rms) < tol * max(rms, 1e-300):
            break
        prev = rms

    if not history:                       # n_iter == 0: residues only
        R, d, e = _residue_solve(F, sN, polesN, W, a)
        err = model_response(sN, polesN, R, d, e) - F
        history = [float(np.sqrt(np.mean(np.abs(err * W) ** 2)))]

    if prune:
        polesN, R, d, e, ndrop = prune_poles(F, sN, polesN, W, a, w_max,
                                             verbose=verbose)
        if ndrop:
            err = model_response(sN, polesN, R, d, e) - F
            history.append(float(np.sqrt(np.mean(np.abs(err * W) ** 2))))

    # ---- undo the frequency normalisation --------------------------------
    poles = polesN * w0
    R = R * w0
    e = e / w0

    return VFResult(poles, R, d, e, history[-1], history)


def fit_errors(F, Fm, floor=0.0):
    """Per-element and overall error metrics.

    `floor` (ohms, for a Z fit) is the same clamp used by `make_weight`.  The
    plain relative error is meaningless wherever the data itself is below the
    measurement noise -- a 14 uohm transfer impedance measured through a 50 ohm
    reference is noise, and dividing by it reports hundreds of percent for a
    model that is in fact fine.  `max_rel_floored` divides by max(|Z|, floor)
    instead, which is what the weighted fit actually minimises.
    """
    F = np.atleast_2d(F)
    Fm = np.atleast_2d(Fm)
    absres = np.abs(Fm - F)
    den = np.maximum(np.abs(F), 1e-300)
    rel = absres / den
    denf = np.maximum(den, floor) if floor > 0 else den
    relf = absres / denf
    return {
        "max_rel": float(rel.max()),
        "rms_rel": float(np.sqrt(np.mean(rel ** 2))),
        "max_rel_floored": float(relf.max()),
        "rms_rel_floored": float(np.sqrt(np.mean(relf ** 2))),
        "floor": float(floor),
        "max_rel_per_elem": rel.max(axis=1),
        "rms_rel_per_elem": np.sqrt(np.mean(rel ** 2, axis=1)),
        "max_abs": float(absres.max()),
        "data_min_abs": float(np.abs(F).min()),
    }
