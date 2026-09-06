"""Network parameter conversions and port bookkeeping.

For a PDN driven by RNM current sources we want Z: current in, voltage out.
All conversions assume a real, per-port reference impedance (the normal case
for a Touchstone file with `R 50`).
"""
from __future__ import annotations

import numpy as np

from .touchstone import Network


def _sqrt_z0(z0):
    return np.diag(np.sqrt(z0)), np.diag(1.0 / np.sqrt(z0))


def s_to_z(net):
    """Z = sqrt(Z0) (I - S)^-1 (I + S) sqrt(Z0)."""
    n = net.n_ports
    I = np.eye(n)
    sq, _ = _sqrt_z0(net.z0)
    out = np.empty_like(net.data)
    conds = np.empty(net.n_freq)
    for k, S in enumerate(net.data):
        M = I - S
        conds[k] = np.linalg.cond(M)
        out[k] = sq @ np.linalg.solve(M, I + S) @ sq
    z = Network(net.f, out, "Z", net.z0, net.name)
    z.cond = conds
    return z


def s_to_y(net):
    """Y = sqrt(Y0) (I + S)^-1 (I - S) sqrt(Y0)."""
    n = net.n_ports
    I = np.eye(n)
    _, isq = _sqrt_z0(net.z0)
    out = np.empty_like(net.data)
    conds = np.empty(net.n_freq)
    for k, S in enumerate(net.data):
        M = I + S
        conds[k] = np.linalg.cond(M)
        out[k] = isq @ np.linalg.solve(M, I - S) @ isq
    y = Network(net.f, out, "Y", net.z0, net.name)
    y.cond = conds
    return y


def z_to_s(net, z0=None):
    """S = (Zn - I)(Zn + I)^-1 with Zn = sqrt(Y0) Z sqrt(Y0)."""
    z0 = net.z0 if z0 is None else np.broadcast_to(z0, (net.n_ports,))
    n = net.n_ports
    I = np.eye(n)
    _, isq = _sqrt_z0(z0)
    out = np.empty_like(net.data)
    for k, Z in enumerate(net.data):
        Zn = isq @ Z @ isq
        out[k] = np.linalg.solve((Zn + I).T, (Zn - I).T).T
    return Network(net.f, out, "S", z0, net.name)


def invert(net):
    """Z <-> Y."""
    out = np.linalg.inv(net.data)
    kind = {"Z": "Y", "Y": "Z"}[net.kind]
    return Network(net.f, out, kind, net.z0, net.name)


def to_param(net, param):
    """Convert any of S/Y/Z to the requested one."""
    param = param.upper()
    if net.kind == param:
        return net
    if net.kind == "S":
        return {"Z": s_to_z, "Y": s_to_y}[param](net)
    if net.kind in ("Z", "Y"):
        if param == "S":
            return z_to_s(net if net.kind == "Z" else invert(net))
        return invert(net)
    raise ValueError(f"cannot convert {net.kind} -> {param}")


def subset_ports(net, ports, unused="open"):
    """Reduce an N-port to a sub-network.

    unused='open'  -> take the Z submatrix  (unused ports left open circuit)
    unused='short' -> take the Y submatrix  (unused ports shorted to the ref)

    For PDN work 'open' is almost always right: unconnected probe/monitor ports
    on the extraction are open in the real system too.
    """
    ports = list(ports)
    if unused == "open":
        z = to_param(net, "Z")
        return z.subset(ports)
    elif unused == "short":
        y = to_param(net, "Y")
        return invert(y.subset(ports))
    raise ValueError("unused must be 'open' or 'short'")


def dc_extrapolate(net, z_dc=None):
    """Prepend an f=0 sample so the fit is pinned at DC.

    z_dc may be
      None      -- linearly extrapolate every entry from the two lowest points
      scalar    -- that DC resistance on every diagonal entry; off-diagonals are
                   still extrapolated from the data
      (N,)      -- per-port diagonal resistances, off-diagonals extrapolated
      (N,N)     -- the complete DC resistance matrix

    Only the diagonal is taken from a scalar/vector on purpose.  Zeroing the
    off-diagonals instead would inject a DC transfer impedance of exactly 0,
    and with relative weighting that single bogus sample gets an essentially
    infinite weight and destroys the whole fit.
    """
    if net.f[0] <= 0:
        return net
    n = net.n_ports
    f0, f1 = net.f[0], net.f[1]
    d0, d1 = net.data[0], net.data[1]
    est = ((d0 * f1 - d1 * f0) / (f1 - f0)).real       # linear extrapolation

    if z_dc is None:
        mat = est
    else:
        z_dc = np.asarray(z_dc, dtype=float)
        if z_dc.ndim == 2:
            mat = z_dc
        else:
            mat = est.copy()
            diag = np.broadcast_to(z_dc.reshape(-1), (n,))
            mat[np.arange(n), np.arange(n)] = diag
    mat = 0.5 * (mat + mat.T)
    f = np.concatenate(([0.0], net.f))
    d = np.concatenate((mat.astype(complex)[None, :, :], net.data), axis=0)
    return Network(f, d, net.kind, net.z0, net.name)


def upper_tri_index(n, symmetric=True):
    """Element (i,j) pairs to fit, and the map back to a full matrix."""
    if symmetric:
        return [(i, j) for i in range(n) for j in range(i, n)]
    return [(i, j) for i in range(n) for j in range(n)]


def flatten(net, elems):
    """(Ns,N,N) -> (K,Ns) in the order given by `elems`."""
    return np.array([net.data[:, i, j] for (i, j) in elems])


def unflatten(vec, elems, n, symmetric=True):
    """(K,...) -> (N,N,...) filling the mirror entries when symmetric."""
    tail = vec.shape[1:]
    out = np.zeros((n, n) + tail, dtype=vec.dtype)
    for k, (i, j) in enumerate(elems):
        out[i, j] = vec[k]
        if symmetric:
            out[j, i] = vec[k]
    return out
