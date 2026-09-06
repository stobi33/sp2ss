"""Touchstone (.sNp) reader/writer -- v1 with minimal v2 support.

Handles the two classic gotchas:
  * 2-port data order is S11 S21 S12 S22 (column major), every other N is row major.
  * v1 noise data may trail the network data in a .s2p.
"""
from __future__ import annotations

import os
import re
import numpy as np

_FREQ_MULT = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9, "THZ": 1e12}
_PARAMS = {"S", "Y", "Z", "G", "H"}
_FORMATS = {"DB", "MA", "RI"}


class Network:
    """N-port frequency-domain network data.

    Attributes
    ----------
    f    : (Ns,) float, frequency in Hz, strictly increasing
    data : (Ns, N, N) complex, the network parameter matrix
    kind : 'S' | 'Y' | 'Z' | ...
    z0   : (N,) float, per-port reference impedance
    """

    def __init__(self, f, data, kind="S", z0=50.0, name=""):
        self.f = np.asarray(f, dtype=float)
        self.data = np.asarray(data, dtype=complex)
        self.kind = kind.upper()
        n = self.data.shape[1]
        self.z0 = np.broadcast_to(np.asarray(z0, dtype=float), (n,)).copy()
        self.name = name

    # -- basics ---------------------------------------------------------
    @property
    def n_ports(self):
        return self.data.shape[1]

    @property
    def n_freq(self):
        return self.f.shape[0]

    @property
    def s(self):
        """Laplace variable on the jw axis."""
        return 1j * 2.0 * np.pi * self.f

    def __repr__(self):
        return (f"<Network {self.name!r} {self.kind} {self.n_ports}-port "
                f"{self.n_freq} pts {self.f[0]:.4g}..{self.f[-1]:.4g} Hz>")

    def reciprocity_error(self):
        """max |X_ij - X_ji| / max|X|, 0 for a perfectly reciprocal network."""
        d = self.data
        num = np.abs(d - np.swapaxes(d, 1, 2)).max()
        den = np.abs(d).max()
        return float(num / den) if den > 0 else 0.0

    def symmetrized(self):
        """Force reciprocity: X <- (X + X^T)/2."""
        d = 0.5 * (self.data + np.swapaxes(self.data, 1, 2))
        return Network(self.f, d, self.kind, self.z0, self.name)

    def subset(self, ports):
        """Keep a subset of ports (index submatrix -- see convert.subset_ports
        for the open/short semantics)."""
        idx = np.asarray(ports, dtype=int)
        d = self.data[np.ix_(np.arange(self.n_freq), idx, idx)]
        return Network(self.f, d, self.kind, self.z0[idx], self.name)


# -- reading -----------------------------------------------------------

def _strip(line):
    """Remove Touchstone comments, keep the payload."""
    i = line.find("!")
    return (line if i < 0 else line[:i]).strip()


def _ports_from_ext(path):
    m = re.search(r"\.[syzgh](\d+)p$", str(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _to_complex(a, b, fmt):
    if fmt == "RI":
        return a + 1j * b
    if fmt == "MA":
        return a * np.exp(1j * np.deg2rad(b))
    if fmt == "DB":
        return 10.0 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))
    raise ValueError(f"bad format {fmt}")


def read_touchstone(path):
    """Parse a Touchstone file into a :class:`Network`."""
    with open(path, "r", errors="replace") as fh:
        raw = fh.readlines()

    fmult, param, fmt, z0scalar = 1e9, "S", "MA", 50.0
    n_ports = _ports_from_ext(path)
    z0_vec = None
    version2 = False
    two_port_order = "21_12"          # Touchstone v1 default for N == 2

    body = []
    in_network = True                  # v1: everything after '#' is network data
    for line in raw:
        t = _strip(line)
        if not t:
            continue
        if t.startswith("#"):
            tok = t[1:].upper().split()
            i = 0
            while i < len(tok):
                w = tok[i]
                if w in _FREQ_MULT:
                    fmult = _FREQ_MULT[w]
                elif w in _PARAMS:
                    param = w
                elif w in _FORMATS:
                    fmt = w
                elif w == "R":
                    i += 1
                    vals = []
                    while i < len(tok) and re.match(r"^[-+0-9.]", tok[i]):
                        vals.append(float(tok[i]))
                        i += 1
                    if vals:
                        z0scalar = vals[0]
                        if len(vals) > 1:
                            z0_vec = np.array(vals)
                    continue
                i += 1
            continue
        if t.startswith("["):
            version2 = True
            kw = t.lower()
            if kw.startswith("[number of ports]"):
                n_ports = int(t.split("]")[1])
            elif kw.startswith("[two-port data order]"):
                two_port_order = "12_21" if "12_21" in kw else "21_12"
            elif kw.startswith("[reference]"):
                z0_vec = np.array([float(x) for x in t.split("]")[1].split()])
            elif kw.startswith("[network data]"):
                in_network = True
            elif kw.startswith(("[noise data]", "[end]")):
                in_network = False
            else:
                in_network = in_network and not kw.startswith("[matrix format]")
            continue
        if in_network:
            body.append(t)

    if n_ports is None:
        raise ValueError(f"cannot determine port count for {path}; "
                         "use a .sNp/.zNp extension or a v2 [Number of Ports] line")

    if version2 and n_ports == 2 and two_port_order == "21_12":
        swap12 = True
    else:
        swap12 = (not version2) and n_ports == 2

    n = n_ports
    per_rec = 1 + 2 * n * n

    # v1 .s2p noise data: 5 numbers per line, appearing after the network block.
    if n == 2 and not version2:
        cut = None
        seen_net = False
        for k, ln in enumerate(body):
            ntok = len(ln.split())
            if ntok >= 9:
                seen_net = True
            elif seen_net and ntok == 5:
                cut = k
                break
        if cut is not None:
            body = body[:cut]

    tok = np.array(" ".join(body).split(), dtype=float)
    if tok.size % per_rec:
        usable = (tok.size // per_rec) * per_rec
        if usable == 0:
            raise ValueError(f"{path}: {tok.size} numbers is not a multiple of "
                             f"{per_rec} for a {n}-port file")
        tok = tok[:usable]
    rec = tok.reshape(-1, per_rec)

    f = rec[:, 0] * fmult
    pairs = rec[:, 1:].reshape(-1, n * n, 2)
    vals = _to_complex(pairs[:, :, 0], pairs[:, :, 1], fmt)
    mat = vals.reshape(-1, n, n)                       # row-major
    if swap12:
        mat = np.swapaxes(mat, 1, 2)                   # S11 S21 S12 S22 -> matrix

    if not np.all(np.diff(f) > 0):
        order = np.argsort(f, kind="stable")
        f, mat = f[order], mat[order]
        keep = np.concatenate(([True], np.diff(f) > 0))
        f, mat = f[keep], mat[keep]

    z0 = z0_vec if z0_vec is not None else np.full(n, z0scalar)
    return Network(f, mat, param, z0, name=os.path.basename(str(path)))


# -- writing -----------------------------------------------------------

def write_touchstone(path, net, fmt="RI", funit="HZ"):
    """Write a Network back out as Touchstone v1 (round-trip / debug aid)."""
    n = net.n_ports
    mult = _FREQ_MULT[funit.upper()]
    lines = [f"! generated by sp2ss",
             f"# {funit.upper()} {net.kind} {fmt.upper()} R {net.z0[0]:g}"]
    for k in range(net.n_freq):
        m = net.data[k]
        if n == 2:
            seq = [m[0, 0], m[1, 0], m[0, 1], m[1, 1]]
        else:
            seq = list(m.reshape(-1))
        nums = []
        for v in seq:
            if fmt.upper() == "RI":
                nums += [v.real, v.imag]
            elif fmt.upper() == "MA":
                nums += [abs(v), np.rad2deg(np.angle(v))]
            else:
                nums += [20 * np.log10(max(abs(v), 1e-300)),
                         np.rad2deg(np.angle(v))]
        row = f"{net.f[k]/mult:.10g}"
        for i, v in enumerate(nums):
            if i and i % 8 == 0:
                row += "\n " + " " * 12
            row += f" {v: .9e}"
        lines.append(row)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
