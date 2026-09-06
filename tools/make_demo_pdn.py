"""Generate a synthetic but realistic PDN Touchstone file for testing.

Topology (2-port, both ports referenced to ground):

  port1                                                            port2
  (die)  --Rpkg/Lpkg--  (package)  --Rbrd/Lbrd--  (board/VRM out)
    |                       |                            |
  Cdie                    Cpkg                    Cbulk  |  VRM (Rv+Lv)
  (ESR/ESL)             (ESR/ESL)               (ESR/ESL)|

This gives the classic PDN impedance profile: VRM-controlled at low frequency,
bulk/package/die anti-resonances in the MHz-to-100MHz range, and a die-cap
dominated floor at the top.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sp2ss.touchstone import Network, write_touchstone
from sp2ss.convert import z_to_s


def cap_branch(s, C, esr, esl, n=1):
    """Admittance of n parallel (C + ESR + ESL) branches."""
    z = esr + s * esl + 1.0 / (s * C)
    return n / z


def series(s, R, L):
    return 1.0 / (R + s * L)


def pdn_z_nport(f, n_ports=4):
    """Ladder with `n_ports` tapped nodes: die -> package -> board -> VRM.

    Each node carries a decoupling branch; adjacent nodes are linked by a
    series R/L.  Ports are taken at every node, which is what a multi-port PDN
    extraction from a field solver typically looks like.
    """
    s = 2j * np.pi * f
    # per-node (C, ESR, ESL, count), coarse -> fine as you approach the die
    nodes = [
        (100e-6, 3e-3, 1.5e-9, 4),    # board bulk
        (10e-6, 3e-3, 300e-12, 6),    # board ceramic
        (1e-6, 3e-3, 25e-12, 8),      # package
        (100e-9, 2e-3, 1e-12, 4),     # on die
    ]
    links = [(3e-3, 2e-9), (2e-3, 400e-12), (0.5e-3, 50e-12)]
    nodes = nodes[-n_ports:] if n_ports <= len(nodes) else nodes
    links = links[-(len(nodes) - 1):] if len(nodes) > 1 else []

    nn = len(nodes)
    Z = np.empty((len(f), nn, nn), dtype=complex)
    yv = series(s, 1.0e-3, 20e-9)                      # VRM at node 0
    for k in range(len(f)):
        Y = np.zeros((nn, nn), dtype=complex)
        for i, (C, esr, esl, cnt) in enumerate(nodes):
            Y[i, i] += cap_branch(s[k:k+1], C, esr, esl, cnt)[0]
        Y[0, 0] += yv[k]
        for i, (R, L) in enumerate(links):
            y = series(s[k:k+1], R, L)[0]
            Y[i, i] += y
            Y[i + 1, i + 1] += y
            Y[i, i + 1] -= y
            Y[i + 1, i] -= y
        Z[k] = np.linalg.inv(Y)
    return Z[:, ::-1, ::-1]        # order ports die-first


def pdn_z(f):
    s = 2j * np.pi * f
    y12 = series(s, 0.5e-3, 50e-12)      # die -> package
    y23 = series(s, 2.0e-3, 400e-12)     # package -> board
    ydie = cap_branch(s, 100e-9, 2e-3, 1e-12, n=4)      # on-die decap
    ypkg = cap_branch(s, 1e-6, 3e-3, 25e-12, n=8)       # package caps
    ybulk = cap_branch(s, 100e-6, 3e-3, 1.5e-9, n=4)    # board bulk
    yvrm = series(s, 1.0e-3, 20e-9)                     # VRM output stage

    Z = np.empty((len(f), 2, 2), dtype=complex)
    for k in range(len(f)):
        Y = np.array([
            [ydie[k] + y12[k],            -y12[k],                     0.0],
            [-y12[k],       ypkg[k] + y12[k] + y23[k],            -y23[k]],
            [0.0,                         -y23[k],  ybulk[k] + yvrm[k] + y23[k]],
        ], dtype=complex)
        Zn = np.linalg.inv(Y)
        Z[k] = Zn[np.ix_([0, 2], [0, 2])]
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="out/demo_pdn.s2p")
    ap.add_argument("--fmin", type=float, default=1e3)
    ap.add_argument("--fmax", type=float, default=5e9)
    ap.add_argument("--points", type=int, default=801)
    ap.add_argument("--z0", type=float, default=50.0,
                    help="reference impedance for the S-parameter export")
    ap.add_argument("--kind", choices=["s", "z"], default="s")
    ap.add_argument("--nports", type=int, default=2,
                    help="2 uses the hand-built 2-port; 3 or 4 uses the ladder")
    a = ap.parse_args()

    f = np.logspace(np.log10(a.fmin), np.log10(a.fmax), a.points)
    Z = pdn_z(f) if a.nports == 2 else pdn_z_nport(f, a.nports)
    znet = Network(f, Z, "Z", a.z0, "demo_pdn")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    net = znet if a.kind == "z" else z_to_s(znet, z0=a.z0)
    write_touchstone(a.out, net, fmt="RI", funit="HZ")

    z11 = Z[:, 0, 0]
    print(f"wrote {a.out}  ({a.points} pts, {a.fmin:g}..{a.fmax:g} Hz, "
          f"{net.kind}-params, R={a.z0:g})")
    print(f"  ports     : {Z.shape[1]}")
    print(f"  |Z11| min {np.abs(z11).min()*1e3:8.3f} mohm @ {f[np.argmin(np.abs(z11))]:.4g} Hz")
    print(f"  |Z11| max {np.abs(z11).max()*1e3:8.3f} mohm @ {f[np.argmax(np.abs(z11))]:.4g} Hz")
    print(f"  Z11 at DC-ish ({f[0]:.4g} Hz): {z11[0].real*1e3:.4f} mohm")


if __name__ == "__main__":
    main()
