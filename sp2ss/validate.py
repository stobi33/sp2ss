"""Validation: frequency-domain accuracy, passivity, and time-domain checks."""
from __future__ import annotations

import numpy as np

from .passivity import herm_eigs

__all__ = ["freq_report", "step_response", "compare_time", "plot_all"]


def freq_report(f, Zdata, ss_or_dss, label="model", floor=0.0):
    """Element-wise relative error of a model against the original data.

    `floor` clamps the denominator (see vectfit.fit_errors) so noise-dominated
    samples do not dominate the metric.
    """
    H = ss_or_dss.freqresp(f)
    den = np.maximum(np.abs(Zdata), 1e-300)
    if floor > 0:
        den = np.maximum(den, floor)
    err = np.abs(H - Zdata) / den
    n = Zdata.shape[1]
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "elem": f"Z{i+1}{j+1}",
                "max_rel": float(err[:, i, j].max()),
                "rms_rel": float(np.sqrt(np.mean(err[:, i, j] ** 2))),
                "f_at_max": float(f[np.argmax(err[:, i, j])]),
            })
    return {"label": label, "per_elem": rows,
            "max_rel": float(err.max()), "rms_rel": float(np.sqrt(np.mean(err ** 2))),
            "H": H, "err": err}


def step_response(dss, port=0, amp=1.0, n_steps=None, t_end=None):
    """Discrete-model step response to a current step injected at `port`."""
    if n_steps is None:
        n_steps = int(np.ceil((t_end or 200 * dss.Ts) / dss.Ts))
    u = np.zeros((n_steps, dss.n_ports))
    u[:, port] = amp
    y = dss.simulate(u)
    t = np.arange(n_steps) * dss.Ts
    return t, y


def compare_time(ss, dss, port=0, amp=1.0, t_end=None, oversample=64,
                 slew_steps=1):
    """Discrete model vs a fine-grid continuous reference, same stimulus.

    The stimulus is a current that ramps to `amp` over `slew_steps` sample
    periods and then holds.  That matters: an RNM current with a genuinely zero
    rise time puts an infinite dv/dt across the series port inductance, so
    comparing against it measures nothing but the impulse the coarse grid cannot
    resolve (it reports ~100% error for a perfectly good model).  A one-sample
    slew is both physically honest and the case the FIR inductance branch is
    exact for.
    """
    from .statespace import discretize

    if t_end is None:
        slow = np.min(np.abs(np.linalg.eigvals(ss.A).real))
        t_end = min(max(20.0 / max(slow, 1e-30), 400 * dss.Ts), 20000 * dss.Ts)
    n_d = max(int(np.ceil(t_end / dss.Ts)), 16)
    n0 = max(n_d // 20, 4)                      # step time, in samples

    u = np.zeros((n_d, ss.n_ports))
    u[:, port] = amp * np.clip((np.arange(n_d) - n0) / max(slew_steps, 1), 0, 1)
    y_d = dss.simulate(u)
    t_d = np.arange(n_d) * dss.Ts

    K = int(oversample)
    fine = discretize(ss, dss.Ts / K, "zoh")
    nf = n_d * K
    uf = np.zeros((nf, ss.n_ports))
    uf[:, port] = amp * np.clip((np.arange(nf) / K - n0) / max(slew_steps, 1), 0, 1)
    y_f = fine.simulate(uf)
    t_f = np.arange(nf) * (dss.Ts / K)

    y_ref = y_f[::K][:n_d]
    denom = max(np.abs(y_ref).max(), 1e-300)
    return {"t_d": t_d, "y_d": y_d, "t_c": t_f, "y_c": y_f,
            "max_abs_err": float(np.abs(y_d - y_ref).max()),
            "max_rel_err": float(np.abs(y_d - y_ref).max() / denom),
            "peak": float(np.abs(y_ref).max()),
            "settled": y_d[-1].copy(),
            "dc_continuous": ss.dc_gain(), "t_end": t_end}


def plot_all(path, f, Zdata, ss, dss, hsv=None, title="sp2ss"):
    """Four-panel diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = Zdata.shape[1]
    Hc = ss.freqresp(f)
    fd = f[f < 0.5 / dss.Ts]
    Hd = dss.freqresp(fd) if len(fd) else None

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle(title, fontsize=12)

    a = ax[0, 0]
    for i in range(n):
        for j in range(i, n):
            a.loglog(f, np.abs(Zdata[:, i, j]), lw=2.4, alpha=.32, color=f"C{i*n+j}",
                     label=f"Z{i+1}{j+1} data")
            a.loglog(f, np.abs(Hc[:, i, j]), lw=1.1, color=f"C{i*n+j}",
                     label=f"Z{i+1}{j+1} fit")
            if Hd is not None:
                a.loglog(fd, np.abs(Hd[:, i, j]), "--", lw=.9, color="k", alpha=.5)
    a.axvline(0.5 / dss.Ts, color="r", ls=":", lw=1)
    a.text(0.5 / dss.Ts, a.get_ylim()[0], " Nyquist", color="r", fontsize=8,
           rotation=90, va="bottom")
    a.set_xlabel("frequency [Hz]"); a.set_ylabel("|Z| [ohm]")
    a.set_title("impedance: data vs continuous fit (black dashed = discrete)")
    a.grid(True, which="both", alpha=.25); a.legend(fontsize=7, ncol=2)

    a = ax[0, 1]
    err = np.abs(Hc - Zdata) / np.maximum(np.abs(Zdata), 1e-300)
    for i in range(n):
        for j in range(i, n):
            a.loglog(f, err[:, i, j], lw=1, label=f"Z{i+1}{j+1}")
    a.set_xlabel("frequency [Hz]"); a.set_ylabel("relative error")
    a.set_title("fit error (relative)")
    a.grid(True, which="both", alpha=.25); a.legend(fontsize=7)

    a = ax[1, 0]
    fp = np.logspace(np.log10(f[0] / 10), np.log10(max(f[-1] * 10, 1e6)), 2000)
    ev = herm_eigs(ss, fp)
    for k in range(ev.shape[1]):
        a.semilogx(fp, ev[:, k], lw=1)
    a.axhline(0, color="k", lw=.8)
    a.set_xlabel("frequency [Hz]"); a.set_ylabel("eig Herm(Z) [ohm]")
    a.set_title(f"passivity: eigenvalues of (Z+Z^H)/2  (min = {ev.min():.3e})")
    a.grid(True, which="both", alpha=.25)
    a.set_yscale("symlog", linthresh=max(abs(ev).max() * 1e-6, 1e-18))

    a = ax[1, 1]
    cmp = compare_time(ss, dss)
    for k in range(cmp["y_c"].shape[1]):
        a.plot(cmp["t_c"] * 1e9, cmp["y_c"][:, k] * 1e3, lw=2.2, alpha=.35,
               color=f"C{k}", label=f"port {k} continuous")
        a.step(cmp["t_d"] * 1e9, cmp["y_d"][:, k] * 1e3, where="post", lw=1,
               color=f"C{k}", label=f"port {k} discrete")
    a.set_xlabel("time [ns]"); a.set_ylabel("port voltage [mV]")
    a.set_title(f"1 A current step (1-sample slew) into port 0  "
                f"(max rel err {cmp['max_rel_err']:.2e})")
    a.grid(True, alpha=.25); a.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return cmp
