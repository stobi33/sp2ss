"""sp2ss command line driver: Touchstone -> discrete state space -> RNM code."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .touchstone import read_touchstone
from .convert import (to_param, subset_ports, dc_extrapolate, upper_tri_index,
                      flatten, unflatten)
from .vectfit import vector_fit, model_response, fit_errors, make_weight
from .statespace import (from_pole_residue, absorb_e, balanced_truncate,
                         reduce_to_target, to_modal,
                         block_structure, discretize, suggest_timestep,
                         hankel_sv)
from .passivity import check_passivity, enforce_passivity
from . import emit, validate


def build_parser():
    p = argparse.ArgumentParser(
        prog="sp2ss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert a Touchstone S-parameter file into a discrete "
                    "state-space PDN macromodel for RNM simulation.",
        epilog="""
example
-------
  sp2ss out/demo_pdn.s2p --order 40 --ts 10e-12 --reduce-tol 1e-4 \\
        --name pdn_ss --outdir out --plot
""")
    p.add_argument("touchstone", help="input .sNp / .zNp file")

    g = p.add_argument_group("network")
    g.add_argument("--param", choices=["z", "y", "s"], default="z",
                   help="what to fit; z (default) is what RNM PDN drivers need")
    g.add_argument("--ports", type=int, nargs="+", default=None,
                   help="1-based port subset to keep")
    g.add_argument("--unused", choices=["open", "short"], default="open",
                   help="termination of the discarded ports (default open)")
    g.add_argument("--no-symmetrize", action="store_true",
                   help="do not force reciprocity before fitting")
    g.add_argument("--fmin", type=float, default=None, help="lower fit limit [Hz]")
    g.add_argument("--fmax", type=float, default=None, help="upper fit limit [Hz]")
    g.add_argument("--dc", type=float, nargs="+", default=None,
                   help="pin the DC resistance [ohm] by adding an f=0 sample: "
                        "one value for every port diagonal, or N values. "
                        "Off-diagonals are extrapolated from the data")
    g.add_argument("--add-dc", action="store_true",
                   help="add an extrapolated f=0 sample")

    g = p.add_argument_group("fit")
    g.add_argument("--order", type=int, default=40, help="pole count (default 40)")
    g.add_argument("--order-sweep", type=int, nargs="+", default=None,
                   help="try these orders and keep the smallest meeting --target")
    g.add_argument("--target", type=float, default=1e-3,
                   help="max relative error target for --order-sweep")
    g.add_argument("--iters", type=int, default=12, help="VF iterations")
    g.add_argument("--asymp", choices=["none", "d", "de"], default="de",
                   help="asymptotic terms: d = series resistance, de = also the "
                        "series port inductance (default de -- a PDN is "
                        "inductive at the top of the band and omitting E makes "
                        "VF run off to huge cancelling poles)")
    g.add_argument("--weight", choices=["inv", "sqrt", "none"], default="inv",
                   help="LS weighting; inv = relative error (default)")
    g.add_argument("--weight-floor", type=float, default=0.0,
                   help="clamp the relative-weight denominator at this "
                        "magnitude (ohms for a Z fit). Below roughly 1 mohm a "
                        "50-ohm-referenced S-parameter is mostly VNA noise; "
                        "set this so the fit does not chase it")
    g.add_argument("--no-relax", action="store_true", help="classic (non-relaxed) VF")

    g = p.add_argument_group("reduce / passivity")
    g.add_argument("--reduce-order", type=int, default=None,
                   help="balanced-truncation target state count")
    g.add_argument("--reduce-tol", type=float, default=None,
                   help="drop Hankel singular values below tol*max")
    g.add_argument("--reduce-target", type=float, default=None,
                   help="search for the smallest state count whose relative "
                        "error stays under this (e.g. 1e-3); overrides the above")
    g.add_argument("--no-passivity", action="store_true",
                   help="skip the passivity check and enforcement")
    g.add_argument("--passivity-iters", type=int, default=20)

    g = p.add_argument_group("discretize / emit")
    g.add_argument("--ts", type=float, default=None,
                   help="RNM timestep [s] (default: 20 pts/cycle at the fastest pole)")
    g.add_argument("--ppc", type=float, default=20.0,
                   help="points per cycle used to pick --ts automatically")
    g.add_argument("--method", choices=["zoh", "bilinear", "foh"], default="zoh")
    g.add_argument("--e-mode", choices=["fir", "pole"], default="fir",
                   help="how to realize the s*E series inductance: fir keeps it "
                        "as a feedthrough (E/Ts)(u[n]-u[n-1]) (default, stable, "
                        "exact for a piecewise-linear current); pole folds it "
                        "into a band-limited state (strictly proper, but "
                        "inflates D by E*wc)")
    g.add_argument("--e-corner", type=float, default=10.0,
                   help="--e-mode pole corner, as a multiple of fmax (default 10)")
    g.add_argument("--form", choices=["modal", "dense"], default="modal",
                   help="modal gives a block-diagonal Ad and much cheaper HDL")
    g.add_argument("--name", default="pdn_ss", help="generated module name")
    g.add_argument("--outdir", default="out")
    g.add_argument("--emit", nargs="+",
                   default=["sv", "vams", "c", "py", "json", "tb"],
                   choices=["sv", "vams", "c", "py", "json", "tb"])
    g.add_argument("--plot", action="store_true", help="write a diagnostic PNG")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def run(argv=None):
    args = build_parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else print

    # ---------------------------------------------------------------- load
    say(f"[1/8] reading {args.touchstone}")
    net = read_touchstone(args.touchstone)
    say(f"      {net}")
    rec = net.reciprocity_error()
    say(f"      reciprocity error {rec:.3e}"
        + ("  (large -- check the extraction)" if rec > 1e-3 else ""))

    if args.ports:
        idx = [p - 1 for p in args.ports]
        say(f"[2/8] keeping ports {args.ports} (others {args.unused})")
        net = subset_ports(net, idx, unused=args.unused)
    else:
        say(f"[2/8] converting {net.kind} -> {args.param.upper()}")
    net = to_param(net, args.param)
    if not args.no_symmetrize:
        net = net.symmetrized()
    if hasattr(net, "cond") and np.max(net.cond) > 1e8:
        say(f"      WARNING: conversion ill-conditioned "
            f"(max cond {np.max(net.cond):.2e}); the source data may be "
            f"near-singular at some frequencies")

    m = np.ones(net.n_freq, dtype=bool)
    if args.fmin:
        m &= net.f >= args.fmin
    if args.fmax:
        m &= net.f <= args.fmax
    if not m.all():
        from .touchstone import Network
        net = Network(net.f[m], net.data[m], net.kind, net.z0, net.name)
        say(f"      band-limited to {net.f[0]:.4g}..{net.f[-1]:.4g} Hz "
            f"({net.n_freq} pts)")
    if args.dc is not None or args.add_dc:
        net = dc_extrapolate(net, args.dc if args.dc is None
                             else (args.dc[0] if len(args.dc) == 1 else args.dc))
        say(f"      added f=0 sample, Z(0) diag = "
            f"{np.diag(net.data[0]).real}")

    n = net.n_ports
    elems = upper_tri_index(n, symmetric=not args.no_symmetrize)
    F = flatten(net, elems)
    s = net.s
    fpos = net.f[net.f > 0]

    # ----------------------------------------------------------------- fit
    say(f"[3/8] vector fitting ({len(elems)} unique elements, common poles)")
    orders = args.order_sweep or [args.order]
    best = None
    for order in orders:
        vf = vector_fit(F, s, order=order, n_iter=args.iters, asymp=args.asymp,
                        weight=args.weight, relax=not args.no_relax,
                        weight_floor=args.weight_floor,
                        verbose=(not args.quiet and len(orders) == 1))
        e = fit_errors(F, model_response(s, vf.poles, vf.residues, vf.d, vf.e),
                       floor=args.weight_floor)
        say(f"      order {order:3d}: max rel {e['max_rel']:.3e}, "
            f"rms rel {e['rms_rel']:.3e}"
            + (f"  (vs floor {args.weight_floor:g}: max "
               f"{e['max_rel_floored']:.3e})" if args.weight_floor else ""))
        key = "max_rel_floored" if args.weight_floor else "max_rel"
        if best is None or e[key] < best[1][key]:
            best = (vf, e, order)
        if len(orders) > 1 and e[key] <= args.target:
            best = (vf, e, order)
            break
    vf, ferr, order = best
    say(f"      -> using order {order}: max rel {ferr['max_rel']:.3e}")
    if args.weight_floor:
        say(f"      smallest |Z| in the data: {ferr['data_min_abs']:.3e} ohm "
            f"(weight floor {args.weight_floor:g} ohm)")
    _key = "max_rel_floored" if args.weight_floor else "max_rel"
    if ferr[_key] > max(args.target, 1e-2):
        say(f"      WARNING: fit error is large. Raise --order, narrow the "
            f"band, or -- if the data has a noise floor -- set --weight-floor "
            f"to the smallest |Z| you trust.")

    # ------------------------------------------------------- realization
    say("[4/8] realizing state space")
    R = unflatten(vf.residues, elems, n, symmetric=not args.no_symmetrize)
    R = np.transpose(R, (2, 0, 1))                     # (M, n, n)
    D = unflatten(vf.d[:, None], elems, n, not args.no_symmetrize)[:, :, 0]
    E = unflatten(vf.e[:, None], elems, n, not args.no_symmetrize)[:, :, 0]
    Euse = E if args.asymp == "de" else None
    if Euse is not None and args.e_mode == "pole":
        wc = 2 * np.pi * args.e_corner * fpos.max()
        vf_poles, R, D, Euse = absorb_e(vf.poles, R, D, Euse, wc)
        say(f"      folded s*E into a band-limited pole at "
            f"{wc/2/np.pi:.4g} Hz (--e-mode pole)")
        _fc = wc / 2 / np.pi
        if args.ts and _fc > 0.1 / args.ts:
            say(f"      WARNING: that corner is above 0.1/Ts "
                f"({0.1/args.ts:.4g} Hz). A band-limiting pole near or beyond "
                f"Nyquist is not resolved by the sample rate and will degrade "
                f"the time-domain response. Lower --e-corner, lower --ts, or "
                f"use the default --e-mode fir.")
        ss = from_pole_residue(vf_poles, R, D, None)
    else:
        ss = from_pole_residue(vf.poles, R, D, Euse)
    say(f"      {ss}  stable={ss.is_stable()}  "
        f"max pole freq {ss.max_pole_freq():.4g} Hz")
    if ss.E is not None:
        Ld = np.diag(ss.E) * 1e12
        say(f"      series port inductance from E: "
            f"{np.array2string(Ld, precision=4)} pH  "
            f"(realized as a 1-tap FIR, see --e-mode)")
    _u = "ohm" if args.param == "z" else ("S" if args.param == "y" else "")
    _lbl = "DC resistance" if args.param == "z" else "DC gain"
    say(f"      {_lbl} (diag): {np.diag(ss.dc_gain())} {_u}")
    dyn = np.abs(vf.residues).max() / max(np.abs(np.diag(ss.dc_gain())).max(), 1e-30)
    if dyn > 1e15:
        say(f"      WARNING: internal dynamic range {dyn:.1e} -- the fit has "
            f"large cancelling residue/d pairs. Use --asymp de.")

    # ---------------------------------------------------------- reduction
    if args.reduce_target:
        say(f"[5/8] balanced truncation, searching for max rel <= "
            f"{args.reduce_target:.3g}"
            + (f" (|Z| floored at {args.weight_floor:g})" if args.weight_floor else ""))
        full = ss
        Zref = net.data[net.f > 0]

        def _err(cand):
            return validate.freq_report(fpos, Zref, cand,
                                        floor=args.weight_floor)["max_rel"]

        red, basis, cache = reduce_to_target(full, _err, args.reduce_target,
                                             verbose=not args.quiet)
        if red is not None:
            say(f"      {full.n_states} -> {red.n_states} states, "
                f"max rel {_err(red):.3e}")
            ss = red
        else:
            e_full = _err(full)
            say(f"      no reduction meets the target; keeping "
                f"{ss.n_states} states")
            if e_full > args.reduce_target:
                say(f"      (the unreduced model is already at "
                    f"{e_full:.3e} -- the target is below the fit error "
                    f"itself, so no truncation could ever meet it)")
    elif args.reduce_order or args.reduce_tol:
        say("[5/8] balanced truncation")
        ss = balanced_truncate(ss, order=args.reduce_order, tol=args.reduce_tol,
                               verbose=not args.quiet)
        e2 = validate.freq_report(fpos, net.data[net.f > 0], ss,
                                  floor=args.weight_floor)
        say(f"      post-reduction max rel {e2['max_rel']:.3e}")
    else:
        say("[5/8] balanced truncation skipped (--reduce-tol / --reduce-order)")
        ss.meta["hsv"] = hankel_sv(ss)[0]

    # ---------------------------------------------------------- passivity
    passive = "not checked"
    if args.no_passivity:
        say("[6/8] passivity check skipped")
    else:
        say("[6/8] passivity")
        ok, viol, dmin = check_passivity(ss, verbose=not args.quiet)
        if not ok and args.param in ("z", "y"):
            ss, rep = enforce_passivity(ss, fpos, n_iter=args.passivity_iters,
                                        verbose=not args.quiet)
            ok = rep["converged"]
            say(f"      enforcement {'succeeded' if ok else 'INCOMPLETE'} "
                f"(worst eig {rep['initial_worst']:.3e} -> {rep['final_worst']:.3e})")
            e3 = validate.freq_report(fpos, net.data[net.f > 0], ss,
                                      floor=args.weight_floor)
            say(f"      post-enforcement max rel {e3['max_rel']:.3e}")
        elif not ok:
            say("      (enforcement only implemented for Z/Y models)")
        passive = "yes" if ok else "no"

    if args.form == "modal":
        try:
            ss = to_modal(ss)
            sizes, clean = block_structure(ss.A)
            say(f"      modal form: {sizes.count(1)} 1x1 + {sizes.count(2)} 2x2 "
                f"blocks (clean={clean})")
        except np.linalg.LinAlgError as exc:
            say(f"      modal transform failed ({exc}); keeping dense form")

    # ------------------------------------------------------- discretize
    ts = args.ts or suggest_timestep(ss, ppc=args.ppc)
    say(f"[7/8] discretizing at Ts = {ts:.6g} s ({1/ts:.6g} Hz), {args.method}")
    if args.ts is None:
        say(f"      (auto: {args.ppc:g} points/cycle at the fastest pole "
            f"{ss.max_pole_freq():.4g} Hz)")
    nyq = 0.5 / ts
    if nyq < fpos.max():
        say(f"      NOTE: Nyquist {nyq:.4g} Hz < data fmax {fpos.max():.4g} Hz -- "
            f"content above Nyquist is not represented. Use --fmax {nyq:.4g} "
            f"when fitting if that band does not matter.")
    dss = discretize(ss, ts, args.method)
    say(f"      {dss}")
    if not dss.is_stable():
        say("      ERROR: discrete model is unstable -- reduce Ts")
        return 2

    # ------------------------------------------------------------- emit
    os.makedirs(args.outdir, exist_ok=True)
    meta = {
        "source": os.path.basename(args.touchstone),
        "param": args.param.upper(),
        "n_ports": n,
        "fit_order": order,
        "fmin": float(fpos.min()), "fmax": float(fpos.max()),
        "max_rel": ferr["max_rel"], "rms_rel": ferr["rms_rel"],
        "passive": passive,
    }
    say(f"[8/8] writing to {args.outdir}/")
    written = []
    def w(fn, txt):
        path = os.path.join(args.outdir, fn)
        with open(path, "w") as fh:
            fh.write(txt if txt.endswith("\n") else txt + "\n")
        written.append(path)

    if "sv" in args.emit:
        w(f"{args.name}.sv", emit.emit_systemverilog(dss, args.name, meta))
    if "vams" in args.emit:
        w(f"{args.name}.vams", emit.emit_verilogams(dss, args.name, meta))
    if "c" in args.emit:
        w(f"{args.name}.c", emit.emit_c(dss, args.name, meta))
    if "py" in args.emit:
        w(f"{args.name}_model.py", emit.emit_python(dss, args.name, meta))
    if "json" in args.emit:
        w(f"{args.name}.json", emit.emit_json(dss, meta))
    if "tb" in args.emit:
        w(f"tb_{args.name}.sv", emit.emit_sv_testbench(dss, args.name, meta))
    np.savez(os.path.join(args.outdir, f"{args.name}.npz"),
             Ad=dss.Ad, Bd=dss.Bd, Cd=dss.Cd, Dd=dss.Dd, Ts=dss.Ts,
             A=ss.A, B=ss.B, C=ss.C, D=ss.D,
             poles=vf.poles, residues=vf.residues, f=net.f, Z=net.data)
    written.append(os.path.join(args.outdir, f"{args.name}.npz"))

    if args.plot:
        png = os.path.join(args.outdir, f"{args.name}_report.png")
        validate.plot_all(png, fpos, net.data[net.f > 0], ss, dss,
                                title=f"{args.name}: {meta['source']} "
                                      f"order {order} -> {dss.n_states} states")
        written.append(png)

    for path in written:
        say(f"      {path}")

    cmp = validate.compare_time(ss, dss)
    say("")
    say("summary")
    say(f"  states        : {dss.n_states}   ports: {dss.n_ports}")
    say(f"  fit error     : max {ferr['max_rel']*100:.4f} %  "
        f"rms {ferr['rms_rel']*100:.4f} %")
    if args.weight_floor:
        say(f"  above floor   : max {ferr['max_rel_floored']*100:.4f} %  "
            f"rms {ferr['rms_rel_floored']*100:.4f} %  "
            f"(|Z| clamped at {args.weight_floor:g} ohm)")
    say(f"  passive       : {passive}")
    rho = dss.spectral_radius()
    say(f"  Ts            : {dss.Ts:.6g} s   spectral radius {rho:.12f} "
        f"(1 - rho = {1.0 - rho:.3e})")
    say(f"  {_lbl:14s}: {np.diag(ss.dc_gain())} {_u}")
    say(f"  mults/timestep: ~{_mults(dss)}")
    say(f"  step response : peak {cmp['peak']*1e3:.4f} mV for 1 A "
        f"(1-sample slew), discrete vs continuous max rel "
        f"{cmp['max_rel_err']:.3e}")
    return 0


def _mults(dss):
    nz = lambda M: int(np.count_nonzero(M))
    return nz(dss.Ad) + nz(dss.Bd) + nz(dss.Cd) + nz(dss.Dd)


def main():
    sys.exit(run())


if __name__ == "__main__":
    main()
