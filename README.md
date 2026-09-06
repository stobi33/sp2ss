# sp2ss — S-parameters to a state-space PDN model for RNM

Converts a Touchstone file into a discrete-time state-space macromodel you can
drop into a real-number-modelling (RNM) simulation, plus the SystemVerilog /
Verilog-AMS / C / Python to run it.

```
x[n+1] = Ad x[n] + Bd i[n]
v[n]   = Cd x[n] + Dd i[n] + (E/Ts)(i[n] - i[n-1])
```

`i` = current injected into each port [A], `v` = port voltage [V].

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib

.venv/bin/python tools/make_demo_pdn.py -o out/demo_pdn.s2p      # synthetic PDN
.venv/bin/python sp2ss.py out/demo_pdn.s2p \
      --order 40 --reduce-target 1e-3 --ts 5e-12 --plot

.venv/bin/python tools/selftest.py                                # verify
```

On the bundled 2-port demo that gives a **6-state** model with 3e-6 relative
error, passive, stable, ~40 multiplies per timestep. The 4-port demo reduces to
8 states.

## The pipeline

| # | step | why |
|---|------|-----|
| 1 | read Touchstone | 2-port column order and trailing noise data are handled |
| 2 | `S -> Z` | RNM injects current and reads voltage, so fit the driving-point impedance |
| 3 | vector fitting | rational `Z(s) ~ D + sE + sum R_m/(s-p_m)`, common poles, stable |
| 4 | realize | real modal state space, 2x2 blocks per conjugate pair |
| 5 | balanced truncation | drop states with negligible Hankel singular values |
| 6 | passivity | positive-real Hamiltonian check, enforce by perturbing `C` |
| 7 | discretize | ZOH at the RNM timestep |
| 8 | emit | SystemVerilog, Verilog-AMS wreal, C, Python, JSON, testbench |

## Five things that decide whether this works

**1. Fit Z, not S.** An RNM load model is a current source; you need
`v = Z i`. Converting S per timestep is both wrong and expensive.
`Z = sqrt(Z0) (I-S)^-1 (I+S) sqrt(Z0)`.

**2. Weight the fit relatively (`--weight inv`, the default).** PDN impedance
spans four or five decades. Plain least squares fits the anti-resonant peaks and
throws away the milliohm mid-band you actually care about — on the demo network
that is a 40x worse worst-case error (1.4% vs 0.034%).

**3. Keep the `sE` term (`--asymp de`, the default).** A PDN is *inductive* at
the top of the band. Without an `s`-proportional term, vector fitting has only
one way to synthesize a rising `|Z|`: run a pole off to infinity with a residue
that cancels against `d`. On the demo file that produces a pole at 1.3e14 Hz
with residue 1.4e20 cancelling `d = 1.6e5` — a fit that *looks* fine (3e-4
error) but has ~1e8 internal dynamic range, which destroys balanced truncation
and wastes precision in the emitted HDL. With `--asymp de` the same fit is 100x
more accurate and the numbers become physical: `d` is the DC resistance, `E` is
the port inductance matrix.

**4. Realize `sE` as a one-tap FIR, not a pole and not bilinear.** A symmetric
real `E` is a lossless inductance, so it contributes nothing to the Hermitian
part and needs no state at all:

- *bilinear* puts a pole at `z = -1` — undamped Nyquist ringing forever.
- *band-limiting pole* works, but pushing the corner far enough out for <1%
  error re-inflates `D` by `E*wc` and brings the cancellation straight back.
  Offered as `--e-mode pole` if you need a strictly proper model.
- *FIR* `(E/Ts)(i[n] - i[n-1])` is unconditionally stable, is exact whenever the
  current is piecewise linear over a sample, and tends to `jwE` at low frequency.

**5. Check passivity, and mean it.** A non-passive macromodel injects energy.
Inside an RNM loop closed around a regulator that shows up as a slowly growing
oscillation which looks exactly like a real instability. The tool runs the
positive-real Hamiltonian test and, if needed, enforces passivity by perturbing
`C` only — poles, and therefore stability, are untouched.

## Choosing the timestep

ZOH is *exact* for the state part under a staircase input, so the binding
constraint is bandwidth, not stability. Rule of thumb: `Ts <= 1/(20 f_max)` for
the highest frequency whose droop you care about — not the top of the Touchstone
file. If the model carries a series inductance, the FIR branch adds a half-sample
lag worth roughly `pi f Ts`, so 20 points per cycle costs ~16% on the inductive
part alone at `f = fs/20`, and 100 points per cycle costs ~3%.

Do not validate against a zero-rise-time current step. That puts infinite `dv/dt`
across the port inductance and reports ~100% error for a perfectly good model.
`sp2ss` validates with a one-sample current slew, which is both physically honest
and the case the FIR branch is exact for; on the demo that gives 1.2e-4 relative
error against a 64x oversampled reference.

## Noise floor

Relative weighting taken literally will also chase the VNA noise floor. Below
roughly 1 mohm a 50-ohm-referenced S-parameter is mostly noise: on the demo, the
smallest `|Z|` is 14 uohm, so 37 uohm of measurement noise makes that point
meaningless and the raw relative error reads 600% for a model that is fine.

Use `--weight-floor 1e-3` to clamp the weight denominator at the smallest
impedance you trust; the summary then also reports the error against that floor.

If you can, **export Z-parameters directly** (`.z2p`) or renormalize to a low
reference impedance. Round-tripping milliohms through a 50-ohm S-parameter with 9
significant digits costs about 25 nohm of quantization before any real noise.

## Options worth knowing

```
--order 40                pole count for the fit
--order-sweep 20 30 40    try several, keep the smallest meeting --target
--reduce-target 1e-3      search for the smallest state count meeting this
--weight-floor 1e-3       do not fit below this impedance (ohms)
--fmax 2e9                band-limit the fit
--ports 1 3 --unused open reduce a multi-port extraction to the ports you drive
--dc 3.5e-3               pin the DC resistance
--ts 5e-12                RNM timestep (default: 20 pts/cycle at the fastest pole)
--form modal              block-diagonal Ad, ~2 multiplies per state (default)
--e-mode fir|pole         how to realize the series port inductance
--plot                    four-panel diagnostic PNG
```

The order search is a geometric ladder with downward refinement, *not* a
bisection: reduced-order error is not monotone in retained order, because cutting
between two near-equal Hankel singular values makes the balancing transform
ill-conditioned. On the 4-port demo, order 150 gives 3.1e-3 while order 20 gives
1.2e-4 — bisection would have missed the 8-state answer entirely.

## Using the output

```systemverilog
pdn_ss_rail #(.VNOM(0.8), .TS(5e-12)) u_pdn (
    .clk(clk_5ps), .rst_n(rst_n),
    .i_load0(i_die), .i_load1(i_vrm),      // current DRAWN, positive
    .v_rail0(v_die), .v_rail1(v_vrm)       // v = VNOM - Z*i
);
```

Clock it at exactly `TS`. Changing the sample rate invalidates the model — re-run
`sp2ss` with the new `--ts`.

## Method write-ups

Two pages explain the reasoning behind the pipeline:

- **The pipeline** (`pipeline.html`) — the eight steps, and the four decisions that determine whether the fit is usable.
- **Where to cut the PDN** (`ports.html`) — 1-port vs 2-port model boundaries, testbench wiring with a VRM in the loop, and why a linear VRM collapses back to a 1-port exactly.

Open `docs/index.html` to read them. The files at the repo root are publishing
fragments (no `<!doctype>` or `<head>`, which the artifact host supplies); the
copies under `docs/` are complete standalone documents with the encoding,
viewport and favicon filled in. After editing a write-up at the root, re-run:

```bash
python tools/build_docs.py
```

## Layout

```
sp2ss/touchstone.py   Touchstone v1 read/write, minimal v2
sp2ss/convert.py      S/Y/Z conversion, port subsetting, DC extrapolation
sp2ss/vectfit.py      relaxed vector fitting, real arithmetic, pruning
sp2ss/statespace.py   realization, balancing, modal form, discretization
sp2ss/passivity.py    positive-real Hamiltonian test, QP-based enforcement
sp2ss/emit.py         SystemVerilog / Verilog-AMS / C / Python / JSON emitters
sp2ss/validate.py     error metrics, time-domain comparison, plots
tools/build_docs.py   wraps the write-ups into standalone docs/ pages
tools/selftest.py     end-to-end verification
```

`tools/selftest.py` verifies the whole chain: vector fitting against a known
rational system, Touchstone round trips, the generated SystemVerilog parsed back
into matrices and compared bit-exact, the generated C compiled and run against
the reference, and the recovered DC resistance and port inductances checked
against the synthetic network that produced them.

## References

- Gustavsen & Semlyen, *Rational approximation of frequency domain responses by
  vector fitting*, IEEE Trans. Power Delivery 14(3), 1999.
- Gustavsen, *Improving the pole relocating properties of vector fitting*, IEEE
  Trans. Power Delivery 21(3), 2006.
- Grivet-Talocia, *Passivity enforcement via perturbation of Hamiltonian
  matrices*, IEEE Trans. CAS-I 51(9), 2004.
