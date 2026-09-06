"""Wrap the artifact HTML fragments into standalone pages under docs/.

The write-ups at the repo root (pipeline.html, ports.html) are artifact
*fragments*: they start at <title> and carry no <!doctype>, <html>, <head> or
<body>, because the publishing platform supplies that skeleton.  That keeps them
republishable, but it also means they declare no character encoding and no
mobile viewport when served straight from the repo.

This script leaves the fragments alone and writes complete documents to docs/,
along with an index page listing them.  Run it after editing any write-up:

    python tools/build_docs.py
"""
from __future__ import annotations

import html
import os
import re
import sys
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")

PAGES = [
    {
        "src": "pipeline.html",
        "icon": "⚡",
        "blurb": "The eight steps from Touchstone file to discrete state space, "
                 "and the four decisions that determine whether the fit is usable.",
    },
    {
        "src": "ports.html",
        "icon": "\U0001f50c",
        "blurb": "1-port versus 2-port model boundaries, the full testbench wiring "
                 "with a VRM in the loop, and why a linear VRM collapses back to a "
                 "1-port exactly.",
    },
]


def favicon(emoji):
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
           f"<text y='.9em' font-size='90'>{emoji}</text></svg>")
    return "data:image/svg+xml," + quote(svg)


def title_of(fragment, fallback):
    m = re.search(r"<title>(.*?)</title>", fragment, re.S | re.I)
    return m.group(1).strip() if m else fallback


def wrap(fragment, title, icon, description):
    body = re.sub(r"<title>.*?</title>\s*", "", fragment, count=1, flags=re.S | re.I)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="{favicon(icon)}">
</head>
<body>
{body.strip()}
</body>
</html>
"""


INDEX_CSS = """
:root{--bg:#F6F7F9;--surface:#FFFFFF;--line:#D8DEE5;--ink:#14181D;--ink-2:#3D4752;
  --muted:#68737F;--accent:#0E7C86;--accent-soft:#E2F1F2;--accent-line:#9FCBCE;
  --sans:"Inter","Helvetica Neue",Helvetica,Arial,system-ui,sans-serif;
  --serif:Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0E1216;--surface:#161C22;--line:#2C353E;--ink:#E6EBF0;--ink-2:#B4BFCA;
  --muted:#7C8896;--accent:#3FB6C4;--accent-soft:#122C30;--accent-line:#2A5F66}}
:root[data-theme="dark"]{--bg:#0E1216;--surface:#161C22;--line:#2C353E;--ink:#E6EBF0;
  --ink-2:#B4BFCA;--muted:#7C8896;--accent:#3FB6C4;--accent-soft:#122C30;--accent-line:#2A5F66}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);
  font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:0 28px 80px}
h1{font-family:var(--sans);font-size:clamp(30px,5vw,42px);letter-spacing:-.024em;
  font-weight:640;margin:0 0 12px;line-height:1.1}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--accent);font-weight:600}
header{padding:64px 0 26px;border-bottom:1px solid var(--line);margin-bottom:34px}
header p{font-size:19px;color:var(--muted);margin:0}
p{color:var(--ink-2)}
a.card{display:block;text-decoration:none;background:var(--surface);
  border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-bottom:16px;
  transition:border-color .15s,transform .15s}
a.card:hover,a.card:focus-visible{border-color:var(--accent-line);transform:translateY(-1px)}
a.card:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
a.card h2{font-family:var(--sans);font-size:19px;font-weight:620;letter-spacing:-.012em;
  margin:0 0 7px;color:var(--ink)}
a.card p{margin:0;font-size:15.5px;color:var(--muted);line-height:1.55}
a.card .go{font-family:var(--mono);font-size:11.5px;color:var(--accent);
  letter-spacing:.06em;display:block;margin-top:12px}
code{font-family:var(--mono);font-size:.87em;background:var(--accent-soft);
  border:1px solid var(--accent-line);border-radius:4px;padding:.08em .34em;color:var(--ink)}
pre{font-family:var(--mono);font-size:13px;line-height:1.7;background:var(--surface);
  border:1px solid var(--line);border-radius:8px;padding:16px 20px;overflow-x:auto;color:var(--ink)}
footer{border-top:1px solid var(--line);margin-top:38px;padding-top:20px;
  font-size:14px;color:var(--muted);font-family:var(--sans)}
"""


def build_index(entries):
    cards = "\n".join(
        f'''<a class="card" href="{e["href"]}">
  <h2>{html.escape(e["title"])}</h2>
  <p>{html.escape(e["blurb"])}</p>
  <span class="go">READ &rarr;</span>
</a>''' for e in entries)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sp2ss</title>
<meta name="description" content="Touchstone S-parameters to a discrete state-space PDN macromodel for RNM simulation.">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="{favicon("⚡")}">
<style>{INDEX_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">Power integrity &middot; RNM</div>
  <h1>sp2ss</h1>
  <p>Touchstone S-parameters to a discrete state-space PDN macromodel.</p>
</header>

<p>Fits a Touchstone file to a state-space model a real-number-modelling testbench
can evaluate in a few dozen multiply-accumulates per timestep. On the bundled
synthetic PDN that is a <strong>6-state</strong> 2-port model at
<code>3e-6</code> relative error, passive and stable.</p>

<pre>python sp2ss.py pdn.s2p --order 40 --reduce-target 1e-3 --ts 5e-12 --plot</pre>

{cards}

<footer>Generated by <code>tools/build_docs.py</code> &mdash; edit the write-ups at
the repo root, then re-run it.</footer>
</div>
</body>
</html>
"""


def main():
    os.makedirs(DOCS, exist_ok=True)
    entries = []
    for page in PAGES:
        src = os.path.join(ROOT, page["src"])
        if not os.path.exists(src):
            print(f"  skip {page['src']} (not found)")
            continue
        frag = open(src, encoding="utf-8").read()
        title = title_of(frag, page["src"])
        out = os.path.join(DOCS, page["src"])
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(wrap(frag, title, page["icon"], page["blurb"]))
        entries.append({"href": page["src"], "title": title, "blurb": page["blurb"]})
        print(f"  docs/{page['src']:16s} <- {page['src']}  ({title})")

    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(build_index(entries))
    print(f"  docs/index.html      <- {len(entries)} page(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
