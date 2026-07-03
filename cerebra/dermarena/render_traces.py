"""Render DermArena agent predictions (JSONL) into a self-contained HTML trace viewer.

Matches the simplified SELECT -> EXECUTE -> VERIFY(confirm|rerun) pipeline:
one card per case, images inline on the left, the agentic chain on the right —
Step 1 tool selection (what Qwen chose + why), Step 2 tool findings, Step 3
confirm/rerun decision, final prediction vs ground truth. Prompt dropdowns show
the exact prompt each step sends. Images embedded as base64 (portable single file).

    PY=/fs04/.../dermagent/bin/python
    PYTHONPATH=. $PY -m cerebra.dermarena.render_traces preds.jsonl --out traces.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from typing import Any, Dict, List

from PIL import Image

from cerebra.dermarena import select as _select, verify as _verify, candidates as _candidates


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


def _img_uri(path: str, max_side: int = 420) -> str:
    try:
        im = Image.open(path).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        b = io.BytesIO(); im.save(b, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""


def _prompt_details(label: str, pairs: List[tuple]) -> str:
    inner = "".join(f'<div class=pl>{_esc(sl)}</div><pre>{_esc(t)}</pre>' for sl, t in pairs)
    return f'<details class=prompt><summary>&#9656; prompt · {_esc(label)}</summary>{inner}</details>'


def _images_html(rec: Dict[str, Any]) -> str:
    figs = []
    for path, mod in (rec.get("case_images") or {}).items():
        uri = _img_uri(path)
        cap = f"{_esc(mod)}<br><span class=fn>{_esc(os.path.basename(path))}</span>"
        figs.append(f'<figure><img src="{uri}"><figcaption>{cap}</figcaption></figure>'
                    if uri else f'<figure class=missing><div>image?</div><figcaption>{cap}</figcaption></figure>')
    return "".join(figs) or "<div class=noimg>no images (text-only case)</div>"


def _render_case(rec: Dict[str, Any]) -> str:
    tr = rec.get("trace", {})

    # Step 1 — selection
    calls = ""
    for c in (tr.get("tool_calls") or []):
        cand = (" · candidates: " + ", ".join(c.get("candidate_diseases") or [])) if c.get("candidate_diseases") else ""
        calls += (f'<div class=finding><div class=tool>{_esc(c.get("tool"))} '
                  f'<span class=fdx>({len(c.get("image_paths") or [])} img)</span></div>'
                  f'<div class=body>{_esc(cand.strip(" ·"))}</div></div>')
    if not calls:
        calls = "<div class=empty>(no tools selected — verify from narrative)</div>"

    # Step 2 — findings
    finds = ""
    for f in (tr.get("findings") or []):
        body = f.get("summary") or f.get("error") or ""
        fd = f' <span class=fdx>→ {_esc(f.get("final_diagnosis"))}</span>' if f.get("final_diagnosis") else ""
        finds += (f'<div class=finding><div class=tool>{_esc(f.get("tool"))}{fd}</div>'
                  f'<div class=body>{_esc(str(body))}</div></div>')
    if not finds:
        finds = "<div class=empty>(no tool findings — tools skipped or none selected)</div>"

    # Step 3 — verify / rerun
    action = tr.get("verify_action")
    rerun = ""
    for rl in (tr.get("rerun_log") or []):
        rerun += f'<div class=finding><div class=tool>rerun → {_esc(rl["call"].get("tool"))}</div><div class=body>{_esc(rl.get("why"))}</div></div>'
    verify_html = f'<div><b>decision:</b> <span class=role>{_esc(action)}</span> · reruns: {rec.get("rerun_count", 0)}</div>{rerun}'

    gt = _esc(rec.get("ground_truth"))
    gt_test = rec.get("diagnostic_test_gt")
    gt_line = f'<div class=gt><b>GT dx:</b> {gt}' + (f' &nbsp;<b>GT test:</b> {_esc(gt_test)}' if gt_test else "") + "</div>"

    raw = ""
    if tr.get("select_raw"):
        raw += f'<details><summary>SELECT raw</summary><pre>{_esc(tr["select_raw"])}</pre></details>'
    if tr.get("verify_raw"):
        raw += f'<details><summary>VERIFY raw</summary><pre>{_esc(tr["verify_raw"])}</pre></details>'

    # Step 1b — candidate DDx (only present when a closed-set classifier was selected)
    cands = tr.get("candidates") or []
    cand_html = ""
    if cands:
        cand_prompt = _prompt_details("Step 1b candidate DDx", [
            ("SYSTEM", _candidates.SYSTEM), ("USER (template)", _candidates.USER_TEMPLATE)])
        cand_html = (f'<h4>Step 1b · CANDIDATES — Qwen top-{len(cands)} DDx → PanDerm ranks</h4>{cand_prompt}'
                     f'<ol class=cands>' + "".join(f"<li>{_esc(c)}</li>" for c in cands) + "</ol>")

    sel_prompt = _prompt_details("Step 1 tool-selection", [
        ("SYSTEM", _select.SYSTEM), ("USER (template)", _select.USER_TEMPLATE), ("TOOL MENU", _select.TOOL_MENU)])
    ver_prompt = _prompt_details("Step 3 confirm-or-rerun", [
        ("SYSTEM", _verify.SYSTEM), ("USER (template)", _verify.USER_TEMPLATE)])

    return f"""
<section class=case>
  <div class=head><span class=badge>{_esc(rec.get("task")).upper()}</span>
    <span class=cid>{_esc(rec.get("_id"))}</span>
    <span class=cost>${rec.get("cumulative_usd", 0):.4f}</span></div>
  {gt_line}
  <div class=cols>
    <div class=left><h4>images (modality hint)</h4>{_images_html(rec)}</div>
    <div class=right>
      <h4>Step 1 · SELECT — Qwen picks the tools</h4>
      {sel_prompt}
      <div class=mut>{_esc(tr.get("select_reasoning"))}</div>
      {calls}
      {cand_html}
      <h4>Step 2 · EXECUTE — tool findings</h4>{finds}
      <h4>Step 3 · VERIFY — confirm / rerun</h4>
      {ver_prompt}
      {verify_html}
      {raw}
      <h4>Final prediction</h4><pre class=pred>{_esc(rec.get("prediction"))}</pre>
    </div>
  </div>
</section>"""


CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1419;color:#e6edf3}
header{padding:14px 20px;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:9}
header h1{margin:0;font-size:16px}header .sub{color:#8b949e;font-size:12px}
.case{margin:16px 20px;background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}
.head{display:flex;align-items:center;gap:10px;padding:10px 14px;background:#1c2128;border-bottom:1px solid #30363d}
.badge{background:#1f6feb;color:#fff;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600}
.cid{color:#8b949e;font-family:monospace}.cost{margin-left:auto;color:#3fb950;font-family:monospace}
.gt{padding:8px 14px;background:#13241a;border-bottom:1px solid #30363d;color:#7ee787}
.cols{display:grid;grid-template-columns:300px 1fr;gap:0}
.left{padding:12px 14px;border-right:1px solid #30363d;background:#0f1419}
.right{padding:12px 16px;min-width:0}
h4{margin:14px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#58a6ff}
.right h4:first-child{margin-top:0}
figure{margin:0 0 12px}figure img{max-width:100%;border-radius:6px;border:1px solid #30363d}
figcaption{font-size:12px;color:#adbac7;margin-top:3px}.fn{color:#6e7681;font-size:11px}
.missing div{padding:30px;text-align:center;color:#6e7681;border:1px dashed #30363d;border-radius:6px}
.noimg{color:#6e7681;font-style:italic}
.mut{font-size:12.5px;color:#adbac7;margin:2px 0 8px;font-style:italic}
.finding{margin:0 0 10px;border-left:3px solid #30363d;padding-left:10px}
.tool{font-weight:600;color:#d2a8ff;font-size:12px}.fdx{color:#7ee787;font-weight:400}
.finding .body{color:#adbac7;font-size:13px;white-space:pre-wrap}
.role{display:inline-block;font-weight:600;color:#f0883e}
ol.cands{margin:2px 0 8px;padding-left:20px}ol.cands li{margin:1px 0;font-size:13px}
.empty{color:#6e7681;font-style:italic}
pre{white-space:pre-wrap;word-break:break-word;background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;font-size:12px}
pre.pred{border-color:#238636;background:#0d1a12;color:#7ee787}
details{margin:6px 0}summary{cursor:pointer;color:#8b949e;font-size:12px}
details.prompt{margin:4px 0 10px;background:#0d1117;border:1px dashed #30363d;border-radius:6px;padding:4px 8px}
details.prompt summary{color:#d2a8ff}
.pl{color:#6e7681;font-size:10px;text-transform:uppercase;letter-spacing:.05em;margin:6px 0 2px}
.setup{margin:16px 20px;background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px}
.setup h2{margin:0 0 8px;font-size:15px;color:#58a6ff}
.setup h3{margin:14px 0 6px;font-size:12px;color:#d2a8ff;text-transform:uppercase;letter-spacing:.04em}
.setup p{color:#c9d1d9;font-size:13px}
.setup table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
.setup td,.setup th{border:1px solid #30363d;padding:6px 9px;text-align:left}
.setup th{background:#1c2128;color:#adbac7}
.setup .flow{font-family:monospace;font-size:12px;white-space:pre;background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;color:#adbac7;overflow:auto}
.setup code{background:#0d1117;padding:1px 5px;border-radius:4px;color:#7ee787}
.setup .star{color:#f0883e;font-weight:600}
"""

SETUP_HTML = """
<div class=setup>
<h2>🧠 How this pipeline works — read before the cases</h2>
<p>A simplified agentic loop for the DermArena dx-track (on the Cerebra repo). Qwen decides
which specialist vision tools to run, then confirms the diagnosis or reruns a tool.
Two LLM calls (<span class=star>★</span>) in the happy path, plus at most one rerun.</p>
<div class=flow>1  SELECT     <span class=star>*</span> Qwen (27B) sees case text + images + tables -> chooses WHICH tools to run
1b CANDIDATES <span class=star>*</span> if panderm/dermogpt is chosen: Qwen -> top-10 differential from
              text+image+table; PanDerm then RANKS that list visually
2  EXECUTE      run the selected tools (max 2 retries per tool)
              panderm / dermogpt = clinical photo & dermoscopy ; medgemma = any other modality
3  VERIFY     <span class=star>*</span> Qwen sees case + tool outputs -> CONFIRM (emit top-5 dx / test)
              or RERUN one tool (max 1 rerun of the loop) -> EMIT</div>
<h3>Models</h3>
<table>
<tr><th>Role</th><th>Model</th><th>Steps</th></tr>
<tr><td>Orchestrator (select + verify)</td><td>qwen/qwen3.5-27b via OpenRouter (no-think for dev)</td><td>1, 3</td></tr>
<tr><td>Clinical &amp; dermoscopy specialist</td><td>PanDerm (DermLIP zero-shot) + DermoGPT-RL</td><td>2</td></tr>
<tr><td>Other / unknown modality</td><td>MedGemma-1.5-4b-it</td><td>2</td></tr>
<tr><td>Grader judge</td><td>google/gemini-2.5-flash + ICD-11 hypernyms</td><td>eval</td></tr>
</table>
<h3>How to read each case</h3>
<p><b>Left</b> = the images. <b>Right</b> = the agentic chain: Step 1 which tools Qwen chose &amp; why
-> Step 2 what each tool reported -> Step 3 the confirm/rerun decision -> the final prediction vs
ground truth. Click <span class=star>&#9656; prompt</span> to see the exact prompt sent at each step.</p>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="one or more predictions JSONL files")
    ap.add_argument("--out", default="dermarena_traces.html")
    args = ap.parse_args()

    recs = []
    for p in args.preds:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))

    cards = "".join(_render_case(r) for r in recs)
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>DermArena agent traces</title><style>{CSS}</style></head><body>
<header><h1>DermArena agent — trace viewer (qualitative)</h1>
<div class=sub>{len(recs)} case(s) · SELECT→EXECUTE→VERIFY · sources: {_esc(', '.join(os.path.basename(p) for p in args.preds))}</div></header>
{SETUP_HTML}
{cards}</body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {len(recs)} case(s) -> {args.out}")


if __name__ == "__main__":
    main()
