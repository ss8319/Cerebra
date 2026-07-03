"""Render DermArena agent predictions (JSONL) into a self-contained HTML trace viewer.

One card per case: images inline (with classified modality) on the left, the full
agentic chain on the right — Step 1 classification, PROPOSE candidates, RUN tool
findings, DEBATE expert opinions + moderator, final prediction vs ground truth.
Images are embedded as base64 so the HTML is a single portable file.

    PY=/fs04/scratch2/ub62/ssim0070/dermagent/bin/python
    PYTHONPATH=. $PY -m cerebra.dermarena.render_traces preds1.jsonl [preds2.jsonl ...] --out traces.html
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


def _img_data_uri(path: str, max_side: int = 420) -> str:
    try:
        img = Image.open(path).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


# Prompt templates, imported from the pipeline modules so the viewer shows exactly
# what each stage asks the model (works on any predictions file, no re-run needed).
from cerebra.dermarena import classify as _classify, propose as _propose, debate as _debate


def _prompt_details(label: str, pairs: List[tuple]) -> str:
    """pairs = [(sublabel, text), ...] rendered as SYSTEM/USER blocks in a <details>."""
    inner = "".join(f'<div class=pl>{_esc(sl)}</div><pre>{_esc(t)}</pre>' for sl, t in pairs)
    return f'<details class=prompt><summary>▸ prompt · {_esc(label)}</summary>{inner}</details>'


def _debate_prompt(task: str) -> str:
    pairs = []
    for role, persona in _debate.EXPERTS:
        pairs.append((f"{role} — SYSTEM", persona))
    pairs.append(("expert — USER (template)", _debate.EXPERT_ASK))
    try:
        ti = _debate._task_instruction(task)
    except Exception:
        ti = ""
    pairs.append(("moderator — SYSTEM", _debate.MODERATOR_SYSTEM))
    pairs.append(("moderator — USER (template)", _debate.MODERATOR_ASK.format(task_instruction=ti)))
    return _prompt_details("debate (experts + moderator)", pairs)


def _case_images(rec: Dict[str, Any]) -> List[Dict[str, str]]:
    """Collect {path, modality} for the case from classified_modality + findings."""
    out, seen = [], set()
    cls = rec.get("classified_modality") or {}
    for path, mod in cls.items():
        if path not in seen:
            out.append({"path": path, "modality": mod}); seen.add(path)
    for f in (rec.get("trace", {}).get("findings") or []):
        p = f.get("image_path")
        if p and p not in seen:
            out.append({"path": p, "modality": f.get("modality") or "?"}); seen.add(p)
    return out


def _render_case(rec: Dict[str, Any]) -> str:
    tr = rec.get("trace", {})
    task = rec.get("task", "")
    # left: images
    imgs_html = []
    for im in _case_images(rec):
        uri = _img_data_uri(im["path"])
        cap = _esc(im["modality"]) + "<br><span class=fn>" + _esc(os.path.basename(im["path"])) + "</span>"
        if uri:
            imgs_html.append(f'<figure><img src="{uri}"><figcaption>{cap}</figcaption></figure>')
        else:
            imgs_html.append(f'<figure class=missing><div>image unavailable</div><figcaption>{cap}</figcaption></figure>')
    if not imgs_html:
        imgs_html.append('<div class=noimg>no images (text-only case)</div>')

    # right: chain
    cands = "".join(f"<li>{_esc(c)}</li>" for c in (rec.get("candidates") or [])) or "<li class=empty>(none)</li>"
    findings = tr.get("findings") or []
    find_html = ""
    for f in findings:
        tag = _esc(f.get("tool"))
        mod = f" · {_esc(f.get('modality'))}" if f.get("modality") else ""
        fd = f' <span class=fdx>→ {_esc(f.get("final_diagnosis"))}</span>' if f.get("final_diagnosis") else ""
        find_html += f'<div class=finding><div class=tool>{tag}{mod}{fd}</div><div class=body>{_esc(f.get("summary"))}</div></div>'
    if not findings:
        find_html = '<div class=empty>(vision skipped / no images)</div>'

    experts = ""
    for o in (tr.get("expert_opinions") or []):
        experts += f'<div class=expert><span class=role>{_esc(o.get("role"))}</span>{_esc(o.get("opinion"))}</div>'

    gt = _esc(rec.get("ground_truth"))
    gt_test = rec.get("diagnostic_test_gt")
    gt_line = f'<div class=gt><b>GT dx:</b> {gt}' + (f' &nbsp;<b>GT test:</b> {_esc(gt_test)}' if gt_test else "") + "</div>"

    raw_blocks = ""
    if tr.get("propose_raw"):
        raw_blocks += f'<details><summary>PROPOSE raw</summary><pre>{_esc(tr["propose_raw"])}</pre></details>'
    if tr.get("moderator_raw"):
        raw_blocks += f'<details><summary>MODERATOR raw</summary><pre>{_esc(tr["moderator_raw"])}</pre></details>'

    return f"""
<section class=case>
  <div class=head>
    <span class=badge>{_esc(task).upper()}</span>
    <span class=cid>{_esc(rec.get("_id"))}</span>
    <span class=cost>${rec.get("cumulative_usd", 0):.4f}</span>
  </div>
  {gt_line}
  <div class=cols>
    <div class=left><h4>Step 1 · images + classified modality</h4>
      {_prompt_details("Step 1 modality classifier", [("SYSTEM", _classify.SYSTEM), ("USER", _classify.USER)])}
      {''.join(imgs_html)}
      <div class=routing>{_esc(tr.get("plan_routing"))}</div>
    </div>
    <div class=right>
      <h4>Step 2 · PROPOSE candidates</h4>
      {_prompt_details("Step 2 candidate proposer", [("SYSTEM", _propose.SYSTEM_PROMPT), ("USER (template)", _propose.USER_TEMPLATE)])}
      <ol class=cands>{cands}</ol>
      <h4>Step 3 · RUN findings</h4>{find_html}
      <h4>Step 4 · DEBATE</h4>
      {_debate_prompt(task)}
      {experts or '<div class=empty>(no experts)</div>'}
      {raw_blocks}
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
.routing{font-size:11px;color:#6e7681;margin-top:8px;font-family:monospace;word-break:break-all}
ol.cands{margin:0;padding-left:20px}ol.cands li{margin:2px 0}
.finding{margin:0 0 10px;border-left:3px solid #30363d;padding-left:10px}
.tool{font-weight:600;color:#d2a8ff;font-size:12px}.fdx{color:#7ee787;font-weight:400}
.finding .body{color:#adbac7;font-size:13px;white-space:pre-wrap}
.expert{margin:0 0 8px}.role{display:inline-block;font-weight:600;color:#f0883e;margin-right:6px}
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
<p>A lean per-case pipeline for the DermArena dx-track (staying on the Cerebra repo). Key idea:
the diagnosis is built from <b>vision + fusion</b> — the shown narrative has its image
descriptions removed, so findings must come from the images. Only two steps call an LLM (<span class=star>★</span>);
the rest is deterministic.</p>
<div class=flow>0 LOAD      case row -> masked report + images (+ examination results for RDC)
1 ANALYSE   classify EACH image's modality with the base MLLM (we do NOT trust the dataset label) -> route
2 PROPOSE <span class=star>*</span> base MLLM sees images + report -> candidate diagnoses
3 RUN       clinical_photo / dermoscopy -> PanDerm (re-ranks the candidates) + DermoGPT
            everything else / unknown   -> MedGemma
4 DEBATE  <span class=star>*</span> 3-expert panel + moderator fuse candidates + tool findings + narrative
            -> RDS/RDC: ranked top-5 diagnosis ; DxTest: test(s) to order
5 EMIT      prediction string (graded by ICD-11 hypernym match + an LLM judge)</div>
<h3>Models used</h3>
<table>
<tr><th>Role</th><th>Model</th><th>Steps</th></tr>
<tr><td>Base MLLM (classify / propose / debate)</td><td>qwen/qwen3.5-27b via OpenRouter (no-think for dev)</td><td>1, 2, 4</td></tr>
<tr><td>Clinical &amp; dermoscopy specialist</td><td>PanDerm (DermLIP zero-shot) + DermoGPT-RL (Qwen3-VL-8B)</td><td>3</td></tr>
<tr><td>Other / unknown modality</td><td>MedGemma-1.5-4b-it</td><td>3</td></tr>
<tr><td>Grader judge</td><td>google/gemini-2.5-flash + ICD-11 hypernym map</td><td>eval</td></tr>
</table>
<h3>How to read each case</h3>
<p><b>Left column</b> = the images with the modality the classifier assigned (Step 1).
<b>Right column</b> = the agentic chain: Step 2 candidates -> Step 3 what each vision tool
reported -> Step 4 the expert debate -> the final prediction vs the ground truth.
Click <span class=star>&#9656; prompt</span> under any step to see the exact prompt that was sent.</p>
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

    total = sum(r.get("cumulative_usd", 0) for r in recs[-1:])  # ledger is cumulative; last = total
    cards = "".join(_render_case(r) for r in recs)
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>DermArena agent traces</title><style>{CSS}</style></head><body>
<header><h1>DermArena agent — trace viewer (qualitative)</h1>
<div class=sub>{len(recs)} case(s) · Cerebra pipeline · sources: {_esc(', '.join(os.path.basename(p) for p in args.preds))}</div></header>
{SETUP_HTML}
{cards}</body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {len(recs)} case(s) -> {args.out}")


if __name__ == "__main__":
    main()
