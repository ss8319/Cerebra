"""Tabbed qualitative comparison viewer: Cerebra vs DermAgent+MedGemma on the same cases.

Renders ONE self-contained HTML with two tabs:
  - Cerebra      : classify -> propose -> tool findings -> debate -> prediction
  - DermAgent+MG : per-tool evidence (PanDerm/MAKE/DermoGPT/MedGemma) -> Qwen reasoning -> prediction
Images embedded (base64). Cases matched by _id.

    PY=/fs04/.../dermagent/bin/python
    PYTHONPATH=. $PY -m cerebra.dermarena.render_compare \
        --cerebra <n1_validate_dir> --dermagent <warm_traces_dir> --out n1_traces.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image

TASKS = ["rds", "rdc", "dxtest"]


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


def _load(path: str) -> Optional[Dict]:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        line = f.readline().strip()  # n=1 per file
    return json.loads(line) if line else None


def _grade(gdir: str, cid: str) -> str:
    p = os.path.join(gdir, f"{cid}.json") if gdir else ""
    if p and os.path.isfile(p):
        d = json.load(open(p))
        t1, t5 = d.get("top1"), d.get("top5")
        if t1 is not None:
            return f'<span class="grade g{ "hit" if t1 else "miss" }">top1={t1} top5={t5}</span>'
    return ""


def _images_html(rec: Dict) -> str:
    """Case images from the Cerebra record's classified_modality (abs paths)."""
    cls = (rec or {}).get("classified_modality") or {}
    figs = []
    for path, mod in cls.items():
        uri = _img_uri(path)
        cap = f"{_esc(mod)}<br><span class=fn>{_esc(os.path.basename(path))}</span>"
        figs.append(f'<figure><img src="{uri}"><figcaption>{cap}</figcaption></figure>'
                    if uri else f'<figure class=missing><div>image?</div><figcaption>{cap}</figcaption></figure>')
    return "<div class=imgs>" + ("".join(figs) or "<span class=mut>no images</span>") + "</div>"


def _cerebra_case(rec: Dict, gdir: str) -> str:
    tr = rec.get("trace", {})
    cands = "".join(f"<li>{_esc(c)}</li>" for c in (rec.get("candidates") or [])) or "<li class=mut>(none)</li>"
    finds = ""
    for f in (tr.get("findings") or []):
        mod = f" · {_esc(f['modality'])}" if f.get("modality") else ""
        finds += (f'<div class=ev><div class=tool>{_esc(f.get("tool"))}{mod}</div>'
                  f'<div class=body>{_esc((f.get("summary") or "")[:600])}</div></div>')
    if not finds:
        finds = "<div class=mut>(vision skipped / no images)</div>"
    experts = "".join(f'<div class=ev><div class=tool>{_esc(o.get("role"))}</div>'
                      f'<div class=body>{_esc(o.get("opinion"))}</div></div>'
                      for o in (tr.get("expert_opinions") or []))
    return f"""
<div class=case>
  <div class=chead><span class=badge>{_esc(rec.get("task")).upper()}</span>
    <span class=cid>{_esc(rec.get("_id"))}</span> {_grade(gdir, rec.get("_id",""))}</div>
  <div class=gt><b>GT:</b> {_esc(rec.get("ground_truth"))}</div>
  {_images_html(rec)}
  <h4>1 · classify + route</h4><div class=mut>{_esc(tr.get("plan_routing"))}</div>
  <h4>2 · PROPOSE candidates</h4><ol class=cands>{cands}</ol>
  <h4>3 · RUN tool findings</h4>{finds}
  <h4>4 · DEBATE</h4>{experts or '<div class=mut>(none)</div>'}
  {('<details><summary>moderator raw</summary><pre>'+_esc(tr.get("moderator_raw"))+'</pre></details>') if tr.get("moderator_raw") else ''}
  <h4>Prediction</h4><pre class=pred>{_esc(rec.get("prediction"))}</pre>
</div>"""


def _dermagent_case(rec: Dict, cerebra_rec: Dict, gdir: str) -> str:
    ev = ""
    for part in (rec.get("evidence_parts") or []):
        # each part is "ToolName ...:\n<text>"; split the header line
        head, _, body = part.partition("\n")
        ev += f'<div class=ev><div class=tool>{_esc(head.rstrip(":"))}</div><div class=body>{_esc(body[:700])}</div></div>'
    if not ev:
        ev = "<div class=mut>(no tool evidence)</div>"
    reasoning = rec.get("reasoning") or ""
    reason_html = (f'<details><summary>Qwen reasoning (CoT, {len(reasoning)} chars)</summary>'
                   f'<pre>{_esc(reasoning[:4000])}</pre></details>') if reasoning else ""
    return f"""
<div class=case>
  <div class=chead><span class="badge da">{_esc(rec.get("task")).upper()}</span>
    <span class=cid>{_esc(rec.get("_id"))}</span> {_grade(gdir, rec.get("_id",""))}</div>
  <div class=gt><b>GT:</b> {_esc((cerebra_rec or {}).get("ground_truth"))}</div>
  {_images_html(cerebra_rec or {})}
  <h4>Tools (PanDerm · MAKE · DermoGPT · MedGemma) → evidence</h4>{ev}
  <h4>Qwen-27B (thinking) synthesis</h4>{reason_html}
  <h4>Prediction</h4><pre class=pred>{_esc(rec.get("prediction"))}</pre>
</div>"""


CSS = """
body{font:14.5px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
.wrap{max-width:1000px;margin:0 auto;padding:20px}
h1{font-size:20px;margin:0 0 12px}h4{margin:14px 0 5px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#0969da}
.tabs{display:flex;gap:8px;margin:12px 0;position:sticky;top:0;background:#f6f8fa;padding:8px 0;z-index:5}
.tab{padding:8px 18px;border:1px solid #d0d7de;border-radius:10px;background:#fff;cursor:pointer;font-weight:600;font-size:14px}
.tab.active{background:#0969da;color:#fff;border-color:#0969da}
.tab.active.da{background:#0a7c8c;border-color:#0a7c8c}
.pane{display:none}.pane.active{display:block}
.case{background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:16px 18px;margin:14px 0;box-shadow:0 1px 2px rgba(31,35,40,.05)}
.chead{display:flex;align-items:center;gap:10px}
.badge{background:#0969da;color:#fff;padding:2px 9px;border-radius:6px;font-size:11px;font-weight:600}
.badge.da{background:#0a7c8c}
.cid{color:#656d76;font-family:ui-monospace,monospace;font-size:12px}
.grade{margin-left:auto;font-family:ui-monospace,monospace;font-size:12px;padding:2px 8px;border-radius:6px}
.grade.ghit{background:#dafbe1;color:#1a7f37}.grade.gmiss{background:#ffebe9;color:#cf222e}
.gt{background:#eef6ff;border:1px solid #cfe3ff;border-radius:6px;padding:6px 10px;margin:8px 0;color:#0a3069}
.imgs{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}
figure{margin:0}figure img{max-height:150px;border-radius:6px;border:1px solid #d0d7de}
figcaption{font-size:11px;color:#656d76}.fn{color:#8c959f}
.missing div{padding:24px;border:1px dashed #d0d7de;border-radius:6px;color:#8c959f;font-size:12px}
ol.cands{margin:0;padding-left:20px}ol.cands li{margin:1px 0}
.ev{border-left:3px solid #d0d7de;padding-left:10px;margin:0 0 9px}
.tool{font-weight:600;color:#8250df;font-size:12.5px}
.ev .body{color:#424a53;font-size:13px;white-space:pre-wrap}
.mut{color:#8c959f;font-style:italic;font-size:13px}
pre{white-space:pre-wrap;background:#0d1117;color:#e6edf3;border-radius:8px;padding:10px;font-size:12px}
pre.pred{background:#0d1a12;color:#7ee787;border:1px solid #238636}
details summary{cursor:pointer;color:#8250df;font-size:12px;margin:4px 0}
"""

JS = """
function show(t){document.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));
document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
document.getElementById(t).classList.add('active');document.getElementById('btn-'+t).classList.add('active');}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cerebra", required=True, help="Cerebra n1_validate dir")
    ap.add_argument("--dermagent", required=True, help="DermAgent warm_traces dir")
    ap.add_argument("--cerebra-grade-suffix", default="_regrade2")
    ap.add_argument("--dermagent-grade-suffix", default="_grade")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cb_cases, da_cases = [], []
    for t in TASKS:
        cb = _load(os.path.join(args.cerebra, f"{t}_pred.jsonl"))
        da = _load(os.path.join(args.dermagent, f"{t}.jsonl"))
        if cb:
            cb_cases.append(_cerebra_case(cb, os.path.join(args.cerebra, f"{t}{args.cerebra_grade_suffix}")))
        if da:
            da_cases.append(_dermagent_case(da, cb, os.path.join(args.dermagent, f"{t}{args.dermagent_grade_suffix}")))

    doc = f"""<!doctype html><html><head><meta charset=utf-8><title>Cerebra vs DermAgent+MedGemma — traces</title>
<style>{CSS}</style></head><body><div class=wrap>
<h1>Qualitative traces — same cases, Qwen-27B</h1>
<div class=tabs>
  <div class=tab id=btn-cerebra onclick="show('cerebra')">Our Cerebra</div>
  <div class="tab da" id=btn-dermagent onclick="show('dermagent')">DermAgent + MedGemma</div>
</div>
<div class="pane active" id=cerebra>{''.join(cb_cases) or '<p class=mut>no Cerebra records</p>'}</div>
<div class=pane id=dermagent>{''.join(da_cases) or '<p class=mut>no DermAgent records</p>'}</div>
</div><script>{JS}
document.getElementById('btn-cerebra').classList.add('active');
</script></body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {len(cb_cases)} Cerebra + {len(da_cases)} DermAgent cases -> {args.out}")


if __name__ == "__main__":
    main()
