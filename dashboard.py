"""Tiny live dashboard for the LibreYOLO COCO val2017 sweep.

Stdlib only. Scans a results dir for va-bench submission JSONs and serves a
self-refreshing table at http://localhost:<port>. Rows appear as each variant
finishes; columns include the correctness status (val2017 / 5000 imgs / sane
detection mAP / GPU).

Usage:
    python dashboard.py [--results results_full_gpu] [--port 8077] [--total 50]
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ARGS = None


def validate(d: dict) -> list[str]:
    prob = []
    if d.get("schema_version") != "va.submission.v1":
        prob.append("schema")
    ds = d.get("dataset", {})
    if ds.get("id") != "coco2017" or ds.get("split") != "val2017":
        prob.append("dataset")
    if ds.get("num_images") != 5000:
        prob.append(f"imgs={ds.get('num_images')}")
    acc = d.get("accuracy", {})
    m = acc.get("mAP_50_95")
    if not isinstance(m, (int, float)) or not (0.0 <= m <= 1.0):
        prob.append("mAP")
    elif m == 0.0:
        prob.append("mAP=0")
    wf = str(d.get("model", {}).get("weights", "")).lower()
    if any(t in wf for t in ("seg", "pose", "keypoint", "l2cs")):
        prob.append("not-detection")
    rt = d.get("runtime", {})
    if rt.get("device") != "gpu" and rt.get("provider") != "cuda":
        prob.append("not-gpu")
    return prob


import re

_PROG_RE = re.compile(r"Benchmarking:\s+\d+%\|.*?\|\s*(\d+)/(\d+)")
_MODEL_RE = re.compile(r"Benchmarking:\s+(.+?)\s+\(([^)]+)\)\s+\[(?:PyTorch|ONNX)\]")
_DONE_RE = re.compile(r"Done\. Results in")


def parse_progress() -> dict:
    """Tail the sweep log to report the currently-running model + image progress."""
    log = getattr(ARGS, "log", None)
    if not log:
        return {"available": False}
    p = Path(log)
    if not p.exists():
        return {"available": False}
    try:
        # Read whole log (a few MB): model headers are rare and may be far
        # behind the latest tqdm lines, so a small tail can miss them.
        with open(p, "rb") as fh:
            tail = fh.read().decode("utf-8", "replace")
    except Exception:
        return {"available": False}

    done = bool(_DONE_RE.search(tail))
    # last model header
    model_name, model_key = None, None
    for m in _MODEL_RE.finditer(tail):
        model_name, model_key = m.group(1), m.group(2)
    # last progress counter (after the last model header if possible)
    cur = total = 0
    region = tail
    if model_key:
        idx = tail.rfind(f"({model_key})")
        region = tail[idx:]
    for m in _PROG_RE.finditer(region):
        cur, total = int(m.group(1)), int(m.group(2))
    return {
        "available": True,
        "done": done,
        "model": model_key or model_name,
        "model_name": model_name,
        "cur": cur,
        "total": total or 5000,
    }


def load_reference() -> dict:
    """Reload the paper/official reference map each call (so it can grow live)."""
    f = Path(__file__).parent / "reference_map.json"
    if not f.exists():
        return {}
    try:
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def collect(results_dir: Path) -> list[dict]:
    ref = load_reference()
    rows = []
    for f in results_dir.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        acc = d.get("accuracy", {})
        prob = validate(d)
        mid = d.get("model", {}).get("id", f.stem)
        m = acc.get("mAP_50_95")
        r = ref.get(mid, {})
        paper = r.get("paper_map")
        delta = None
        if isinstance(m, (int, float)) and isinstance(paper, (int, float)):
            delta = round((m - paper) * 100, 2)  # mAP points
        rows.append({
            "id": mid,
            "family": d.get("model", {}).get("family", ""),
            "mAP": m,
            "mAP50": acc.get("mAP_50"),
            "mAP75": acc.get("mAP_75"),
            "mAP_s": acc.get("mAP_small"),
            "mAP_m": acc.get("mAP_medium"),
            "mAP_l": acc.get("mAP_large"),
            "paper": paper,
            "delta": delta,
            "ref_src": r.get("source", ""),
            "fps": d.get("throughput", {}).get("fps_mean"),
            "params": d.get("model_stats", {}).get("params_millions"),
            "imgs": d.get("dataset", {}).get("num_images"),
            "device": d.get("runtime", {}).get("device"),
            "status": "OK" if not prob else "PROBLEM: " + ", ".join(prob),
        })
    rows.sort(key=lambda r: (r["mAP"] if isinstance(r["mAP"], (int, float)) else -1), reverse=True)
    return rows


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LibreYOLO COCO val2017 - live</title>
<style>
 body{font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0d1117;color:#e6edf3}
 header{padding:16px 24px;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:0}
 h1{margin:0 0 6px;font-size:18px}
 .bar{height:10px;background:#30363d;border-radius:5px;overflow:hidden;margin-top:8px;max-width:520px}
 .bar>i{display:block;height:100%;background:linear-gradient(90deg,#2ea043,#3fb950);width:0;transition:width .4s}
 .meta{color:#8b949e;font-size:12px}
 table{border-collapse:collapse;width:100%}
 th,td{padding:7px 12px;text-align:right;border-bottom:1px solid #21262d;white-space:nowrap}
 th{position:sticky;top:84px;background:#161b22;color:#8b949e;font-weight:600;cursor:default;font-size:12px}
 td.l,th.l{text-align:left}
 tr:hover{background:#161b22}
 .id{font-weight:600;color:#58a6ff}
 .fam{color:#8b949e;font-size:12px}
 .map{font-variant-numeric:tabular-nums;font-weight:600}
 .ok{color:#3fb950}.bad{color:#f85149}
 .dgood{color:#3fb950;font-weight:600}.dwarn{color:#d29922;font-weight:600}.dbad{color:#f85149;font-weight:600}
 .muted{color:#6e7681}
 .src{color:#58a6ff;text-decoration:none;font-size:12px}
 .src:hover{text-decoration:underline}
 tr.sec td{text-align:left;font-weight:700;font-size:12px;letter-spacing:.04em;padding:10px 12px;border-bottom:2px solid #30363d;position:sticky;top:84px}
 tr.secbad td{color:#f85149;background:#2d1416}
 tr.secok td{color:#3fb950;background:#0f1c12}
 tr.secmuted td{color:#8b949e;background:#161b22}
 tr.offrow{background:#2d141608}
 tr.offrow:hover{background:#2d1416}
 tr.offrow .id{color:#f0a39a}
 .new{animation:flash 1.6s ease-out}
 @keyframes flash{from{background:#1f6feb55}to{background:transparent}}
 .rank{color:#6e7681;width:28px}
 .now{margin-top:10px;display:flex;align-items:center;gap:10px;font-size:13px}
 .now b{color:#58a6ff}
 .dot{width:9px;height:9px;border-radius:50%;background:#3fb950;box-shadow:0 0 0 0 #3fb95080;animation:pulse 1.4s infinite}
 @keyframes pulse{0%{box-shadow:0 0 0 0 #3fb95080}70%{box-shadow:0 0 0 8px #3fb95000}100%{box-shadow:0 0 0 0 #3fb95000}}
 .ibar{flex:1;max-width:340px;height:8px;background:#30363d;border-radius:4px;overflow:hidden}
 .ibar>i{display:block;height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff);width:0;transition:width .3s}
</style></head><body>
<header>
 <h1>LibreYOLO - COCO val2017 (full 5000) live</h1>
 <div class="meta"><span id="count">0</span>/<span id="total">?</span> variants -
   GPU - object detection - sorted by mAP@50-95 - <span id="upd"></span></div>
 <div class="bar"><i id="prog"></i></div>
 <div id="now" class="now">
   <span class="dot"></span><b id="nowmodel">waiting...</b>
   <span id="nowimgs" class="meta"></span>
   <div class="ibar"><i id="iprog"></i></div>
 </div>
</header>
<table><thead><tr>
 <th class="rank"></th><th class="l">model</th><th class="l">family</th>
 <th>mAP<br>50-95</th><th>paper<br>mAP</th><th>&Delta;<br>pts</th><th>mAP50</th><th>mAP75</th>
 <th>AP<br>small</th><th>AP<br>med</th><th>AP<br>large</th>
 <th>FPS<br>(bs=1)</th><th>params<br>(M)</th><th>imgs</th><th class="l">status</th><th class="l">paper source</th>
</tr></thead><tbody id="rows"></tbody></table>
<script>
let seen=new Set();
function num(x,d=4){return (typeof x==='number')?x.toFixed(d):'-';}
function srcCell(u){
 if(!u) return '<td class="l muted">-</td>';
 let lbl=u;
 try{
  const m=u.match(/arxiv\.org\/abs\/([\d.]+)/);
  if(m){lbl='arXiv:'+m[1];}
  else if(u.includes('huggingface.co')){lbl='HF:'+u.split('huggingface.co/')[1];}
  else if(u.includes('github.com')){lbl='GitHub:'+u.split('github.com/')[1].split('/').slice(0,2).join('/');}
  else {lbl=u.replace(/^https?:\/\//,'').split('/')[0];}
 }catch(e){}
 return `<td class="l"><a href="${u}" target="_blank" rel="noopener" class="src">${lbl}</a></td>`;
}
async function tick(){
 try{
  const r=await fetch('/api/results');const j=await r.json();
  document.getElementById('count').textContent=j.count;
  document.getElementById('total').textContent=j.total;
  document.getElementById('prog').style.width=(100*j.count/j.total)+'%';
  document.getElementById('upd').textContent='updated '+new Date().toLocaleTimeString();
  const tb=document.getElementById('rows');tb.innerHTML='';
  function renderRow(m,rank,off){
   const tr=document.createElement('tr');
   let cls=off?'offrow':'';
   if(!seen.has(m.id)) cls+=' new';
   tr.className=cls.trim();
   const ok=m.status==='OK';
   let dcell='<td class="muted">-</td>';
   if(typeof m.delta==='number'){
    const a=Math.abs(m.delta);
    const dc=a<=0.3?'dgood':(a<=1.0?'dwarn':'dbad');
    const sign=m.delta>=0?'+':'';
    dcell=`<td class="${dc}" title="measured - paper, in mAP points">${sign}${m.delta.toFixed(2)}</td>`;
   }
   const paper=(typeof m.paper==='number')?m.paper.toFixed(4):'<span class="muted">no ref</span>';
   tr.innerHTML=`<td class="rank">${rank}</td>`+
    `<td class="l id">${m.id}</td><td class="l fam">${m.family}</td>`+
    `<td class="map">${num(m.mAP)}</td><td class="muted">${paper}</td>${dcell}`+
    `<td>${num(m.mAP50)}</td><td>${num(m.mAP75)}</td>`+
    `<td>${num(m.mAP_s)}</td><td>${num(m.mAP_m)}</td><td>${num(m.mAP_l)}</td>`+
    `<td>${num(m.fps,1)}</td><td>${num(m.params,2)}</td><td>${m.imgs??'-'}</td>`+
    `<td class="l ${ok?'ok':'bad'}">${ok?'OK':'WARN '+m.status}</td>`+
    srcCell(m.ref_src);
   return tr;
  }
  function section(label,cls){const tr=document.createElement('tr');tr.className='sec '+(cls||'');tr.innerHTML=`<td colspan="16">${label}</td>`;tb.appendChild(tr);}
  const off=[],okg=[],noref=[];
  j.results.forEach(m=>{ if(typeof m.delta==='number'){(Math.abs(m.delta)>0.3?off:okg).push(m);} else noref.push(m); });
  off.sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));
  let rank=0;
  if(off.length){ section('OFF FROM PAPER - abs(delta) > 0.3 pts ('+off.length+')','secbad'); off.forEach(m=>tb.appendChild(renderRow(m,++rank,true))); }
  section('matches paper - abs(delta) <= 0.3 ('+okg.length+')','secok'); okg.forEach(m=>tb.appendChild(renderRow(m,++rank,false)));
  if(noref.length){ section('no paper reference ('+noref.length+')','secmuted'); noref.forEach(m=>tb.appendChild(renderRow(m,++rank,false))); }
  j.results.forEach(m=>seen.add(m.id));
 }catch(e){document.getElementById('upd').textContent='(waiting...)';}
}
async function progress(){
 try{
  const r=await fetch('/api/progress');const p=await r.json();
  const nm=document.getElementById('nowmodel'), ni=document.getElementById('nowimgs'),
        ip=document.getElementById('iprog'), dot=document.querySelector('.dot');
  if(!p.available){nm.textContent='(progress unavailable)';return;}
  if(p.done){nm.textContent='sweep complete';ni.textContent='';ip.style.width='100%';dot.style.background='#6e7681';dot.style.animation='none';return;}
  const pct=p.total?Math.round(100*p.cur/p.total):0;
  nm.textContent='> '+(p.model||'loading...');
  ni.textContent=p.cur?`${p.cur}/${p.total} images (${pct}%)`:'loading weights / warming up...';
  ip.style.width=pct+'%';
 }catch(e){}
}
tick();setInterval(tick,2000);
progress();setInterval(progress,1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path.startswith("/api/progress"):
            body = json.dumps(parse_progress()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/results"):
            rows = collect(Path(ARGS.results))
            body = json.dumps({"count": len(rows), "total": ARGS.total, "results": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_full_gpu")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--log", default=None, help="Path to the sweep stdout log (for live progress bar)")
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"Dashboard: http://localhost:{ARGS.port}  (results: {ARGS.results}, total: {ARGS.total})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
