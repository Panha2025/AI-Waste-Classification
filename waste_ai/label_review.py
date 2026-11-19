"""Local, offline image-label and duplicate review interface.

Candidate labels are provisional folder-derived cues. This module never promotes
them to verified labels without an explicit human decision in the review UI.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from waste_ai.config import ROOT, PROCESSED

CANDIDATES = PROCESSED / "candidate_manifest.csv"
DECISIONS = PROCESSED / "human_label_reviews.csv"
VERIFIED_EXPORT = PROCESSED / "human_verified_labels.csv"
AI_SUGGESTIONS = PROCESSED / "ai_label_suggestions.csv"
DUPLICATE_REVIEW = PROCESSED / "duplicate_review.csv"
NEAR_REVIEW = PROCESSED / "near_duplicate_review.csv"
PILOT_QUEUE = PROCESSED / "pilot_review_queue.csv"
LABELS = ("recyclable", "non_recyclable", "uncertain")
DECISIONS_ALLOWED = (*LABELS, "unreviewed")

PAGE = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>WasteAI local label review</title>
<style>
body{font:16px system-ui;margin:0;background:#f3f5f4;color:#17221e}header{background:#173d31;color:white;padding:18px 4vw}main{max-width:1120px;margin:auto;padding:20px}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0}.card{background:white;border:1px solid #d4ded9;border-radius:10px;padding:16px}.image{max-height:55vh;max-width:100%;object-fit:contain;background:#eee;display:block;margin:auto}.meta{font-size:14px;line-height:1.6;overflow-wrap:anywhere}.buttons button,button{padding:10px 14px;border:0;border-radius:6px;cursor:pointer;background:#e2ebe6}.buttons button:hover,button.primary{background:#176747;color:white}.warn{color:#884615}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}select,input{padding:8px}.muted{color:#53635b}#status{min-height:1.4em}@media(max-width:720px){.grid{grid-template-columns:1fr}}
</style><header><h1>WasteAI image review</h1><div>Offline local review. Source folders and AI suggestions are cues, never verified truth. Image appearance cannot establish Cambodian collection acceptance.</div></header>
<main><section class="card"><div class="bar"><label>View <select id="mode"><option value="pilot">600-image pilot queue</option><option value="all">All candidates</option><option value="duplicates">Duplicate pairs</option></select></label><label>Filter <select id="filter"><option value="unreviewed">Unreviewed</option><option value="all">All in view</option><option value="recyclable">Provisional recyclable</option><option value="non_recyclable">Provisional residual</option><option value="uncertain">Human-marked uncertain</option><option value="near">Near-duplicate flagged</option></select></label><button id="reload">Reload</button><span id="count"></span></div><div class="bar"><b>Pilot review progress</b><span id="progress">Loading…</span></div><p class="muted">Keys: ← previous · → next · 1 recyclable · 2 residual · 3 uncertain/exclude · 0 clear decision. Keys work when focus is outside a text field.</p><div id="card"></div><div class="bar"><button id="prev">Previous</button><span id="position"></span><button id="next">Next</button><span id="status"></span></div></section><p class="muted">Human decisions save locally to data/processed/human_label_reviews.csv. Export contains only explicit binary approvals; uncertain images remain excluded. This page does not create splits or start training.</p></main>
<script>
let rows=[],i=0,mode='pilot';const $=x=>document.getElementById(x);
async function load(){mode=$('mode').value;const q=new URLSearchParams({filter:$('filter').value,pool:mode});const endpoint=mode==='duplicates'?'/api/duplicates':'/api/items';const r=await fetch(endpoint+'?'+q);rows=await r.json();i=Math.min(i,Math.max(0,rows.length-1));render();await updateProgress()}
async function updateProgress(){const r=await fetch('/api/progress');if(r.ok){const p=await r.json();$('progress').textContent=`${p.reviewed}/${p.total} reviewed · recyclable ${p.approved.recyclable}/${p.target_per_class} · residual ${p.approved.non_recyclable}/${p.target_per_class} · uncertain ${p.uncertain} · pending ${p.pending}`}}
function render(){ $('count').textContent=rows.length+' review records';$('position').textContent=rows.length?`${i+1} / ${rows.length}`:'No records';const root=$('card');if(!rows.length){root.innerHTML='<p>No records for this filter.</p>';return}const x=rows[i];if(mode!=='duplicates'){root.innerHTML=`<div class="grid"><div><img class="image" src="/image?id=${encodeURIComponent(x.id)}"><div class="meta"><b>${x.name}</b><br>Source: ${x.source}; original category: ${x.original_category}<br>Provisional cue: ${x.proposed_binary_class||'uncertain'} <span class="warn">(folder-derived; not verified)</span><br>AI suggestion: ${x.ai_suggestion||'Unavailable: no local vision model/weights installed'}<br>AI confidence: ${x.ai_confidence??'—'}<br>Reason/status: ${x.ai_reason||x.ai_status||'No model inference available; local recycling rules not inferred.'}<br>QA flags: ${x.qa_flags||'none recorded'}<br>Reviewed: ${x.review_decision||'no'}</div></div><div class="meta"><b>Human decision</b><p>Assess the visible object and its condition. If its material or disposal route is unclear, choose uncertain.</p><div class="buttons"><button onclick="decide('recyclable')">Approve recyclable (1)</button> <button onclick="decide('non_recyclable')">Approve residual (2)</button><p><button onclick="decide('uncertain')">Uncertain / exclude (3)</button> <button onclick="decide('unreviewed')">Clear decision (0)</button></p></div><label>Reviewer <input id="reviewer" placeholder="name or initials"></label><br><label>Reason / note<br><input id="note" style="width:95%" placeholder="optional human review note"></label><p class="warn">Approval records your human label; it does not prove acceptance by a local recycler.</p></div></div>`}else{root.innerHTML=`<div class="meta"><b>${x.kind}</b> · ${x.label_a||''} ↔ ${x.label_b||''}<div class="grid">${x.paths.map((p,j)=>`<div><img class="image" src="/pair-image?index=${i}&side=${j}"><p>${p}</p></div>`).join('')}</div><p>${x.summary}</p><div class="buttons">${x.kind.startsWith('Exact')?`<button onclick="duplicate('same_image')">Confirm exact same image</button>`:`<button onclick="duplicate('same_image')">Same image, resized/edited</button> <button onclick="duplicate('same_object')">Same object, different photo</button>`} <button onclick="duplicate('distinct')">Distinct images / objects</button> <button onclick="duplicate('uncertain')">Uncertain; quarantine</button></div></div>`}}
async function decide(d){const x=rows[i],r=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:x.id,decision:d,reviewer:$('reviewer')?.value||'',note:$('note')?.value||''})});const msg=await r.text();$('status').textContent=msg;const f=$('filter').value,willDisappear=(f==='unreviewed'||(f==='uncertain'&&d!=='uncertain'));if(r.ok&&d!=='unreviewed'&&!willDisappear)i=Math.min(i+1,rows.length-1);await load()}
async function duplicate(d){const r=await fetch('/api/duplicate-decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:rows[i].id,decision:d})});$('status').textContent=await r.text();if(r.ok)i=Math.min(i+1,rows.length-1);await load()}
$('mode').onchange=()=>{i=0;load()};$('filter').onchange=()=>{i=0;load()};$('reload').onclick=load;$('prev').onclick=()=>{i=Math.max(0,i-1);render()};$('next').onclick=()=>{i=Math.min(rows.length-1,i+1);render()};document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowRight')$('next').click();if(e.key==='ArrowLeft')$('prev').click();if(mode!=='duplicates'&&e.key==='1')decide('recyclable');if(mode!=='duplicates'&&e.key==='2')decide('non_recyclable');if(mode!=='duplicates'&&e.key==='3')decide('uncertain');if(mode!=='duplicates'&&e.key==='0')decide('unreviewed')};load();
</script>'''


def _rows():
    frame = pd.read_csv(CANDIDATES).fillna("")
    flagged = {}
    if NEAR_REVIEW.exists():
        near_frame = pd.read_csv(NEAR_REVIEW).fillna("")
        for index, near_row in near_frame.iterrows():
            flag = f"suspected near-duplicate pair near-{index}"
            for source_path in (str(near_row["path"]), str(near_row["candidate_duplicate_of"])):
                flagged.setdefault(source_path, set()).add(flag)
    ai = {}
    if AI_SUGGESTIONS.exists():
        suggestion_frame = pd.read_csv(AI_SUGGESTIONS).fillna("")
        if {"id", "ai_suggestion", "confidence", "reason"}.issubset(suggestion_frame.columns):
            ai = {str(r["id"]): r for _, r in suggestion_frame.iterrows()}
    decisions = {}
    if DECISIONS.exists():
        with DECISIONS.open(newline="", encoding="utf-8") as f:
            decisions = {r["id"]: r for r in csv.DictReader(f)}
    result = []
    for _, row in frame.iterrows():
        ident = str(row["sha256"])
        d = decisions.get(ident, {})
        suggestion = ai.get(ident, {})
        ai_label = str(suggestion.get("ai_suggestion", ""))
        try:
            ai_confidence = float(suggestion.get("confidence"))
            if not math.isfinite(ai_confidence) or not 0 <= ai_confidence <= 1:
                ai_confidence = None
        except (TypeError, ValueError):
            ai_confidence = None
        ai_valid = ai_label in LABELS
        cat = str(row["original_category"])
        result.append({"id": ident, "name": Path(str(row["staged_path"])).name,
                       "path": str(row["staged_path"]), "source": row["source"],
                       "original_category": cat,
                       "proposed_binary_class": row["proposed_binary_class"],
                       "ai_suggestion": ai_label if ai_valid else "",
                       "ai_confidence": ai_confidence,
                       "ai_status": ("ai_suggested_unverified" if ai_valid else
                                     "invalid_suggestion_format" if suggestion else "unavailable_no_local_model"),
                       "ai_reason": str(suggestion.get("reason", "No local vision model or cached weights are installed. Folder label is a provisional cue only.")),
                       "qa_flags": "; ".join(sorted(flagged.get(str(row.get("original_path", "")), set()))),
                       "review_decision": d.get("decision", "")})
    result.sort(key=lambda r: (not bool(r["qa_flags"]), r["proposed_binary_class"] != "non_recyclable", r["source"], r["name"]))
    return result


def _duplicate_rows():
    rows = []
    if DUPLICATE_REVIEW.exists():
        frame = pd.read_csv(DUPLICATE_REVIEW).fillna("")
        for index, row in frame.iterrows():
            try:
                paths = json.loads(row["paths"])
                cats = json.loads(row["categories"])
            except Exception:
                continue
            if len(paths) < 2 or len(set(cats)) == 1:
                continue
            rows.append({"id": f"exact-{index}", "kind": "Exact duplicate label conflict",
                         "paths": paths[:2], "label_a": cats[0], "label_b": cats[1] if len(cats)>1 else "",
                         "summary": "Byte-identical files occur under conflicting source categories; keep quarantined until a human resolves label eligibility."})
    if NEAR_REVIEW.exists():
        frame = pd.read_csv(NEAR_REVIEW).fillna("")
        for index, row in frame.iterrows():
            rows.append({"id": f"near-{index}", "kind": "Suspected near duplicate (not confirmed)",
                         "paths": [str(row["path"]), str(row["candidate_duplicate_of"])],
                         "label_a": str(row["folder_label"]), "label_b": str(row["other_folder_label"]),
                         "summary": "Similarity detector candidate only. Same-image decisions prevent leakage; same-object decisions keep images in one split. Distinct candidates remain separate."})
    return rows


def _queue_ids():
    if not PILOT_QUEUE.exists():
        return set()
    frame = pd.read_csv(PILOT_QUEUE).fillna("")
    return set(frame.sha256.astype(str))


def _pilot_rows():
    if not PILOT_QUEUE.exists():
        return []
    queue = pd.read_csv(PILOT_QUEUE).fillna("").sort_values("queue_order")
    available = {row["id"]: row for row in _rows()}
    return [available[str(ident)] for ident in queue.sha256.astype(str) if str(ident) in available]


def _pilot_progress():
    if not PILOT_QUEUE.exists():
        return {"error": "Pilot queue missing; run python -m waste_ai.pilot first"}
    ids = _queue_ids()
    rows = [row for row in _rows() if row["id"] in ids]
    approved = {"recyclable": 0, "non_recyclable": 0}
    uncertain = 0
    for row in rows:
        if row["review_decision"] in approved:
            approved[row["review_decision"]] += 1
        elif row["review_decision"] == "uncertain":
            uncertain += 1
    reviewed = sum(approved.values()) + uncertain
    status = json.loads((PROCESSED / "pilot_review_queue_status.json").read_text()) if (PROCESSED / "pilot_review_queue_status.json").exists() else {}
    target = int(status.get("target_per_class", 300))
    total = len(ids)
    return {"total": total, "reviewed": reviewed, "approved": approved,
            "uncertain": uncertain, "pending": total - reviewed,
            "target_per_class": target}


def _save_csv(path: Path, rows, columns):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _save_decision(data):
    if data.get("decision") not in DECISIONS_ALLOWED:
        raise ValueError("Invalid review decision")
    ident = str(data.get("id", ""))
    candidates = _rows()
    match = next((r for r in candidates if r["id"] == ident), None)
    if match is None:
        raise ValueError("Unknown candidate id")
    existing = {}
    if DECISIONS.exists():
        with DECISIONS.open(newline="", encoding="utf-8") as f:
            existing = {r["id"]: r for r in csv.DictReader(f)}
    if data["decision"] == "unreviewed":
        existing.pop(ident, None)
    else:
        existing[ident] = {"id": ident, "path": match["path"], "source": match["source"],
                            "original_category": match["original_category"],
                            "decision": data["decision"], "review_status": "human_verified" if data["decision"] in LABELS[:2] else "human_uncertain_excluded",
                            "reviewer": str(data.get("reviewer", "")).strip(),
                            "note": str(data.get("note", "")).strip()}
    cols = ["id", "path", "source", "original_category", "decision", "review_status", "reviewer", "note"]
    _save_csv(DECISIONS, list(existing.values()), cols)
    verified = [r for r in existing.values() if r["decision"] in LABELS[:2]]
    _save_csv(VERIFIED_EXPORT, verified, cols)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, status, body, content_type="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if parsed.path == "/api/items":
            query = parse_qs(parsed.query); filt = query.get("filter", ["unreviewed"])[0]
            pool = query.get("pool", ["pilot"])[0]
            if pool == "pilot":
                if not PILOT_QUEUE.exists():
                    return self._send(409, json.dumps({"error": "Pilot queue missing; run python -m waste_ai.pilot first"}))
                rows = _pilot_rows()
            else:
                rows = _rows()
            if filt == "unreviewed": rows = [r for r in rows if not r["review_decision"]]
            elif filt in LABELS: rows = [r for r in rows if r["proposed_binary_class"] == filt]
            elif filt == "near": rows = [r for r in rows if r["qa_flags"]]
            elif filt == "uncertain": rows = [r for r in rows if r["review_decision"] == "uncertain"]
            return self._send(200, json.dumps(rows))
        if parsed.path == "/api/duplicates":
            return self._send(200, json.dumps(_duplicate_rows()))
        if parsed.path == "/api/progress":
            progress = _pilot_progress()
            return self._send(200 if "error" not in progress else 409, json.dumps(progress))
        if parsed.path in ("/image", "/pair-image"):
            if parsed.path == "/image":
                ident = parse_qs(parsed.query).get("id", [""])[0]
                row = next((r for r in _rows() if r["id"] == ident), None)
                path = row["path"] if row else ""
            else:
                index = int(parse_qs(parsed.query).get("index", ["-1"])[0]); side = int(parse_qs(parsed.query).get("side", ["-1"])[0])
                pairs = _duplicate_rows()
                path = pairs[index]["paths"][side] if 0 <= index < len(pairs) and 0 <= side < len(pairs[index]["paths"]) else ""
            resolved = (ROOT / path).resolve() if path else None
            if resolved is None or not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_file():
                return self._send(404, "Image not found", "text/plain")
            return self._send(200, resolved.read_bytes(), mimetypes.guess_type(str(resolved))[0] or "application/octet-stream")
        return self._send(404, "Not found", "text/plain")

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/api/decision":
                _save_decision(data)
            elif self.path == "/api/duplicate-decision":
                path = PROCESSED / "human_duplicate_reviews.csv"
                old = {}
                if path.exists():
                    with path.open(newline="", encoding="utf-8") as f: old = {r["id"]: r for r in csv.DictReader(f)}
                if data.get("decision") not in ("same_image", "same_object", "distinct", "uncertain"): raise ValueError("Invalid duplicate decision")
                pair = next((r for r in _duplicate_rows() if r["id"] == str(data["id"])), None)
                if pair is None: raise ValueError("Unknown duplicate pair")
                old[str(data["id"])] = {"id": str(data["id"]), "left_path": pair["paths"][0],
                                         "right_path": pair["paths"][1], "kind": pair["kind"],
                                         "decision": data["decision"], "review_status": "human_reviewed"}
                _save_csv(path, list(old.values()), ["id", "left_path", "right_path", "kind", "decision", "review_status"])
            else:
                return self._send(404, "Unknown route", "text/plain")
            return self._send(200, "Saved locally", "text/plain")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self._send(400, str(exc), "text/plain")


def main():
    parser = argparse.ArgumentParser(description="Start local WasteAI image-label review UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not CANDIDATES.exists():
        parser.error(f"Candidate manifest missing: {CANDIDATES}")
    if not PILOT_QUEUE.exists():
        parser.error(f"Pilot queue missing; first run `python -m waste_ai.pilot`: {PILOT_QUEUE}")
    print(f"Local review UI: http://{args.host}:{args.port} (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
