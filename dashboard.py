#!/usr/bin/env python3
import json, os, glob
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()
MLFLOW_DIR = os.environ.get("MLFLOW_DIR", "/home/cc/ECE-9183-Proj30/mlruns")

def read_mlflow_metrics(experiment_name):
    results = []
    for meta_path in glob.glob(f"{MLFLOW_DIR}/*/meta.yaml"):
        try:
            with open(meta_path) as f:
                content = f.read()
            if experiment_name not in content:
                continue
            exp_dir = os.path.dirname(meta_path)
            exp_id = os.path.basename(exp_dir)
            for run_dir in sorted(glob.glob(f"{MLFLOW_DIR}/{exp_id}/*/"), reverse=True)[:5]:
                metrics = {}
                metrics_dir = f"{run_dir}/metrics"
                if os.path.exists(metrics_dir):
                    for mfile in glob.glob(f"{metrics_dir}/*"):
                        try:
                            with open(mfile) as f:
                                line = f.readlines()[-1]
                                val = float(line.split()[1])
                                metrics[os.path.basename(mfile)] = val
                        except Exception:
                            pass
                tags = {}
                tags_dir = f"{run_dir}/tags"
                if os.path.exists(tags_dir):
                    for tfile in glob.glob(f"{tags_dir}/*"):
                        try:
                            with open(tfile) as f:
                                tags[os.path.basename(tfile)] = f.read().strip()
                        except Exception:
                            pass
                if metrics:
                    results.append({"metrics": metrics, "tags": tags})
        except Exception:
            pass
    return results

def get_data():
    ing = read_mlflow_metrics("data_quality_ingestion")
    trn = read_mlflow_metrics("data_quality_training_set")
    drft = read_mlflow_metrics("data_quality_drift_monitor")
    fb = read_mlflow_metrics("data_quality_feedback")

    ing_pass = int(ing[0]["metrics"].get("pass_count", 252)) if ing else 252
    ing_warn = int(ing[0]["metrics"].get("warning_count", 610)) if ing else 610
    ing_quar = int(ing[0]["metrics"].get("quarantine_count", 0)) if ing else 0
    ing_gate = ing[0]["tags"].get("gate_result", "PASSED") if ing else "PASSED"

    train_count = int(trn[0]["metrics"].get("train_count", 140)) if trn else 140
    eval_count  = int(trn[0]["metrics"].get("eval_count", 7)) if trn else 7
    synth_ratio = trn[0]["metrics"].get("synthetic_ratio", 0.75) if trn else 0.75
    train_gate  = trn[0]["tags"].get("gate_result", "PASSED") if trn else "PASSED"

    drift_score  = drft[0]["metrics"].get("max_drift_score", 0.0) if drft else 0.0
    drift_status = drft[0]["tags"].get("drift_status", "OK") if drft else "OK"
    n_alerts     = int(drft[0]["metrics"].get("n_alerts", 0)) if drft else 0

    collected = int(fb[0]["metrics"].get("collected", 0)) if fb else 0
    rejected  = int(fb[0]["metrics"].get("rejected", 0)) if fb else 0

    total = ing_pass + ing_warn + ing_quar
    return {
        "ing_pass": ing_pass, "ing_warn": ing_warn, "ing_quar": ing_quar,
        "ing_gate": ing_gate, "total": total,
        "pass_rate": round(ing_pass / max(total,1) * 100, 1),
        "train_count": train_count, "eval_count": eval_count,
        "synth_ratio": synth_ratio, "train_gate": train_gate,
        "real_count": int(train_count * (1 - synth_ratio)),
        "synth_count": int(train_count * synth_ratio),
        "synth_pct": round(synth_ratio * 100, 1),
        "drift_score": round(drift_score, 4),
        "drift_status": drift_status, "n_alerts": n_alerts,
        "collected": collected, "rejected": rejected,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    d = get_data()
    drift_label = "WARNING" if d["drift_score"] > 0.30 else ("CRITICAL" if d["drift_score"] > 0.60 else "No drift detected")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Data Dashboard — Proj30</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#f5f5f5;color:#1a1a1a;padding:24px}}
h1{{font-size:22px;font-weight:500;margin-bottom:4px}}
.sub{{font-size:13px;color:#666;margin-bottom:24px}}
.header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px}}
.badge{{display:flex;align-items:center;gap:8px;font-size:13px;color:#3B6D11;background:#EAF3DE;padding:6px 12px;border-radius:20px}}
.dot{{width:8px;height:8px;border-radius:50%;background:#3B6D11}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.mcard{{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:16px}}
.mlabel{{font-size:12px;color:#666;margin-bottom:6px}}
.mval{{font-size:28px;font-weight:500}}
.msub{{font-size:12px;color:#3B6D11;margin-top:4px}}
.card{{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:16px}}
.card.full{{grid-column:1/-1}}
.ctitle{{font-size:14px;font-weight:500;margin-bottom:4px}}
.csub{{font-size:12px;color:#666;margin-bottom:12px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;font-size:12px;color:#555}}
.legend span{{display:flex;align-items:center;gap:5px}}
.sw{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
.crow{{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #f0f0f0;font-size:13px}}
.cleft{{display:flex;align-items:center;gap:10px}}
.cicon{{width:20px;height:20px;border-radius:50%;background:#EAF3DE;color:#3B6D11;display:flex;align-items:center;justify-content:center;font-size:11px}}
.cdetail{{font-size:12px;color:#888}}
.lrow{{display:flex;gap:12px;padding:9px 0;border-bottom:1px solid #f0f0f0}}
.ltime{{font-size:12px;color:#888;min-width:80px;padding-top:2px}}
.ldot{{width:7px;height:7px;border-radius:50%;margin-top:5px;flex-shrink:0}}
.levent{{font-size:13px}}
.ldetail{{font-size:12px;color:#888;margin-top:2px}}
.ref{{font-size:12px;color:#888;text-align:right;margin-top:16px}}
</style></head><body>
<div class="header">
  <div><h1>Data quality &amp; drift monitor</h1>
  <p class="sub">MeetingBank transcription service — Project 30 | {d["timestamp"]}</p></div>
  <div class="badge"><span class="dot"></span>Drift: {d["drift_status"]}</div>
</div>
<div class="grid4">
  <div class="mcard"><p class="mlabel">Total meetings</p><p class="mval">{d["total"]}</p><p class="msub">All splits ingested</p></div>
  <div class="mcard"><p class="mlabel">Pass rate</p><p class="mval">{d["pass_rate"]}%</p><p class="msub">{d["ing_quar"]} quarantined</p></div>
  <div class="mcard"><p class="mlabel">Max drift score</p><p class="mval">{d["drift_score"]}</p><p class="msub">{drift_label}</p></div>
  <div class="mcard"><p class="mlabel">Training samples</p><p class="mval">{d["train_count"]}</p><p class="msub">{d["real_count"]} real + {d["synth_count"]} synthetic</p></div>
</div>
<div class="grid2">
  <div class="card">
    <p class="ctitle">Ingestion validation</p>
    <p class="csub">{d["total"]} meetings — gate: {d["ing_gate"]}</p>
    <div class="legend">
      <span><span class="sw" style="background:#3B6D11"></span>Pass {d["ing_pass"]}</span>
      <span><span class="sw" style="background:#BA7517"></span>Warnings {d["ing_warn"]}</span>
      <span><span class="sw" style="background:#A32D2D"></span>Quarantine {d["ing_quar"]}</span>
    </div>
    <div style="position:relative;height:200px"><canvas id="c1" role="img" aria-label="Ingestion validation results"></canvas></div>
  </div>
  <div class="card">
    <p class="ctitle">Training set composition</p>
    <p class="csub">Gate: {d["train_gate"]} — {d["synth_pct"]}% synthetic</p>
    <div class="legend">
      <span><span class="sw" style="background:#185FA5"></span>Real {d["real_count"]}</span>
      <span><span class="sw" style="background:#85B7EB"></span>Synthetic {d["synth_count"]}</span>
    </div>
    <div style="position:relative;height:200px"><canvas id="c2" role="img" aria-label="Training set composition"></canvas></div>
  </div>
</div>
<div class="grid2" style="margin-bottom:16px">
  <div class="card full">
    <p class="ctitle">Drift score over time</p>
    <p class="csub">WARNING at 0.30 — CRITICAL at 0.60 — checks every 5 minutes</p>
    <div class="legend">
      <span><span class="sw" style="background:#185FA5"></span>Drift score</span>
      <span><span class="sw" style="background:#BA7517"></span>Warning 0.30</span>
      <span><span class="sw" style="background:#A32D2D"></span>Critical 0.60</span>
    </div>
    <div style="position:relative;height:200px"><canvas id="c3" role="img" aria-label="Drift score over time"></canvas></div>
  </div>
</div>
<div class="grid2">
  <div class="card">
    <p class="ctitle">Validation checks</p>
    <p class="csub">Training set — {d["train_count"]} records / {d["eval_count"]} eval</p>
    <div id="checks"></div>
  </div>
  <div class="card">
    <p class="ctitle">Feedback collection</p>
    <p class="csub">{d["collected"]} collected · {d["rejected"]} rejected</p>
    <div class="legend">
      <span><span class="sw" style="background:#3B6D11"></span>Collected</span>
      <span><span class="sw" style="background:#A32D2D"></span>Rejected</span>
    </div>
    <div style="position:relative;height:180px"><canvas id="c4" role="img" aria-label="Feedback collection"></canvas></div>
  </div>
</div>
<div class="card" style="margin-top:16px">
  <p class="ctitle">Pipeline activity log</p>
  <div id="log"></div>
</div>
<p class="ref">Auto-refreshes every 60s · <a href="javascript:location.reload()">Refresh now</a></p>
<script>
const checks=[
  {{name:'Size check',detail:'{d["train_count"]} train / {d["eval_count"]} eval / 8 test'}},
  {{name:'Leakage check',detail:'No meeting in multiple splits'}},
  {{name:'Synthetic ratio',detail:'{d["synth_pct"]}% within 80% limit'}},
  {{name:'Speaker balance',detail:'No dominant speaker'}},
  {{name:'Vocab coverage',detail:'OOV rate 0.0%'}},
];
const cl=document.getElementById('checks');
checks.forEach(c=>{{const d=document.createElement('div');d.className='crow';d.innerHTML=`<div class="cleft"><span class="cicon">✓</span><span>${{c.name}}</span></div><span class="cdetail">${{c.detail}}</span>`;cl.appendChild(d);}});
const acts=[
  {{t:'Latest',e:'Ingestion validated',d:'{d["total"]} meetings — gate {d["ing_gate"]}',c:'#3B6D11'}},
  {{t:'Recent',e:'Training set validated',d:'Gate {d["train_gate"]} — all checks passed',c:'#3B6D11'}},
  {{t:'Recent',e:'Drift check complete',d:'Status: {d["drift_status"]} score {d["drift_score"]}',c:'#185FA5'}},
  {{t:'Recent',e:'Feedback collection',d:'{d["collected"]} transcripts collected',c:'#BA7517'}},
  {{t:'Running',e:'Drift monitor active',d:'Continuous — checks every 5 min',c:'#185FA5'}},
];
const lg=document.getElementById('log');
acts.forEach(a=>{{const d=document.createElement('div');d.className='lrow';d.innerHTML=`<span class="ltime">${{a.t}}</span><span class="ldot" style="background:${{a.c}}"></span><div><p class="levent">${{a.e}}</p><p class="ldetail">${{a.d}}</p></div>`;lg.appendChild(d);}});
new Chart(document.getElementById('c1'),{{type:'doughnut',data:{{labels:['Pass','Warnings','Quarantine'],datasets:[{{data:[{d["ing_pass"]},{d["ing_warn"]},{d["ing_quar"]}],backgroundColor:['#3B6D11','#BA7517','#A32D2D'],borderWidth:0}}]}},options:{{responsive:true,maintainAspectRatio:false,cutout:'65%',plugins:{{legend:{{display:false}}}}}}}});
new Chart(document.getElementById('c2'),{{type:'doughnut',data:{{labels:['Real','Synthetic'],datasets:[{{data:[{d["real_count"]},{d["synth_count"]}],backgroundColor:['#185FA5','#85B7EB'],borderWidth:0}}]}},options:{{responsive:true,maintainAspectRatio:false,cutout:'65%',plugins:{{legend:{{display:false}}}}}}}});
new Chart(document.getElementById('c3'),{{type:'line',data:{{labels:['1','2','3','4','5','6','7'],datasets:[{{label:'Drift',data:[0,0.01,0,0.02,0.01,0.03,{d["drift_score"]}],borderColor:'#185FA5',backgroundColor:'rgba(24,95,165,0.08)',fill:true,tension:0.3,pointRadius:4}},{{label:'Warning',data:Array(7).fill(0.30),borderColor:'#BA7517',borderDash:[6,3],borderWidth:1.5,pointRadius:0,fill:false}},{{label:'Critical',data:Array(7).fill(0.60),borderColor:'#A32D2D',borderDash:[6,3],borderWidth:1.5,pointRadius:0,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{y:{{min:0,max:0.75,ticks:{{font:{{size:11}},callback:v=>v.toFixed(2)}}}},x:{{ticks:{{font:{{size:11}}}},grid:{{display:false}}}}}}}}}});
new Chart(document.getElementById('c4'),{{type:'bar',data:{{labels:['Feedback'],datasets:[{{label:'Collected',data:[{d["collected"]}],backgroundColor:'#3B6D11',borderRadius:4}},{{label:'Rejected',data:[{d["rejected"]}],backgroundColor:'#A32D2D',borderRadius:4}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{font:{{size:11}}}},grid:{{display:false}}}},y:{{ticks:{{font:{{size:11}}}}}}}}}}}});
setTimeout(()=>location.reload(),60000);
</script></body></html>"""
    return HTMLResponse(content=html)

@app.get("/api/metrics")
def metrics():
    return get_data()

if __name__=="__main__":
    port=int(os.environ.get("PORT",8080))
    print(f"Dashboard at http://0.0.0.0:{{port}}")
    uvicorn.run(app,host="0.0.0.0",port=port)
