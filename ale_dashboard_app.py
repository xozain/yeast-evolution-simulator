"""
S. cerevisiae ALE Live Dashboard
=================================
Flask + Server-Sent Events (SSE) real-time simulation dashboard.
Run with:  python app.py
Then open: http://localhost:5000
"""

import json, math, time, threading, queue, os
from flask import Flask, Response, render_template_string, request, jsonify
import numpy as np

app = Flask(__name__)

# ══════════════════════════════════════════════════════════
#  SIMULATION ENGINE  (vectorised, streamable)
# ══════════════════════════════════════════════════════════

def build_temp_schedule(ramp_speed, t_start=30, t_end=42):
    """Build temperature step schedule from ramp speed slider (0.5–3x)."""
    base = [(0,30),(100,33),(200,36),(300,38),(450,40),(600,42)]
    if ramp_speed == 1.0:
        return base
    compressed = []
    for (g, T) in base:
        new_g = int(g / ramp_speed)
        compressed.append((new_g, T))
    return compressed

def current_temp_fn(gen, schedule):
    T = schedule[0][1]
    for (g0, t0) in schedule:
        if gen >= g0:
            T = t0
    return float(T)

def growth_rate(T, Topt, Tmax, muopt, Tmin=4.0):
    if T <= Tmin or T >= Tmax:
        return 0.0
    num = (T - Tmax) * (T - Tmin)**2
    d1  = (Topt - Tmin)
    d2  = (Topt - Tmin)*(T - Topt) - (Topt - Tmax)*(Topt + Tmin - 2*T)
    den = d1 * d2
    if abs(den) < 1e-14:
        return 0.0
    return float(max(0.0, muopt * num / den))

def growth_rate_vec(T, Topt_a, Tmax_a, muopt_a, Tmin=4.0):
    rates = np.zeros(len(Topt_a))
    mask  = (T > Tmin) & (T < Tmax_a)
    if not np.any(mask):
        return rates
    To=Topt_a[mask]; Tx=Tmax_a[mask]; mu=muopt_a[mask]
    num = (T - Tx)*(T - Tmin)**2
    d1  = (To - Tmin)
    d2  = (To - Tmin)*(T - To) - (To - Tx)*(To + Tmin - 2*T)
    den = d1*d2
    vals = np.where(np.abs(den)>1e-14, mu*num/den, 0.0)
    rates[mask] = np.maximum(0.0, vals)
    return rates

# Gene modules
GENE_MODULES = {
    "ERG3_LOF":   (1.6,2.2,-0.08,3.5,0.4),
    "ETC_LOF":    (1.0,1.8,-0.12,3.0,0.3),
    "CDC25_reg":  (0.9,1.1,-0.03,2.5,0.9),
    "HSF1_pt":    (0.5,0.9,-0.01,0.6,2.2),
    "SKN7_pt":    (0.4,0.7,-0.01,0.5,1.8),
    "HSP104":     (0.45,0.8,-0.015,1.2,2.2),
    "TPS1_TPS2":  (0.35,0.6,-0.01,0.7,1.8),
    "HOG1_reg":   (0.35,0.6,-0.01,0.9,1.6),
    "RAS2_IRA2":  (0.5,0.7,-0.02,1.5,1.0),
    "Cell_wall":  (0.2,0.3,-0.005,0.7,1.3),
}
MODULE_NAMES = list(GENE_MODULES.keys())
N_MODULES    = len(MODULE_NAMES)
N_CHROM      = 16
WT_TOPT=30.8; WT_TMAX=40.0; WT_MUOPT=0.40
CHROM_BENEFIT = {3:0.04, 6:0.03, 9:0.035, 13:0.03}
CHROM_NGENES  = np.array([119,428,170,773,289,262,578,298,439,373,349,579,476,405,290,513],dtype=float)

MAX_LINEAGES = 300
DILUTION_FREQ = 8
N_BOTTLE = 100

class SimPool:
    def __init__(self, Ne, mu_mult, aneu_on, rng):
        self.Ne = Ne
        self.mu_mult = mu_mult
        self.aneu_on = aneu_on
        self.rng = rng
        self.Topt   = np.array([WT_TOPT])
        self.Tmax   = np.array([WT_TMAX])
        self.muopt  = np.array([WT_MUOPT])
        self.counts = np.array([Ne], dtype=np.int64)
        self.modules= [frozenset()]
        self.aneu   = [frozenset()]
        self.n = 1
        self._init_sgv()

    def _init_sgv(self):
        for _ in range(60):
            f = self.rng.beta(0.4, 10)
            nc = max(1, int(f*self.Ne))
            dT = self.rng.normal(0.08,0.18)
            self.Topt  = np.append(self.Topt,  WT_TOPT+dT)
            self.Tmax  = np.append(self.Tmax,  WT_TMAX+dT*1.3)
            self.muopt = np.append(self.muopt, max(WT_MUOPT*(1+self.rng.normal(0,0.03)),0.04))
            self.counts= np.append(self.counts, nc)
            self.modules.append(frozenset())
            self.aneu.append(frozenset())
            self.n += 1
        self.counts = (self.counts/self.counts.sum()*self.Ne).astype(np.int64)
        self.counts[0] = max(1, self.Ne - self.counts[1:].sum())

    def freqs(self):
        s = self.counts.sum()
        return self.counts/s if s>0 else np.ones(self.n)/self.n

    def step(self, T, gen):
        # Bottleneck
        if gen>0 and gen%DILUTION_FREQ==0:
            p = self.freqs()
            surv = self.rng.multinomial(N_BOTTLE, p).astype(np.int64)
            p2 = surv/N_BOTTLE
            self.counts = self.rng.multinomial(self.Ne, p2).astype(np.int64)

        # Selection + WF drift
        w = growth_rate_vec(T, self.Topt, self.Tmax, self.muopt)
        p = self.freqs()
        wbar = float(np.dot(w, p))
        if wbar > 1e-12:
            ps = np.where(np.isfinite(p*w/wbar), p*w/wbar, 0.0)
            ps = np.clip(ps, 0, None)
            s  = ps.sum()
            if s > 1e-12:
                ps /= s
            else:
                ps = np.ones(self.n)/self.n
        else:
            ps = np.ones(self.n)/self.n
        self.counts = self.rng.multinomial(self.Ne, ps).astype(np.int64)

        # Mutation
        U_base = 2e-10 * 1.2e7
        heat_f = 1.0 + 2.6*max(0,(T-30)/12)
        U_eff  = U_base * heat_f * self.mu_mult
        phase  = min(1.0, gen/300)

        new_To=[]; new_Tx=[]; new_mu=[]
        new_c=[]; new_mo=[]; new_an=[]

        for i in range(self.n):
            ni = int(self.counts[i])
            if ni == 0: continue

            # Beneficial SNP/module
            k_b = self.rng.poisson(max(ni*U_eff*0.10, 0))
            for _ in range(min(k_b,5)):
                s_raw = (self.rng.gamma(0.4,0.008) if self.rng.random()<0.65
                         else self.rng.gamma(1.5,0.03))
                wts = []
                for m in MODULE_NAMES:
                    dTo,dTx,dmu,we,wl = GENE_MODULES[m]
                    ww = (1-phase)*we + phase*wl
                    if m in self.modules[i]: ww *= 0.05
                    wts.append(max(ww,0.001))
                wts = np.array(wts); wts /= wts.sum()
                mi  = self.rng.choice(N_MODULES, p=wts)
                mod = MODULE_NAMES[mi]
                dTo,dTx,dmu,_,_ = GENE_MODULES[mod]
                W_bg = growth_rate(T, self.Topt[i], self.Tmax[i], self.muopt[i])
                s_ep = s_raw/(1+3.0*W_bg) + self.rng.normal(0,0.004)
                sc   = s_ep/0.05
                new_To.append(min(self.Topt[i]+dTo*sc*self.rng.uniform(0.5,1.5), WT_TOPT+8))
                new_Tx.append(min(self.Tmax[i]+dTx*sc*self.rng.uniform(0.5,1.5), WT_TMAX+10))
                new_mu.append(max(self.muopt[i]*(1+dmu*abs(sc)),0.04))
                new_mo.append(self.modules[i]|{mod})
                new_an.append(self.aneu[i])
                new_c.append(max(1,self.rng.poisson(1)))

            # Deleterious
            k_d = self.rng.poisson(max(ni*U_eff*0.35,0))
            for _ in range(min(k_d,4)):
                sd = -self.rng.gamma(0.5,0.015)
                new_To.append(self.Topt[i]+self.rng.normal(0,0.05))
                new_Tx.append(self.Tmax[i]+self.rng.normal(0,0.08))
                new_mu.append(max(self.muopt[i]*(1+sd*0.3),0.04))
                new_mo.append(self.modules[i])
                new_an.append(self.aneu[i])
                new_c.append(max(1,self.rng.poisson(1)))

            # Aneuploidy
            if self.aneu_on:
                k_a = self.rng.poisson(max(ni*5e-6*N_CHROM,0))
                for _ in range(min(k_a,2)):
                    ch = self.rng.integers(0,N_CHROM)
                    if ch in self.aneu[i]: continue
                    b = CHROM_BENEFIT.get(ch,0.005)*max(0,(T-32)/10)
                    c_cost = 0.002*CHROM_NGENES[ch]/500
                    if b-c_cost > -0.015:
                        new_To.append(self.Topt[i]+max(0,self.rng.normal(0.4,0.2))*(T>34))
                        new_Tx.append(self.Tmax[i]+max(0,self.rng.normal(0.6,0.3))*(T>34))
                        new_mu.append(max(self.muopt[i]*(1-c_cost),0.04))
                        new_mo.append(self.modules[i])
                        new_an.append(self.aneu[i]|{ch})
                        new_c.append(max(1,self.rng.poisson(1)))

        if new_To:
            self.Topt  = np.append(self.Topt,  new_To)
            self.Tmax  = np.append(self.Tmax,  new_Tx)
            self.muopt = np.append(self.muopt, new_mu)
            self.counts= np.append(self.counts, new_c)
            self.modules += new_mo
            self.aneu    += new_an
            self.n = len(self.modules)

        # Prune + cap
        alive = self.counts > 0
        if not np.all(alive):
            self.Topt=self.Topt[alive]; self.Tmax=self.Tmax[alive]
            self.muopt=self.muopt[alive]; self.counts=self.counts[alive]
            idx=np.where(alive)[0]
            self.modules=[self.modules[k] for k in idx]
            self.aneu=[self.aneu[k] for k in idx]
            self.n=len(self.modules)

        if self.n > MAX_LINEAGES:
            order=np.argsort(self.counts)[::-1][:MAX_LINEAGES]
            lost=self.counts[~np.isin(np.arange(self.n),order)].sum()
            self.Topt=self.Topt[order]; self.Tmax=self.Tmax[order]
            self.muopt=self.muopt[order]; self.counts=self.counts[order]
            self.counts[0]+=lost
            self.modules=[self.modules[k] for k in order]
            self.aneu=[self.aneu[k] for k in order]
            self.n=len(self.modules)

        # Metrics
        fr = self.freqs()
        w2 = growth_rate_vec(T, self.Topt, self.Tmax, self.muopt)
        wbar2 = float(np.dot(w2, fr))
        topt_m = float(np.dot(self.Topt, fr))
        tmax_m = float(np.dot(self.Tmax, fr))
        H = float(-np.sum(fr[fr>1e-8]*np.log(fr[fr>1e-8])))
        aneu_f = float(sum(f for a,f in zip(self.aneu,fr) if a))
        n_lin  = int(np.sum(fr>0.01))
        mfreq  = {m: float(sum(f for mo,f in zip(self.modules,fr) if m in mo))
                  for m in MODULE_NAMES}
        # Top lineage freqs for Muller (top 20)
        top_idx = np.argsort(fr)[::-1][:20]
        muller  = [float(fr[k]) for k in top_idx]
        noise   = float(np.random.normal(0, 0.006))

        return {
            "wbar":   round(wbar2 + noise, 5),
            "topt":   round(topt_m, 3),
            "tmax":   round(tmax_m, 3),
            "div":    round(H, 4),
            "aneu":   round(aneu_f, 4),
            "n_lin":  n_lin,
            "temp":   T,
            "muller": muller,
            "mfreq":  {k: round(v,4) for k,v in mfreq.items()},
        }


# ══════════════════════════════════════════════════════════
#  GLOBAL SIM STATE
# ══════════════════════════════════════════════════════════
sim_state = {
    "running":   False,
    "gen":       0,
    "config":    {},
    "history":   {r: {"gens":[],"wbar":[],"topt":[],"tmax":[],"div":[],
                       "aneu":[],"n_lin":[],"temp":[],
                       "muller":[],"mfreq":{m:[] for m in MODULE_NAMES}}
                  for r in range(4)},
    "pools":     [],
    "lock":      threading.Lock(),
    "clients":   [],          # SSE queues
}

def reset_sim(config):
    with sim_state["lock"]:
        sim_state["running"]  = False
        sim_state["gen"]      = 0
        sim_state["config"]   = config
        n_reps = config.get("n_reps", 2)
        Ne     = config.get("ne", 200_000)
        mu_m   = config.get("mu_mult", 1.0)
        aneu   = config.get("aneu_on", True)
        sim_state["history"] = {
            r: {"gens":[],"wbar":[],"topt":[],"tmax":[],"div":[],
                "aneu":[],"n_lin":[],"temp":[],
                "muller":[],"mfreq":{m:[] for m in MODULE_NAMES}}
            for r in range(n_reps)
        }
        seeds = [int(np.random.randint(0,2**31)) for _ in range(n_reps)]
        sim_state["pools"] = [
            SimPool(Ne, mu_m, aneu, np.random.default_rng(seeds[r]))
            for r in range(n_reps)
        ]

def run_sim_thread():
    config   = sim_state["config"]
    n_reps   = config.get("n_reps", 2)
    n_gens   = config.get("n_gens", 700)
    ramp     = config.get("ramp_speed", 1.0)
    schedule = build_temp_schedule(ramp)

    for gen in range(n_gens + 1):
        with sim_state["lock"]:
            if not sim_state["running"]:
                break
        T = current_temp_fn(gen, schedule)
        rep_data = []
        for r in range(n_reps):
            pool = sim_state["pools"][r]
            metrics = pool.step(T, gen)
            with sim_state["lock"]:
                h = sim_state["history"][r]
                h["gens"].append(gen)
                h["wbar"].append(metrics["wbar"])
                h["topt"].append(metrics["topt"])
                h["tmax"].append(metrics["tmax"])
                h["div"].append(metrics["div"])
                h["aneu"].append(metrics["aneu"])
                h["n_lin"].append(metrics["n_lin"])
                h["temp"].append(metrics["temp"])
                h["muller"].append(metrics["muller"])
                for m in MODULE_NAMES:
                    h["mfreq"][m].append(metrics["mfreq"].get(m,0))
            rep_data.append(metrics)

        with sim_state["lock"]:
            sim_state["gen"] = gen

        # Broadcast to all SSE clients every 2 gens
        if gen % 2 == 0:
            payload = json.dumps({
                "gen":      gen,
                "n_gens":   n_gens,
                "reps":     rep_data,
                "schedule": schedule,
                "history":  {str(r): sim_state["history"][r]
                             for r in range(n_reps)},
            }, default=float)
            dead = []
            for q in sim_state["clients"]:
                try:
                    q.put_nowait(payload)
                except:
                    dead.append(q)
            for q in dead:
                try: sim_state["clients"].remove(q)
                except: pass

        time.sleep(0.01)   # ~100 gen/s max

    with sim_state["lock"]:
        sim_state["running"] = False
    # Send done signal
    payload = json.dumps({"done": True, "gen": sim_state["gen"]})
    for q in sim_state["clients"]:
        try: q.put_nowait(payload)
        except: pass


# ══════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json()
    config = {
        "n_reps":     int(data.get("n_reps", 2)),
        "n_gens":     int(data.get("n_gens", 700)),
        "ne":         int(data.get("ne", 200_000)),
        "mu_mult":    float(data.get("mu_mult", 1.0)),
        "aneu_on":    bool(data.get("aneu_on", True)),
        "ramp_speed": float(data.get("ramp_speed", 1.0)),
    }
    reset_sim(config)
    with sim_state["lock"]:
        sim_state["running"] = True
    t = threading.Thread(target=run_sim_thread, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with sim_state["lock"]:
        sim_state["running"] = False
    return jsonify({"status": "stopped"})

@app.route("/api/status")
def api_status():
    with sim_state["lock"]:
        return jsonify({
            "running": sim_state["running"],
            "gen":     sim_state["gen"],
            "config":  sim_state["config"],
        })

@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=50)
    sim_state["clients"].append(q)
    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                    if json.loads(data).get("done"):
                        break
                except queue.Empty:
                    yield "data: {\"ping\":1}\n\n"
        finally:
            try: sim_state["clients"].remove(q)
            except: pass
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache",
                             "X-Accel-Buffering":"no"})


# ══════════════════════════════════════════════════════════
#  HTML TEMPLATE  (single-file, Chart.js, SSE)
# ══════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S. cerevisiae ALE Simulator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
:root {
  --bg:      #080c10;
  --surface: #0d1520;
  --panel:   #111d2e;
  --border:  #1a2d44;
  --glow:    #0ff;
  --text:    #c8dff0;
  --muted:   #4a6a88;
  --accent1: #00d4ff;
  --accent2: #00ff9d;
  --accent3: #ff6b35;
  --accent4: #bd93f9;
  --accent5: #f1fa8c;
  --danger:  #ff5555;
  --rep0: #00d4ff; --rep1: #00ff9d; --rep2: #ff6b35; --rep3: #bd93f9;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text);
             font-family: 'JetBrains Mono', 'Fira Code', monospace; overflow-x: hidden; }

/* ── scanline overlay ── */
body::before {
  content:''; position:fixed; top:0; left:0; width:100%; height:100%;
  background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.08) 2px,rgba(0,0,0,.08) 4px);
  pointer-events:none; z-index:9999;
}

/* ── header ── */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 28px; border-bottom: 1px solid var(--border);
  background: linear-gradient(90deg,rgba(0,212,255,.06),rgba(0,255,157,.04));
  position: sticky; top:0; z-index:100; backdrop-filter: blur(8px);
}
.logo { display:flex; align-items:center; gap:12px; }
.logo-icon { width:36px; height:36px; }
.logo-text { font-size:1.1rem; font-weight:700; letter-spacing:.08em;
             background: linear-gradient(90deg,var(--accent1),var(--accent2));
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.logo-sub  { font-size:.65rem; color:var(--muted); letter-spacing:.15em; margin-top:1px; }
.status-bar { display:flex; align-items:center; gap:20px; font-size:.75rem; }
.status-pill { display:flex; align-items:center; gap:6px; padding:4px 12px;
               border-radius:20px; border:1px solid var(--border); background:var(--surface); }
.pulse { width:8px; height:8px; border-radius:50%; background:var(--muted); }
.pulse.running { background:var(--accent2); box-shadow:0 0 8px var(--accent2);
                 animation: blink 1s infinite; }
.pulse.done { background:var(--accent1); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── layout ── */
.app { display:grid; grid-template-columns:280px 1fr; height:calc(100vh - 61px); }

/* ── sidebar ── */
aside {
  background: var(--surface); border-right: 1px solid var(--border);
  overflow-y: auto; padding: 20px 16px;
  display: flex; flex-direction: column; gap: 16px;
}
.section-title {
  font-size:.65rem; letter-spacing:.2em; color:var(--muted);
  text-transform:uppercase; padding-bottom:8px;
  border-bottom:1px solid var(--border); margin-bottom:4px;
}
.control-group { display:flex; flex-direction:column; gap:12px; }
.ctrl { display:flex; flex-direction:column; gap:5px; }
.ctrl label { font-size:.72rem; color:var(--muted); letter-spacing:.05em; }
.ctrl .val { font-size:.85rem; color:var(--accent1); font-weight:700; }

input[type=range] {
  -webkit-appearance:none; width:100%; height:4px;
  background: linear-gradient(90deg,var(--accent1),var(--accent2));
  border-radius:2px; outline:none; cursor:pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none; width:14px; height:14px; border-radius:50%;
  background:var(--text); border:2px solid var(--accent1);
  box-shadow:0 0 6px var(--accent1);
}

.toggle-row { display:flex; align-items:center; justify-content:space-between; }
.toggle-label { font-size:.78rem; color:var(--text); }
.toggle { position:relative; width:44px; height:22px; cursor:pointer; }
.toggle input { opacity:0; width:0; height:0; }
.slider-t {
  position:absolute; inset:0; border-radius:22px;
  background:var(--border); transition:.3s;
}
.slider-t:before {
  content:''; position:absolute; width:16px; height:16px; left:3px; bottom:3px;
  border-radius:50%; background:var(--muted); transition:.3s;
}
.toggle input:checked + .slider-t { background:var(--accent1); }
.toggle input:checked + .slider-t:before { transform:translateX(22px); background:#fff; }

.btn {
  width:100%; padding:11px; border:none; border-radius:6px; cursor:pointer;
  font-family:inherit; font-size:.85rem; font-weight:700; letter-spacing:.08em;
  transition:.2s; position:relative; overflow:hidden;
}
.btn-run {
  background: linear-gradient(135deg,var(--accent1),var(--accent2));
  color:#000;
}
.btn-run:hover { filter:brightness(1.15); box-shadow:0 0 20px rgba(0,212,255,.4); }
.btn-stop {
  background:transparent; color:var(--danger); border:1px solid var(--danger);
}
.btn-stop:hover { background:rgba(255,85,85,.12); }
.btn:disabled { opacity:.4; cursor:not-allowed; }

.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.stat-card {
  background:var(--panel); border:1px solid var(--border); border-radius:6px;
  padding:8px 10px;
}
.stat-name { font-size:.6rem; color:var(--muted); letter-spacing:.1em; text-transform:uppercase; }
.stat-val  { font-size:1.0rem; font-weight:700; margin-top:2px; }

.progress-bar-wrap { height:3px; background:var(--border); border-radius:2px; overflow:hidden; }
.progress-bar { height:100%; background:linear-gradient(90deg,var(--accent1),var(--accent2));
                border-radius:2px; transition:width .3s; width:0%; }

/* ── main content ── */
main { overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:14px; }

.chart-grid-top { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.chart-grid-mid { display:grid; grid-template-columns:2fr 1fr 1fr; gap:14px; }
.chart-grid-bot { display:grid; grid-template-columns:1fr 1fr; gap:14px; }

.card {
  background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:14px; position:relative; overflow:hidden;
}
.card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--accent1),transparent);
  opacity:.5;
}
.card-title {
  font-size:.65rem; letter-spacing:.15em; color:var(--muted);
  text-transform:uppercase; margin-bottom:10px;
}
.chart-wrap { position:relative; width:100%; }

/* ── muller canvas ── */
#mullerCanvas { width:100%; border-radius:4px; image-rendering:pixelated; }

/* ── module bars ── */
.module-bars { display:flex; flex-direction:column; gap:4px; margin-top:4px; }
.mod-row { display:flex; align-items:center; gap:8px; font-size:.63rem; }
.mod-name { width:90px; color:var(--muted); white-space:nowrap; overflow:hidden;
            text-overflow:ellipsis; flex-shrink:0; }
.mod-bar-bg { flex:1; height:6px; background:var(--border); border-radius:3px; overflow:hidden; }
.mod-bar-fill { height:100%; border-radius:3px; transition:width .4s; }
.mod-val { width:34px; text-align:right; color:var(--accent1); font-weight:700; }

/* ── temp indicator ── */
.temp-badge {
  display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
  border-radius:12px; font-size:.72rem; font-weight:700;
  background:rgba(255,107,53,.15); border:1px solid var(--accent3); color:var(--accent3);
}

/* ── rep legend ── */
.rep-legend { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
.rep-dot { display:flex; align-items:center; gap:4px; font-size:.65rem; color:var(--muted); }
.dot { width:8px; height:8px; border-radius:50%; }

/* ── scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
  <div class="logo">
    <svg class="logo-icon" viewBox="0 0 36 36" fill="none">
      <circle cx="18" cy="18" r="17" stroke="#00d4ff" stroke-width="1.5" opacity=".4"/>
      <circle cx="18" cy="18" r="6" fill="none" stroke="#00ff9d" stroke-width="1.5"/>
      <circle cx="18" cy="11" r="2.5" fill="#00d4ff"/>
      <circle cx="24" cy="21" r="2.5" fill="#ff6b35"/>
      <circle cx="12" cy="21" r="2.5" fill="#bd93f9"/>
      <line x1="18" y1="13" x2="18" y2="16" stroke="#00d4ff" stroke-width="1"/>
      <line x1="22" y1="20" x2="20" y2="19" stroke="#ff6b35" stroke-width="1"/>
      <line x1="14" y1="20" x2="16" y2="19" stroke="#bd93f9" stroke-width="1"/>
    </svg>
    <div>
      <div class="logo-text">S. cerevisiae ALE Simulator</div>
      <div class="logo-sub">Adaptive Laboratory Evolution · Live Dashboard</div>
    </div>
  </div>
  <div class="status-bar">
    <div class="status-pill">
      <div class="pulse" id="statusPulse"></div>
      <span id="statusText">Ready</span>
    </div>
    <div class="status-pill">
      Gen <strong id="genCounter" style="color:var(--accent1);margin-left:4px">0</strong>
      <span style="color:var(--muted);margin:0 4px">/</span>
      <span id="genTotal">700</span>
    </div>
    <div class="status-pill" id="tempBadge">
      <span style="color:var(--muted)">T=</span>
      <strong id="tempVal" style="color:var(--accent3)">30°C</strong>
    </div>
  </div>
</header>

<!-- ── MAIN LAYOUT ── -->
<div class="app">

<!-- SIDEBAR -->
<aside>
  <div>
    <div class="section-title">Simulation Config</div>
    <div class="control-group">

      <div class="ctrl">
        <label>Generations <span class="val" id="vGens">700</span></label>
        <input type="range" id="sGens" min="200" max="1000" step="50" value="700"
               oninput="document.getElementById('vGens').textContent=this.value">
      </div>

      <div class="ctrl">
        <label>Population Size (Ne) <span class="val" id="vNe">200k</span></label>
        <input type="range" id="sNe" min="1" max="5" step="1" value="2"
               oninput="updateNeLabel(this.value)">
      </div>

      <div class="ctrl">
        <label>Replicates <span class="val" id="vReps">2</span></label>
        <input type="range" id="sReps" min="1" max="4" step="1" value="2"
               oninput="document.getElementById('vReps').textContent=this.value">
      </div>

      <div class="ctrl">
        <label>Temp Ramp Speed <span class="val" id="vRamp">1.0×</span></label>
        <input type="range" id="sRamp" min="0.5" max="3.0" step="0.25" value="1.0"
               oninput="document.getElementById('vRamp').textContent=parseFloat(this.value).toFixed(2)+'×'">
      </div>

      <div class="ctrl">
        <label>Mutation Rate Multiplier <span class="val" id="vMu">1.0×</span></label>
        <input type="range" id="sMu" min="0.1" max="5.0" step="0.1" value="1.0"
               oninput="document.getElementById('vMu').textContent=parseFloat(this.value).toFixed(1)+'×'">
      </div>

      <div class="toggle-row">
        <span class="toggle-label">Aneuploidy Events</span>
        <label class="toggle">
          <input type="checkbox" id="cbAneu" checked>
          <span class="slider-t"></span>
        </label>
      </div>

    </div>
  </div>

  <div>
    <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar"></div></div>
  </div>

  <button class="btn btn-run" id="btnRun" onclick="startSim()">▶ RUN SIMULATION</button>
  <button class="btn btn-stop" id="btnStop" onclick="stopSim()" disabled>■ STOP</button>

  <!-- Live stats -->
  <div>
    <div class="section-title">Live Statistics</div>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-name">Mean Fitness</div>
        <div class="stat-val" id="sv-wbar" style="color:var(--accent2)">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-name">Mean Topt</div>
        <div class="stat-val" id="sv-topt" style="color:var(--accent1)">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-name">Diversity H</div>
        <div class="stat-val" id="sv-div" style="color:var(--accent4)">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-name">Lineages >1%</div>
        <div class="stat-val" id="sv-lin" style="color:var(--accent5)">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-name">Aneuploidy</div>
        <div class="stat-val" id="sv-aneu" style="color:var(--accent4)">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-name">Mean Tmax</div>
        <div class="stat-val" id="sv-tmax" style="color:var(--accent3)">—</div>
      </div>
    </div>
  </div>

  <!-- Gene module bars -->
  <div>
    <div class="section-title">Gene Module Freq (mean)</div>
    <div class="module-bars" id="moduleBars"></div>
  </div>

</aside>

<!-- MAIN -->
<main>
  <!-- rep legend -->
  <div class="rep-legend" id="repLegend"></div>

  <div class="chart-grid-top">
    <div class="card">
      <div class="card-title">Mean Fitness — All Replicates (noisy)</div>
      <div class="chart-wrap" style="height:180px">
        <canvas id="cFitness"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Topt & Tmax Evolution</div>
      <div class="chart-wrap" style="height:180px">
        <canvas id="cTopt"></canvas>
      </div>
    </div>
  </div>

  <div class="chart-grid-mid">
    <div class="card">
      <div class="card-title">Muller Plot — Clonal Interference (Rep 1)</div>
      <div class="chart-wrap" style="height:160px">
        <canvas id="mullerCanvas"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Clonal Diversity (Shannon H)</div>
      <div class="chart-wrap" style="height:160px">
        <canvas id="cDiv"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Aneuploidy Waves</div>
      <div class="chart-wrap" style="height:160px">
        <canvas id="cAneu"></canvas>
      </div>
    </div>
  </div>

  <div class="chart-grid-bot">
    <div class="card">
      <div class="card-title">Top Gene Module Sweeps (mean across reps)</div>
      <div class="chart-wrap" style="height:170px">
        <canvas id="cModules"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Competing Lineages (>1% freq)</div>
      <div class="chart-wrap" style="height:170px">
        <canvas id="cLineages"></canvas>
      </div>
    </div>
  </div>
</main>
</div>

<script>
// ── Ne label mapping ──
const neMap = {1:50000,2:200000,3:500000,4:1000000,5:3000000};
const neLabel = {1:'50k',2:'200k',3:'500k',4:'1M',5:'3M'};
function updateNeLabel(v){
  document.getElementById('vNe').textContent = neLabel[v] || v+'k';
}

// ── Colours ──
const REP_COLS = ['#00d4ff','#00ff9d','#ff6b35','#bd93f9'];
const MOD_COLS = ['#00d4ff','#00ff9d','#ff6b35','#bd93f9','#f1fa8c',
                  '#ff79c6','#50fa7b','#8be9fd','#ffb86c','#ff5555'];
const MODULE_NAMES = ["ERG3_LOF","ETC_LOF","CDC25_reg","HSF1_pt","SKN7_pt",
                      "HSP104","TPS1_TPS2","HOG1_reg","RAS2_IRA2","Cell_wall"];

// ── Chart.js defaults ──
Chart.defaults.color = '#4a6a88';
Chart.defaults.borderColor = '#1a2d44';
Chart.defaults.font.family = "'JetBrains Mono','Fira Code',monospace";
Chart.defaults.font.size = 10;

function makeChart(id, type, datasets, opts={}) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type, data: { labels: [], datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: true,
        backgroundColor:'#0d1520', borderColor:'#1a2d44', borderWidth:1,
        titleColor:'#c8dff0', bodyColor:'#4a6a88' }},
      scales: {
        x: { grid:{color:'#1a2d44'}, ticks:{maxTicksLimit:8, color:'#4a6a88'} },
        y: { grid:{color:'#1a2d44'}, ticks:{color:'#4a6a88'}, ...opts.y },
      },
      elements: { point:{radius:0}, line:{tension:0.2, borderWidth:1.5} },
      ...opts.extra,
    }
  });
}

// ── Initialise charts ──
const chartFitness  = makeChart('cFitness','line',[],[]);
const chartTopt     = makeChart('cTopt','line',[],[]);
const chartDiv      = makeChart('cDiv','line',[],[]);
const chartAneu     = makeChart('cAneu','line',[],[]);
const chartModules  = makeChart('cModules','line',[],{});
const chartLineages = makeChart('cLineages','line',[],[]);

// ── Muller canvas ──
const mullerEl  = document.getElementById('mullerCanvas');
const mullerCtx = mullerEl.getContext('2d');
const MULLER_W  = 600;
const MULLER_H  = 120;
mullerEl.width  = MULLER_W;
mullerEl.height = MULLER_H;
const MULLER_COLS = [
  '#00d4ff','#00ff9d','#ff6b35','#bd93f9','#f1fa8c',
  '#ff79c6','#50fa7b','#8be9fd','#ffb86c','#ff5555',
  '#6272a4','#44475a','#8be9fd','#ffb86c','#50fa7b',
  '#ff79c6','#bd93f9','#f1fa8c','#ff6b35','#00ff9d',
];

let mullerHistory = []; // array of freq arrays per gen
let mullerXOffset = 0;

function drawMuller(freqHistory, nGens) {
  if (!freqHistory || freqHistory.length === 0) return;
  mullerEl.width = mullerEl.offsetWidth || MULLER_W;
  mullerEl.height = MULLER_H;
  const W = mullerEl.width, H = mullerEl.height;
  mullerCtx.clearRect(0, 0, W, H);
  mullerCtx.fillStyle = '#080c10';
  mullerCtx.fillRect(0, 0, W, H);

  const nT = freqHistory.length;
  const nL = freqHistory[0] ? freqHistory[0].length : 0;
  if (nL === 0) return;

  for (let t = 0; t < nT; t++) {
    const x  = Math.floor(t / nGens * W);
    const nx = Math.floor((t+1) / nGens * W);
    const freqs = freqHistory[t] || [];
    let y = 0;
    for (let l = 0; l < freqs.length; l++) {
      const h = Math.round(freqs[l] * H);
      mullerCtx.fillStyle = MULLER_COLS[l % MULLER_COLS.length];
      mullerCtx.fillRect(x, y, Math.max(nx-x,1), h);
      y += h;
    }
    // fill remainder
    if (y < H) {
      mullerCtx.fillStyle = '#0d1520';
      mullerCtx.fillRect(x, y, Math.max(nx-x,1), H-y);
    }
  }
}

// ── Module bars ──
function renderModuleBars(mfreqMean) {
  const container = document.getElementById('moduleBars');
  container.innerHTML = '';
  MODULE_NAMES.forEach((m, i) => {
    const v = mfreqMean[m] || 0;
    const pct = Math.round(v * 100);
    const row = document.createElement('div');
    row.className = 'mod-row';
    row.innerHTML = `
      <span class="mod-name" title="${m}">${m.replace(/_/g,' ')}</span>
      <div class="mod-bar-bg">
        <div class="mod-bar-fill" style="width:${pct}%;background:${MOD_COLS[i%MOD_COLS.length]}"></div>
      </div>
      <span class="mod-val">${pct}%</span>`;
    container.appendChild(row);
  });
}

// ── Rep legend ──
function updateLegend(nReps) {
  const el = document.getElementById('repLegend');
  el.innerHTML = '';
  for (let r = 0; r < nReps; r++) {
    const d = document.createElement('div');
    d.className = 'rep-dot';
    d.innerHTML = `<div class="dot" style="background:${REP_COLS[r]}"></div>Replicate ${r+1}`;
    el.appendChild(d);
  }
}

// ── Chart update helpers ──
function syncLabels(chart, labels) {
  chart.data.labels = labels;
}

function ensureDatasets(chart, n, makeDs) {
  while (chart.data.datasets.length < n) {
    chart.data.datasets.push(makeDs(chart.data.datasets.length));
  }
}

function setData(chart, dsIdx, data) {
  if (chart.data.datasets[dsIdx]) chart.data.datasets[dsIdx].data = data;
}

// ── SSE state ──
let evtSource = null;
let simRunning = false;

// ── Update all charts ──
function updateCharts(payload) {
  const { gen, n_gens, reps, history, schedule } = payload;
  if (!history) return;

  const nReps = Object.keys(history).length;
  updateLegend(nReps);

  const gens0 = history['0'] ? history['0'].gens : [];

  // --- Fitness chart ---
  syncLabels(chartFitness, gens0);
  ensureDatasets(chartFitness, nReps, r => ({
    label: `Rep ${r+1}`, data: [], borderColor: REP_COLS[r],
    backgroundColor:'transparent', borderWidth: 1.3,
  }));
  for (let r = 0; r < nReps; r++) {
    const h = history[String(r)];
    setData(chartFitness, r, h ? h.wbar : []);
  }
  chartFitness.update('none');

  // --- Topt/Tmax chart ---
  syncLabels(chartTopt, gens0);
  const needTDS = nReps * 2;
  ensureDatasets(chartTopt, needTDS, i => {
    const r = Math.floor(i/2), isMax = i%2===1;
    return { data:[], borderColor: REP_COLS[r],
             borderDash: isMax ? [4,2] : [], borderWidth:1.4,
             backgroundColor:'transparent' };
  });
  for (let r = 0; r < nReps; r++) {
    const h = history[String(r)];
    setData(chartTopt, r*2,   h ? h.topt : []);
    setData(chartTopt, r*2+1, h ? h.tmax : []);
  }
  chartTopt.update('none');

  // --- Diversity chart ---
  syncLabels(chartDiv, gens0);
  ensureDatasets(chartDiv, nReps, r => ({
    data:[], borderColor: REP_COLS[r], backgroundColor:'transparent', borderWidth:1.3,
  }));
  for (let r = 0; r < nReps; r++) {
    setData(chartDiv, r, history[String(r)] ? history[String(r)].div : []);
  }
  chartDiv.update('none');

  // --- Aneuploidy chart ---
  syncLabels(chartAneu, gens0);
  ensureDatasets(chartAneu, nReps, r => ({
    data:[], borderColor: REP_COLS[r], backgroundColor: REP_COLS[r]+'22',
    fill:true, borderWidth:1.3,
  }));
  for (let r = 0; r < nReps; r++) {
    setData(chartAneu, r, history[String(r)] ? history[String(r)].aneu : []);
  }
  chartAneu.update('none');

  // --- Module sweeps (mean) ---
  const TOP_MODS = ["ERG3_LOF","ETC_LOF","CDC25_reg","HSF1_pt","HSP104","TPS1_TPS2"];
  syncLabels(chartModules, gens0);
  ensureDatasets(chartModules, TOP_MODS.length, i => ({
    label: TOP_MODS[i], data:[], borderColor: MOD_COLS[i],
    backgroundColor:'transparent', borderWidth:1.5,
  }));
  TOP_MODS.forEach((m, i) => {
    const nPts = gens0.length;
    const mean = new Array(nPts).fill(0);
    for (let r = 0; r < nReps; r++) {
      const h = history[String(r)];
      if (h && h.mfreq && h.mfreq[m]) {
        h.mfreq[m].forEach((v, t) => { if(t < nPts) mean[t] += v / nReps; });
      }
    }
    setData(chartModules, i, mean);
  });
  chartModules.update('none');

  // --- Lineages chart ---
  syncLabels(chartLineages, gens0);
  ensureDatasets(chartLineages, nReps, r => ({
    data:[], borderColor: REP_COLS[r], backgroundColor:'transparent', borderWidth:1.3,
  }));
  for (let r = 0; r < nReps; r++) {
    setData(chartLineages, r, history[String(r)] ? history[String(r)].n_lin : []);
  }
  chartLineages.update('none');

  // --- Muller ---
  const h0 = history['0'];
  if (h0 && h0.muller) {
    drawMuller(h0.muller, n_gens);
  }

  // --- Module bars sidebar ---
  const mfreqMean = {};
  MODULE_NAMES.forEach(m => {
    let s = 0, cnt = 0;
    for (let r = 0; r < nReps; r++) {
      const h = history[String(r)];
      if (h && h.mfreq && h.mfreq[m] && h.mfreq[m].length > 0) {
        s += h.mfreq[m][h.mfreq[m].length-1];
        cnt++;
      }
    }
    mfreqMean[m] = cnt > 0 ? s/cnt : 0;
  });
  renderModuleBars(mfreqMean);

  // --- Live stats (last rep 0 value) ---
  if (h0 && h0.wbar.length > 0) {
    const last = h0.wbar.length - 1;
    document.getElementById('sv-wbar').textContent = h0.wbar[last].toFixed(4);
    document.getElementById('sv-topt').textContent = h0.topt[last].toFixed(2)+'°C';
    document.getElementById('sv-tmax').textContent = h0.tmax[last].toFixed(2)+'°C';
    document.getElementById('sv-div').textContent  = h0.div[last].toFixed(3);
    document.getElementById('sv-lin').textContent  = h0.n_lin[last];
    document.getElementById('sv-aneu').textContent = (h0.aneu[last]*100).toFixed(1)+'%';
  }

  // --- Header counters ---
  document.getElementById('genCounter').textContent = gen;
  document.getElementById('tempVal').textContent    = (reps[0]?.temp || 30)+'°C';
  const pct = Math.round(gen / n_gens * 100);
  document.getElementById('progressBar').style.width = pct+'%';
}

// ── Start simulation ──
function startSim() {
  if (evtSource) { evtSource.close(); evtSource = null; }

  const config = {
    n_reps:     parseInt(document.getElementById('sReps').value),
    n_gens:     parseInt(document.getElementById('sGens').value),
    ne:         neMap[parseInt(document.getElementById('sNe').value)] || 200000,
    mu_mult:    parseFloat(document.getElementById('sMu').value),
    aneu_on:    document.getElementById('cbAneu').checked,
    ramp_speed: parseFloat(document.getElementById('sRamp').value),
  };

  document.getElementById('genTotal').textContent = config.n_gens;

  // Reset charts
  [chartFitness, chartTopt, chartDiv, chartAneu,
   chartModules, chartLineages].forEach(c => {
    c.data.labels = []; c.data.datasets = []; c.update('none');
  });
  mullerCtx.clearRect(0,0,mullerEl.width,mullerEl.height);
  document.getElementById('moduleBars').innerHTML = '';
  document.getElementById('progressBar').style.width = '0%';

  fetch('/api/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(config)
  }).then(r => r.json()).then(() => {
    simRunning = true;
    document.getElementById('btnRun').disabled  = true;
    document.getElementById('btnStop').disabled = false;
    const p = document.getElementById('statusPulse');
    p.className = 'pulse running';
    document.getElementById('statusText').textContent = 'Running…';

    evtSource = new EventSource('/stream');
    evtSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.ping) return;
      if (data.done) {
        simRunning = false;
        document.getElementById('btnRun').disabled  = false;
        document.getElementById('btnStop').disabled = true;
        p.className = 'pulse done';
        document.getElementById('statusText').textContent = 'Complete';
        evtSource.close();
        return;
      }
      updateCharts(data);
    };
    evtSource.onerror = () => {
      document.getElementById('statusText').textContent = 'Stream error';
    };
  });
}

function stopSim() {
  fetch('/api/stop',{method:'POST'});
  simRunning = false;
  document.getElementById('btnRun').disabled  = false;
  document.getElementById('btnStop').disabled = true;
  document.getElementById('statusPulse').className = 'pulse';
  document.getElementById('statusText').textContent = 'Stopped';
  if (evtSource) { evtSource.close(); evtSource = null; }
}

// ── Init sidebar module bars ──
renderModuleBars({});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*55}")
    print(f"  S. cerevisiae ALE Live Dashboard")
    print(f"  Open your browser at:  http://localhost:{port}")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
