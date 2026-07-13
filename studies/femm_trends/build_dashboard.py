"""Generate a single self-contained interactive HTML dashboard from the FEMM study results
(no server, no build step, no external deps -- matches tools/web/optimizer_dashboard.html).
Embeds every design + precomputed OLS interaction matrix, then renders filterable SVG charts
in vanilla JS. Run: python build_dashboard.py  ->  study/dashboard.html"""
import json, sys
import numpy as np
import study_viz as V

SUB = sys.argv[1] if len(sys.argv) > 1 else None
rows = V.load(SUB)

# pair femm + analytic per (cell, driver)
key = lambda x: (x["cell_id"], x["bus_voltage_v"], x["driver_bipolar"], x["pump_envelope"], x["i_max_a"])
fe = {key(x): x for x in rows if x["force_law"] == "femm" and x.get("exit_speed_mps") is not None}
an = {key(x): x["exit_speed_mps"] for x in rows if x["force_law"] == "analytic"}
designs = []
for k, x in fe.items():
    designs.append({**{kk: x[kk] for kk in V.ALL}, "cell_id": x["cell_id"],
                    "femm": x["exit_speed_mps"], "analytic": an.get(k)})

# columnar (categoricals coded 0/1 already via numeric); keep raw for labels
cols = {}
for k in V.ALL:
    cols[k] = [V.numeric(d, k) for d in designs]
cols["femm"] = [d["femm"] for d in designs]
cols["analytic"] = [(d["analytic"] if d["analytic"] is not None else None) for d in designs]
cols["cell_id"] = [d["cell_id"] for d in designs]

# OLS standardized interaction matrix (femm)
r, y, X = V.arrays(rows, "femm")
beta = V.std_ols(y, X)
n = len(V.ALL); M = [[None]*n for _ in range(n)]
for i, a in enumerate(V.ALL):
    M[i][i] = round(beta[a], 3)
    for j, b in enumerate(V.ALL):
        if j > i:
            M[i][j] = M[j][i] = round(beta.get((a, b), beta.get((b, a))), 3)

DATA = {
    "n": len(designs), "cols": cols,
    "knobs": V.ALL,
    "factors": {k: sorted(set(cols[k])) for k in V.ALL},
    "labels": V.LABEL, "short": {k: s for k, s in zip(V.ALL,
        ["V","Imax","N","Lcoil","Twind","Rmag","Lmag","Br","Bipolar","Square"])},
    "mm": list(V.MM), "beta": M,
    "cont": V.CONT, "cat": list(V.BOOL01),
}

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMAC FEMM design-trends dashboard</title>
<style>
:root{--bg:#f7f8fa;--panel:#fff;--ink:#1c2530;--mut:#667085;--line:#e4e7ec;--accent:#4C72B0;
 --blue:#4C72B0;--orange:#DD8452;--green:#55A868;--red:#C44E52;}
@media (prefers-color-scheme:dark){:root{--bg:#12151a;--panel:#1b1f27;--ink:#e6eaf0;--mut:#94a0b0;--line:#2b313c;}}
*{box-sizing:border-box}body{margin:0;font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
header h1{margin:0;font-size:16px}header p{margin:3px 0 0;color:var(--mut);font-size:12px}
.wrap{display:flex;align-items:flex-start;gap:16px;padding:16px}
.side{flex:0 0 240px;position:sticky;top:70px}
.main{flex:1;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.card h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile .v{font-size:22px;font-weight:700}.tile .l{color:var(--mut);font-size:11px;margin-top:2px}
.knob{margin-bottom:9px}.knob b{font-size:11px;color:var(--mut);display:block;margin-bottom:3px}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:11px;padding:2px 7px;border:1px solid var(--line);border-radius:12px;cursor:pointer;user-select:none;background:transparent;color:var(--mut)}
.chip.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.row{display:flex;gap:16px;flex-wrap:wrap}.row>div{flex:1;min-width:300px}
select,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--ink)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:5px 11px;cursor:pointer;background:transparent}
.seg button.on{background:var(--accent);color:#fff}
.ctl{display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap;color:var(--mut);font-size:12px}
svg text{fill:var(--ink)}.gl{stroke:var(--line)}.reset{cursor:pointer;color:var(--accent);font-size:11px}
small{color:var(--mut)}
</style></head><body>
<header><h1>EMAC linear stepper — real-FEMM design-trends dashboard</h1>
<p id="sub"></p></header>
<div class="wrap">
 <aside class="side card"><h2>Filters <span class="reset" onclick="resetF()">reset</span></h2><div id="filters"></div></aside>
 <div class="main">
  <div class="tiles" id="tiles"></div>
  <div class="card"><h2>Main effects — mean exit speed vs each knob</h2>
   <div class="ctl">force law: <span class="seg" id="lawseg"></span> <small>(filtered subset; error caps = ±SEM)</small></div>
   <div class="grid" id="maincurves"></div></div>
  <div class="row">
   <div class="card"><h2>Interaction strength (standardized OLS β, FEMM)</h2><div id="heat"></div>
     <small>diagonal = main effect; off-diagonal = 2-way interaction. Red raises speed, blue lowers. (whole dataset)</small></div>
   <div class="card"><h2>Moderation — knob split by a moderator</h2>
     <div class="ctl">knob <select id="modknob"></select> by <select id="modby"></select></div>
     <div id="moder"></div></div>
  </div>
  <div class="card"><h2>Design map — mean exit speed over two knobs</h2>
   <div class="ctl">X <select id="mapx"></select> Y <select id="mapy"></select>
     &nbsp; drive: <span class="seg" id="polseg"></span></div>
   <div id="map"></div></div>
  <div class="row">
   <div class="card"><h2>Analytic model vs real FEMM</h2><div id="scatter"></div>
     <small id="scatlbl"></small></div>
   <div class="card"><h2>Feasibility — stalled designs (&lt;0.5 m/s)</h2>
     <div class="ctl">by <select id="feasknob"></select></div><div id="feas"></div></div>
  </div>
  <div class="card"><small>Self-contained · <span id="ndesign"></span> designs · generated from study/results · no server/build/deps.</small></div>
 </div>
</div>
<script>
const D=__DATA__;
const NS="http://www.w3.org/2000/svg";
const C={blue:"#4C72B0",orange:"#DD8452",green:"#55A868",red:"#C44E52",purple:"#8172B3",grey:"#98a2b3"};
const isMM=k=>D.mm.includes(k);
function fmtLv(k,v){ if(k==="driver_bipolar")return v?"bipolar":"unipolar";
 if(k==="pump_envelope")return v?"square":"rcos"; return isMM(k)?(v*1000).toFixed(v*1000<10?0:0)+"":( +(+v).toPrecision(3)+"" );}
function shortLv(k,v){ if(k==="driver_bipolar")return v?"bi":"uni"; if(k==="pump_envelope")return v?"sq":"rc";
 return isMM(k)?(v*1000)+"":(""+ (+(+v).toPrecision(3))); }
let law="femm";
let pol="all";
const F={}; D.knobs.forEach(k=>F[k]=new Set(D.factors[k]));

function passes(i){ for(const k of D.knobs){ if(!F[k].has(D.cols[k][i]))return false; } return true; }
function idx(){ const a=[]; for(let i=0;i<D.n;i++) if(passes(i))a.push(i); return a; }

// ---------- svg helpers ----------
function E(t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;return e;}
function svg(w,h){const s=E("svg",{viewBox:`0 0 ${w} ${h}`,width:"100%",height:h});return s;}
function clear(id){const e=document.getElementById(id);e.innerHTML="";return e;}
function lerp(a,b,t){return a+(b-a)*t;}
function hex(c){return[parseInt(c.substr(1,2),16),parseInt(c.substr(3,2),16),parseInt(c.substr(5,2),16)];}
function mix(c1,c2,t){const a=hex(c1),b=hex(c2);return`rgb(${a.map((x,i)=>Math.round(lerp(x,b[i],t))).join(",")})`;}
function diverge(v,m){const t=Math.max(-1,Math.min(1,v/m));return t<0?mix("#f7f9fc","#3b4cc0",-t):mix("#f7f9fc","#b40426",t);}
function seqcol(t){t=Math.max(0,Math.min(1,t));return t<.5?mix("#440154","#21908c",t*2):mix("#21908c","#fde725",(t-.5)*2);}

function mean(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:NaN;}
function sem(a){if(a.length<2)return 0;const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1))/Math.sqrt(a.length);}

// mean speed of `col` grouped by knob levels over index set `ix`, optional extra filter fn
function byLevel(ix,knob,col,extra){const lv=D.factors[knob],out=[];for(const v of lv){const vals=[];
 for(const i of ix){if(D.cols[knob][i]===v&&(!extra||extra(i))){const s=D.cols[col][i];if(s!=null)vals.push(s);}}
 out.push({v,mean:mean(vals),sem:sem(vals),n:vals.length});}return out;}

// ---------- charts ----------
function lineChart(el,knob,series,{w=300,h=150,ylab=""}={}){
 const s=svg(w,h),ml=42,mb=26,mt=8,mr=8;const lv=D.factors[knob];
 const xs=v=>ml+(lv.length<2?.5:lv.indexOf(v)/(lv.length-1))*(w-ml-mr);
 let ymin=1e9,ymax=-1e9;series.forEach(se=>se.pts.forEach(p=>{if(!isNaN(p.m)){ymin=Math.min(ymin,p.m-(p.s||0));ymax=Math.max(ymax,p.m+(p.s||0));}}));
 if(ymin>ymax){ymin=0;ymax=1;}const pad=(ymax-ymin)*.12||.5;ymin-=pad;ymax+=pad;
 const ys=v=>h-mb-(v-ymin)/(ymax-ymin)*(h-mb-mt);
 for(let g=0;g<=3;g++){const yv=ymin+(ymax-ymin)*g/3;s.appendChild(E("line",{x1:ml,x2:w-mr,y1:ys(yv),y2:ys(yv),class:"gl","stroke-width":1}));
  s.appendChild(E("text",{x:ml-5,y:ys(yv)+3,"text-anchor":"end","font-size":9,fill:"var(--mut)"},yv.toFixed(1)));}
 lv.forEach(v=>s.appendChild(E("text",{x:xs(v),y:h-mb+12,"text-anchor":"middle","font-size":9,fill:"var(--mut)"},shortLv(knob,v))));
 series.forEach(se=>{let d="";se.pts.forEach((p,i)=>{if(isNaN(p.m))return;d+=(d?"L":"M")+xs(p.v)+" "+ys(p.m)+" ";
   if(p.s){s.appendChild(E("line",{x1:xs(p.v),x2:xs(p.v),y1:ys(p.m-p.s),y2:ys(p.m+p.s),stroke:se.color,"stroke-width":1.2,opacity:.6}));}});
  s.appendChild(E("path",{d,fill:"none",stroke:se.color,"stroke-width":2}));
  se.pts.forEach(p=>{if(!isNaN(p.m))s.appendChild(E("circle",{cx:xs(p.v),cy:ys(p.m),r:3,fill:se.color}));});});
 el.appendChild(s);return series;
}
function heatmap(el,mat,rl,cl,{cell=34,fmt=v=>v.toFixed(2),color,title}={}){
 const n=mat.length,mc=mat[0].length,ml=54,mt=54;const w=ml+mc*cell+8,h=mt+n*cell+8;const s=svg(w,h);
 let amax=0;mat.forEach(r=>r.forEach(v=>{if(v!=null)amax=Math.max(amax,Math.abs(v));}));
 for(let i=0;i<n;i++)for(let j=0;j<mc;j++){const v=mat[i][j];if(v==null)continue;
  s.appendChild(E("rect",{x:ml+j*cell,y:mt+i*cell,width:cell-1,height:cell-1,fill:color(v,amax),rx:2}));
  s.appendChild(E("text",{x:ml+j*cell+cell/2,y:mt+i*cell+cell/2+3,"text-anchor":"middle","font-size":8.5,
    fill:Math.abs(v)>amax*.6?"#fff":"var(--ink)"},fmt(v)));}
 rl.forEach((t,i)=>s.appendChild(E("text",{x:ml-4,y:mt+i*cell+cell/2+3,"text-anchor":"end","font-size":9},t)));
 cl.forEach((t,j)=>{const x=ml+j*cell+cell/2,y=mt-6;const g=E("text",{x,y,"text-anchor":"start","font-size":9,
   transform:`rotate(-45 ${x} ${y})`});g.textContent=t;s.appendChild(g);});
 el.appendChild(s);
}
function scatter(el,ix){const w=360,h=300,ml=44,mb=32,mt=8,mr=8;const s=svg(w,h);
 let mx=1;ix.forEach(i=>{const a=D.cols.analytic[i],f=D.cols.femm[i];if(a!=null)mx=Math.max(mx,a,f);});mx*=1.05;
 const X=v=>ml+v/mx*(w-ml-mr),Y=v=>h-mb-v/mx*(h-mb-mt);
 for(let g=0;g<=4;g++){const t=g/4*mx;s.appendChild(E("line",{x1:ml,x2:w-mr,y1:Y(t),y2:Y(t),class:"gl","stroke-width":1}));
  s.appendChild(E("text",{x:ml-4,y:Y(t)+3,"text-anchor":"end","font-size":9,fill:"var(--mut)"},t.toFixed(0)));
  s.appendChild(E("text",{x:X(t),y:h-mb+12,"text-anchor":"middle","font-size":9,fill:"var(--mut)"},t.toFixed(0)));}
 s.appendChild(E("line",{x1:X(0),y1:Y(0),x2:X(mx),y2:Y(mx),stroke:"var(--mut)","stroke-dasharray":"4 3","stroke-width":1}));
 let over=[];ix.forEach(i=>{const a=D.cols.analytic[i],f=D.cols.femm[i];if(a==null)return;
  s.appendChild(E("circle",{cx:X(f),cy:Y(a),r:2.4,fill:D.cols.driver_bipolar[i]?C.orange:C.blue,opacity:.4}));
  if(a>.1&&f>.1)over.push((a-f)/f*100);});
 s.appendChild(E("text",{x:ml+6,y:mt+12,"font-size":10,fill:"var(--mut)"},"analytic (y) vs FEMM (x)"));
 el.appendChild(s);
 const med=over.sort((a,b)=>a-b)[Math.floor(over.length/2)]||0;
 document.getElementById("scatlbl").textContent=`${over.length} moving designs · analytic overpredicts FEMM by median ${med>=0?"+":""}${med.toFixed(0)}%`;
}
function barsH(el,items,{w=340,color}={}){const bh=22,ml=90,h=items.length*bh+10;const s=svg(w,h);
 const mx=Math.max(...items.map(d=>Math.abs(d.v)),0.001);const X=v=>ml+v/mx*(w-ml-34);
 items.forEach((d,i)=>{const y=i*bh+6;
  s.appendChild(E("rect",{x:ml,y,width:Math.max(1,X(d.v)-ml),height:bh-7,fill:color(d.v),rx:2}));
  s.appendChild(E("text",{x:ml-4,y:y+bh-11,"text-anchor":"end","font-size":9},d.l));
  s.appendChild(E("text",{x:X(d.v)+4,y:y+bh-11,"text-anchor":"start","font-size":9,fill:"var(--mut)"},d.t));});
 el.appendChild(s);
}

// ---------- render ----------
function bestOf(ix){let bi=-1,bv=-1;for(const i of ix){if(D.cols.femm[i]>bv){bv=D.cols.femm[i];bi=i;}}return{bi,bv};}
function renderTiles(ix){const t=clear("tiles");const sp=ix.map(i=>D.cols.femm[i]).filter(x=>x!=null);
 const {bi,bv}=bestOf(ix);const stall=sp.filter(x=>x<=.5).length/Math.max(1,sp.length)*100;
 const tile=(v,l)=>{const d=document.createElement("div");d.className="tile";d.innerHTML=`<div class="v">${v}</div><div class="l">${l}</div>`;t.appendChild(d);};
 tile(ix.length,"designs in view");tile(mean(sp).toFixed(2)+" m/s","mean exit speed");
 tile(bv.toFixed(2)+" m/s","best exit speed");tile(stall.toFixed(0)+"%","stalled (<0.5 m/s)");
 if(bi>=0){const d=document.createElement("div");d.className="tile";d.style.gridColumn="1/-1";
  const parts=D.knobs.map(k=>D.short[k]+"="+fmtLv(k,D.cols[k][bi]));
  d.innerHTML=`<div class="l">best design in view</div><div style="font-size:13px;margin-top:3px">${parts.join(" · ")}</div>`;t.appendChild(d);}
}
const MAIN_ORDER=["driver_bipolar","coil_length_m","bus_voltage_v","remanence_t","i_max_a","pump_envelope","turns","magnet_radius_m","magnet_length_m","radial_thickness_m"];
function renderMain(ix){const g=clear("maincurves");MAIN_ORDER.forEach(k=>{const box=document.createElement("div");
  const lab=document.createElement("div");lab.style.cssText="font-size:11px;color:var(--mut);text-align:center;margin-bottom:-4px";lab.textContent=D.labels[k];box.appendChild(lab);
  const pts=byLevel(ix,k,law).map(o=>({v:o.v,m:o.mean,s:o.sem}));
  const up=pts.filter(p=>!isNaN(p.m));const rising=up.length&&up[up.length-1].m>=up[0].m;
  lineChart(box,k,[{color:rising?C.orange:C.red,pts}],{w:300,h:140});g.appendChild(box);});}
function renderHeat(){const el=clear("heat");heatmap(el,D.beta,D.knobs.map(k=>D.short[k]),D.knobs.map(k=>D.short[k]),
  {cell:34,color:diverge,fmt:v=>(v>=0?"+":"")+v.toFixed(2)});}
function renderModer(ix){const knob=document.getElementById("modknob").value,by=document.getElementById("modby").value;
 const el=clear("moder");const lv=D.factors[by];const cols=[C.blue,C.orange,C.green,C.purple];
 const series=lv.map((bv,i)=>({color:cols[i%4],name:fmtLv(by,bv),
   pts:byLevel(ix,knob,law,i2=>D.cols[by][i2]===bv).map(o=>({v:o.v,m:o.mean}))}));
 lineChart(el,knob,series,{w:520,h:220});
 const leg=document.createElement("div");leg.style.cssText="font-size:11px;color:var(--mut);margin-top:4px";
 leg.innerHTML=series.map(s=>`<span style="color:${s.color}">■</span> ${s.name}`).join("&nbsp;&nbsp;");el.appendChild(leg);}
function renderMap(ix){const kx=document.getElementById("mapx").value,ky=document.getElementById("mapy").value;
 const el=clear("map");const lx=D.factors[kx],ly=D.factors[ky];
 const pf=i=>pol==="all"||D.cols.driver_bipolar[i]===(pol==="bi"?1:0);
 const mat=[];for(let a=ly.length-1;a>=0;a--){const row=[];for(const vx of lx){const vals=[];
   for(const i of ix){if(pf(i)&&D.cols[kx][i]===vx&&D.cols[ky][i]===ly[a]){const s=D.cols.femm[i];if(s!=null)vals.push(s);}}
   row.push(vals.length?mean(vals):null);}mat.push(row);}
 let mn=1e9,mx=-1e9;mat.forEach(r=>r.forEach(v=>{if(v!=null){mn=Math.min(mn,v);mx=Math.max(mx,v);}}));
 const rl=[...ly].reverse().map(v=>shortLv(ky,v)),cl=lx.map(v=>shortLv(kx,v));
 heatmap(el,mat,rl,cl,{cell:52,color:(v)=>v==null?"transparent":seqcol((v-mn)/((mx-mn)||1)),fmt:v=>v==null?"":v.toFixed(1)});
 const ax=document.createElement("div");ax.style.cssText="font-size:11px;color:var(--mut);margin-top:4px";
 ax.textContent=`X: ${D.labels[kx]}   ·   Y: ${D.labels[ky]}   ·   color = mean m/s`;el.appendChild(ax);}
function renderScatter(ix){scatter(clear("scatter"),ix);}
function renderFeas(ix){const knob=document.getElementById("feasknob").value;const el=clear("feas");
 const items=D.factors[knob].map(v=>{const sp=ix.filter(i=>D.cols[knob][i]===v).map(i=>D.cols.femm[i]).filter(x=>x!=null);
   const r=sp.length?sp.filter(x=>x<=.5).length/sp.length*100:0;return{l:fmtLv(knob,v),v:r,t:r.toFixed(0)+"%"};});
 barsH(el,items,{w:340,color:()=>C.blue});}

function renderAll(){const ix=idx();document.getElementById("sub").textContent=
  `${D.n} FEMM designs (66 geometries × 32 driver settings) · showing ${ix.length} after filters`;
 document.getElementById("ndesign").textContent=D.n;
 renderTiles(ix);renderMain(ix);renderModer(ix);renderMap(ix);renderScatter(ix);renderFeas(ix);}

// ---------- controls ----------
function buildFilters(){const f=document.getElementById("filters");D.knobs.forEach(k=>{const d=document.createElement("div");d.className="knob";
  d.innerHTML=`<b>${D.labels[k]}</b>`;const c=document.createElement("div");c.className="chips";
  D.factors[k].forEach(v=>{const b=document.createElement("span");b.className="chip on";b.textContent=fmtLv(k,v);
   b.onclick=()=>{if(F[k].has(v)){if(F[k].size>1){F[k].delete(v);b.classList.remove("on");}}else{F[k].add(v);b.classList.add("on");}renderAll();};
   c.appendChild(b);});d.appendChild(c);f.appendChild(d);});}
function resetF(){D.knobs.forEach(k=>F[k]=new Set(D.factors[k]));document.querySelectorAll(".chip").forEach(c=>c.classList.add("on"));renderAll();}
function seg(id,opts,cur,cb){const s=document.getElementById(id);opts.forEach(([v,t])=>{const b=document.createElement("button");
  b.textContent=t;if(v===cur)b.classList.add("on");b.onclick=()=>{s.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");cb(v);};s.appendChild(b);});}
function opts(sel,list,cur){const s=document.getElementById(sel);list.forEach(k=>{const o=document.createElement("option");o.value=k;o.textContent=D.labels[k];if(k===cur)o.selected=true;s.appendChild(o);});s.onchange=renderAll;}

buildFilters();
seg("lawseg",[["femm","FEMM"],["analytic","analytic"]],"femm",v=>{law=v;renderAll();});
seg("polseg",[["all","all"],["uni","unipolar"],["bi","bipolar"]],"all",v=>{pol=v;renderAll();});
opts("modknob",D.cont,"bus_voltage_v");opts("modby",D.cat,"driver_bipolar");
opts("mapx",D.cont,"bus_voltage_v");opts("mapy",D.cont,"i_max_a");opts("feasknob",D.knobs,"driver_bipolar");
renderHeat();renderAll();
</script></body></html>"""

out = (V.HERE if SUB in (None, ".", "", "study") else V.HERE / SUB) / "dashboard.html"
out.write_text(HTML.replace("__DATA__", json.dumps(DATA, separators=(",", ":"))), encoding="utf-8")
print("wrote", out, f"({out.stat().st_size//1024} KB, {len(designs)} designs)")
