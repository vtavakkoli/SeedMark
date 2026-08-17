"""Dependency-free report writers for SeedMark experiments."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def histogram_svg(null_scores: list[float], marked_scores: list[float], threshold: float) -> str:
    width, height = 920, 420
    ml, mr, mt, mb = 62, 24, 28, 54
    pw, ph = width - ml - mr, height - mt - mb
    minimum = min(min(null_scores), min(marked_scores), -3.5)
    maximum = max(max(null_scores), max(marked_scores), threshold + 1.0, 6.0)
    bins = 28
    step = (maximum - minimum) / bins

    def counts(values: list[float]) -> list[int]:
        out = [0] * bins
        for value in values:
            index = min(bins - 1, max(0, int((value - minimum) / step)))
            out[index] += 1
        return out

    nc = counts(null_scores)
    mc = counts(marked_scores)
    y_max = max(max(nc), max(mc), 1)
    def x(v: float) -> float: return ml + (v - minimum) / (maximum - minimum) * pw

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white" rx="14"/>']
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#334155"/>')
    bar_w = pw / bins
    for i, (a, b) in enumerate(zip(nc, mc)):
        bx = ml + i * bar_w
        ah, bh = a / y_max * ph, b / y_max * ph
        parts.append(f'<rect x="{bx+1:.2f}" y="{mt+ph-ah:.2f}" width="{bar_w*.45:.2f}" height="{ah:.2f}" fill="#64748b" opacity=".78"/>')
        parts.append(f'<rect x="{bx+bar_w*.5:.2f}" y="{mt+ph-bh:.2f}" width="{bar_w*.45:.2f}" height="{bh:.2f}" fill="#ec4899" opacity=".78"/>')
    tx = x(threshold)
    parts.append(f'<line x1="{tx:.2f}" y1="{mt}" x2="{tx:.2f}" y2="{mt+ph}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{tx+6:.2f}" y="{mt+16}" font-family="sans-serif" font-size="13" fill="#991b1b">threshold z={threshold:.1f}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="13">detector z-score</text>')
    parts.append('</svg>')
    return ''.join(parts)


def roc_svg(points: list[dict[str, float]]) -> str:
    width, height = 620, 500
    ml, mr, mt, mb = 64, 26, 32, 58
    pw, ph = width - ml - mr, height - mt - mb
    coords = []
    for point in points:
        x = ml + point['fpr'] * pw
        y = mt + (1.0 - point['tpr']) * ph
        coords.append(f"{x:.2f},{y:.2f}")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" rx="14"/>
<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt}" stroke="#cbd5e1" stroke-dasharray="6 5"/>
<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#334155"/>
<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#334155"/>
<polyline points="{' '.join(coords)}" fill="none" stroke="#7c3aed" stroke-width="3"/>
<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="13">false-positive rate</text>
</svg>'''


def write_interactive_report(path: Path, data: dict[str, object]) -> None:
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    summary = data['summary']
    assert isinstance(summary, dict)
    marked = summary['marked_example']
    null = summary['unmarked_example']
    assert isinstance(marked, dict) and isinstance(null, dict)
    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>SeedMark interactive report</title>
<style>
:root{{--bg:#f8fafc;--card:#fff;--ink:#172033;--muted:#64748b;--accent:#ec4899;--accent2:#7c3aed;--line:#e2e8f0}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#fff,var(--bg) 60%,#fdf2f8);font:15px/1.55 system-ui;color:var(--ink)}}main{{max-width:1160px;margin:auto;padding:28px 18px 60px}}h1{{font-size:clamp(30px,5vw,54px);letter-spacing:-.04em;margin:.15em 0}}h2{{margin:0 0 12px}}.lead{{color:#475569;max-width:880px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}.card{{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 10px 30px #0f172a0d}}.metric strong{{display:block;font-size:27px}}.metric span{{color:var(--muted);font-size:12px;text-transform:uppercase}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}.text{{background:#0f172a;color:#e2e8f0;border-radius:14px;padding:15px;max-height:180px;overflow:auto;font-family:monospace}}.toolbar{{display:flex;gap:10px;align-items:center;margin:12px 0}}button{{border:0;background:var(--accent2);color:#fff;border-radius:10px;padding:9px 13px;font-weight:700}}input{{flex:1}}.candidate{{display:grid;grid-template-columns:130px 1fr 85px;gap:10px;align-items:center;margin:8px 0}}.bar{{height:10px;border-radius:999px;background:#e2e8f0;overflow:hidden;margin:3px 0}}.bar i{{display:block;height:100%;background:var(--accent2)}}.bar.mark i{{background:var(--accent)}}.chosen{{font-weight:800;color:var(--accent)}}.note{{border-left:4px solid #f59e0b;background:#fffbeb;padding:12px;border-radius:9px}}.flow{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.pill{{padding:7px 10px;background:#f1f5f9;border-radius:999px}}svg.chart{{width:100%;height:230px;border:1px solid var(--line);border-radius:12px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div style="font-weight:800;color:#7c3aed;text-transform:uppercase;letter-spacing:.12em;font-size:12px">SeedMark · reproducible teaching experiment</div><h1>Detect a keyed token pattern without the model distribution</h1><p class="lead">The first word creates a public seed. A secret-keyed PRF assigns each candidate token a reproducible pseudorandom score. Generation favors high scores; detection sees only the final token sequence, first word, and key.</p>
<div class="grid"><div class="card metric"><span>Marked z</span><strong>{marked['z_score']:.2f}</strong></div><div class="card metric"><span>Unmarked z</span><strong>{null['z_score']:.2f}</strong></div><div class="card metric"><span>Detection rate</span><strong>{summary['true_positive_rate']*100:.1f}%</strong></div><div class="card metric"><span>False positive</span><strong>{summary['false_positive_rate']*100:.2f}%</strong></div></div>
<section class="card"><h2>Algorithm</h2><div class="flow"><span class="pill">first word</span>→<span class="pill">SHA-256 seed</span>→<span class="pill">HMAC(key, seed ∥ position ∥ token)</span>→<span class="pill">u∈[0,1)</span>→<span class="pill">probability tilt</span>→<span class="pill">z-test</span></div></section>
<div class="two"><section class="card"><h2>Watermarked text</h2><div class="text">{_escape(data['watermarked_text'])}</div></section><section class="card"><h2>Unwatermarked control</h2><div class="text">{_escape(data['unwatermarked_text'])}</div></section></div>
<section class="card"><h2>Interactive token microscope</h2><div class="toolbar"><button id="play">▶ Play</button><input id="slider" type="range" min="0" value="0"/><b id="step"></b><span id="z"></span></div><div id="explain" class="note"></div><div id="cand"></div></section>
<section class="card"><h2>Cumulative evidence</h2><svg id="chart" class="chart" viewBox="0 0 760 230" preserveAspectRatio="none"></svg></section>
<section class="card"><h2>Scientific boundary</h2><div class="note"><b>This is a teaching prototype.</b> It is not Anthropic's production watermark, Google SynthID-Text, or C2PA, and it does not claim that any vendor uses a first-word seed. The detector here intentionally does not receive the language model's probability distribution.</div></section>
<script>const D={payload};const T=D.watermarked_trace,s=document.getElementById('slider');s.max=T.length-1;let timer=null;function pct(x){{return Math.min(100,x*100).toFixed(1)+'%'}}function render(i){{const a=T[i];s.value=i;document.getElementById('step').textContent=`step ${{a.position}}/${{T.length}}`;document.getElementById('z').textContent=`z=${{a.cumulative_z.toFixed(2)}}`;document.getElementById('explain').innerHTML=`context <b>${{a.context_token}}</b> → chose <b>${{a.chosen_token}}</b> · PRF score <b>${{a.chosen_watermark_score.toFixed(3)}}</b>`;document.getElementById('cand').innerHTML=a.candidates.map(c=>`<div class="candidate ${{c.chosen?'chosen':''}}"><span>${{c.token}}${{c.chosen?' ✓':''}}</span><div><div class="bar"><i style="width:${{pct(c.base_probability)}}"></i></div><div class="bar mark"><i style="width:${{pct(c.generation_probability)}}"></i></div></div><span>u=${{c.watermark_score.toFixed(3)}}</span></div>`).join('');draw(i)}}function draw(upto){{const svg=document.getElementById('chart'),w=760,h=230,p=28,zs=T.map(x=>x.cumulative_z),lo=Math.min(-1,...zs),hi=Math.max(D.summary.threshold_z+1,...zs),X=i=>p+i/Math.max(1,T.length-1)*(w-2*p),Y=z=>h-p-(z-lo)/(hi-lo)*(h-2*p),pts=T.slice(0,upto+1).map((x,i)=>`${{X(i)}},${{Y(x.cumulative_z)}}`).join(' '),ty=Y(D.summary.threshold_z);svg.innerHTML=`<line x1="${{p}}" y1="${{ty}}" x2="${{w-p}}" y2="${{ty}}" stroke="#dc2626" stroke-dasharray="7 5"/><polyline points="${{pts}}" fill="none" stroke="#ec4899" stroke-width="3" vector-effect="non-scaling-stroke"/>`}}s.addEventListener('input',e=>render(+e.target.value));document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;return}}timer=setInterval(()=>render((+s.value+1)%T.length),450)}};render(0);</script></main></body></html>'''
    path.write_text(doc, encoding='utf-8')
