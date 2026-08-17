"""Polished standalone HTML reporting for real-LLM SeedMark experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import html
import json
import math


def _result_dict(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload["detection"] = asdict(result.detection)
    return payload


def _confidence_from_z(z: float) -> float:
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return 1.0 - p


def write_qwen_report(
    output_dir: Path,
    marked: Any,
    control: Any,
    *,
    assets: dict[str, str] | None = None,
) -> Path:
    """Write traces, summary JSON, and a self-contained interactive HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = dict(assets or {})
    marked_data, control_data = _result_dict(marked), _result_dict(control)

    (output_dir / "generated_watermarked.txt").write_text(marked.text, encoding="utf-8")
    (output_dir / "generated_control.txt").write_text(control.text, encoding="utf-8")
    (output_dir / "watermarked-trace.json").write_text(
        json.dumps(marked_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "control-trace.json").write_text(
        json.dumps(control_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    threshold_confidence = _confidence_from_z(marked.detection.threshold_z)
    summary = {
        "model": marked.model_name,
        "prompt": marked.prompt,
        "first_word": marked.first_word,
        "top_k": marked.top_k,
        "temperature": marked.temperature,
        "strength": marked.strength,
        "generated_tokens": len(marked.generated_token_ids),
        "watermarked_detection": asdict(marked.detection),
        "control_detection": asdict(control.detection),
        "decision_threshold": {
            "z": marked.detection.threshold_z,
            "one_sided_confidence_equivalent": threshold_confidence,
        },
        "visual_assets": assets,
        "preprocessing": "AutoTokenizer text-only; no image/video processor",
        "note": (
            "Real Qwen logits were used for generation. Detection used token IDs, "
            "the first-word seed and the secret key, but not Qwen logits."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    marked_payload = json.dumps(marked_data, ensure_ascii=False).replace("</", "<\\/")
    control_payload = json.dumps(control_data, ensure_ascii=False).replace("</", "<\\/")
    assets_payload = json.dumps(assets, ensure_ascii=False).replace("</", "<\\/")
    safe_model = html.escape(marked.model_name)
    safe_prompt = html.escape(marked.prompt)
    safe_marked_text = html.escape(marked.text)
    safe_control_text = html.escape(control.text)

    generation_asset = html.escape(assets.get("generation_gif", ""))
    detection_asset = html.escape(assets.get("detection_gif", ""))
    generation_preview = html.escape(assets.get("generation_preview", ""))
    detection_preview = html.escape(assets.get("detection_preview", ""))

    def visual_card(
        *,
        number: str,
        title: str,
        subtitle: str,
        gif: str,
        preview: str,
        caption: str,
    ) -> str:
        if gif:
            media = (
                f'<img class="viz" src="{gif}" alt="{html.escape(title)} animation">'
                f'<div class="asset-actions"><a href="{gif}" download>Download GIF</a>'
                + (f'<a href="{preview}" target="_blank">Open static preview</a>' if preview else "")
                + "</div>"
            )
        else:
            media = (
                '<div class="media-placeholder"><b>Animation disabled for this run.</b>'
                "<span>Run qwen-demo without <code>--no-gif</code> to create this visual.</span></div>"
            )
        return f"""
        <section class="story card">
          <div class="story-head"><span class="step-no">{number}</span><div><h2>{html.escape(title)}</h2>
          <p>{html.escape(subtitle)}</p></div></div>
          {media}
          <p class="caption">{html.escape(caption)}</p>
        </section>
        """

    generation_card = visual_card(
        number="01",
        title="How generation is nudged",
        subtitle="Real Qwen probabilities are gently reweighted by a keyed pseudorandom score before sampling.",
        gif=generation_asset,
        preview=generation_preview,
        caption=(
            "Gray shows the base model distribution. The adjusted distribution is drawn over it. "
            "Teal/coral candidate markers indicate the keyed preference direction, while the exact "
            "token appended at the current step is highlighted in pink."
        ),
    )
    detection_card = visual_card(
        number="02",
        title="How the watermark is detected",
        subtitle="After generation, the detector reconstructs keyed token scores and accumulates evidence over the text.",
        gif=detection_asset,
        preview=detection_preview,
        caption=(
            "The detector does not need the original Qwen probability distribution. It tests whether "
            "the observed token sequence aligns with the secret keyed pattern more strongly than expected "
            "under the null. The displayed 1−p value is test confidence, not a posterior probability of AI authorship."
        ),
    )

    detected_label = "Detected" if marked.detection.detected else "Below threshold"
    detected_class = "good" if marked.detection.detected else "warn"
    control_label = "False positive" if control.detection.detected else "Not detected"
    control_class = "bad" if control.detection.detected else "good"

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeedMark · Real-LLM watermark report</title>
<style>
:root {{
  --bg:#f5f8fb; --paper:#fff; --ink:#182b3d; --muted:#657789; --line:#dbe4ec;
  --blue:#275073; --blue-soft:#e3edf6; --teal:#0d9488; --teal-soft:#daf5ef;
  --coral:#dc5649; --coral-soft:#fdebe7; --pink:#db2777; --pink-soft:#fce7f3;
  --violet:#6f42c1; --shadow:0 18px 55px rgba(31,55,79,.09);
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:
  radial-gradient(circle at 7% 2%,#e6f3f5 0,transparent 26rem),
  radial-gradient(circle at 94% 8%,#fae8f2 0,transparent 25rem),var(--bg);
  font:15px/1.58 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink)}}
a{{color:var(--blue);text-decoration:none}} a:hover{{text-decoration:underline}}
main{{max-width:1260px;margin:auto;padding:34px 22px 84px}}
.hero{{position:relative;overflow:hidden;background:linear-gradient(135deg,#173957,#244f70 55%,#355f78);
  color:white;border-radius:30px;padding:38px 42px;box-shadow:var(--shadow)}}
.hero:after{{content:"";position:absolute;width:330px;height:330px;border-radius:50%;right:-120px;top:-170px;
  background:radial-gradient(circle,rgba(255,255,255,.22),rgba(255,255,255,0))}}
.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.14em;font-weight:800;color:#bfe9e4}}
h1{{font-size:clamp(36px,5.2vw,64px);line-height:1.03;letter-spacing:-.045em;margin:10px 0 14px;max-width:880px}}
.hero p{{max-width:850px;color:#dceaf2;font-size:17px}}
.pills{{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}}
.pill{{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);padding:7px 11px;border-radius:999px;font-size:12px}}
.metrics{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0 26px}}
.card{{background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:22px;box-shadow:0 11px 34px rgba(31,55,79,.055)}}
.metric{{padding:17px 18px}}.metric small{{display:block;color:var(--muted);text-transform:uppercase;font-size:10px;letter-spacing:.08em;font-weight:800}}
.metric b{{font-size:27px;display:block;margin-top:2px}}.status{{display:inline-flex!important;width:max-content;font-size:13px!important;padding:5px 9px;border-radius:999px;margin-top:8px!important}}
.good{{background:var(--teal-soft);color:#08766e}}.bad{{background:var(--coral-soft);color:#aa3f35}}.warn{{background:var(--blue-soft);color:var(--blue)}}
.explainer{{display:grid;grid-template-columns:1.1fr .9fr;gap:16px;margin:20px 0 30px}}
.explain-card{{padding:22px}}.formula{{font:15px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f3f6f8;border:1px solid var(--line);padding:13px;border-radius:13px}}
.flow{{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-top:12px}}.node{{padding:8px 10px;border-radius:10px;background:#eef4f8;border:1px solid #d9e5ed;font-size:12px;font-weight:700}}.arrow{{color:#94a6b6}}
.story{{padding:24px;margin:18px 0}}.story-head{{display:flex;gap:14px;align-items:flex-start;margin-bottom:14px}}
.step-no{{display:grid;place-items:center;min-width:46px;height:46px;border-radius:14px;background:var(--blue);color:#fff;font-weight:900}}
.story h2{{margin:1px 0 2px;font-size:25px;letter-spacing:-.02em}}.story-head p{{margin:0;color:var(--muted)}}
.viz{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:16px;background:#f8fafc}}
.caption{{color:var(--muted);max-width:1050px;margin:12px 3px 0}}
.asset-actions{{display:flex;gap:12px;margin-top:10px}}.asset-actions a{{font-size:12px;font-weight:800;padding:7px 10px;border:1px solid var(--line);border-radius:9px;background:#fff}}
.media-placeholder{{min-height:220px;border:1px dashed #b7c6d2;border-radius:16px;display:grid;place-content:center;text-align:center;color:var(--muted);gap:7px}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}}
.output{{padding:22px}}.output h3{{margin-top:0}}pre{{white-space:pre-wrap;word-break:break-word;margin:0;background:#10293d;color:#e8f1f5;padding:17px;border-radius:14px;max-height:310px;overflow:auto}}
.lab{{padding:24px;margin-top:18px}}.lab-head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;flex-wrap:wrap}}
.toolbar{{display:flex;gap:10px;align-items:center;flex:1;min-width:280px}}button{{border:0;border-radius:10px;background:var(--blue);color:#fff;padding:9px 14px;font-weight:800;cursor:pointer}}
input[type=range]{{flex:1;accent-color:var(--pink)}}.sentence-live{{font:20px/1.7 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f9fbfc;border:1px solid var(--line);padding:15px;border-radius:14px;margin:14px 0}}
.now{{background:var(--pink-soft);border:2px solid var(--pink);color:#9d174d;padding:1px 4px;border-radius:6px}}
.candidates{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}}.candidate{{border:1px solid var(--line);border-radius:12px;padding:10px;background:#fff}}
.candidate.chosen{{border-color:#f09bc2;background:#fff6fa}}.candidate-top{{display:flex;justify-content:space-between;gap:8px;font-size:12px}}
.bars{{margin-top:8px}}.bar{{height:6px;background:#e8edf1;border-radius:99px;margin:4px 0;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--blue)}}.bar.mark i{{background:var(--pink)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin:10px 0}}.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}}.teal{{background:var(--teal)}}.coral{{background:var(--coral)}}.pink{{background:var(--pink)}}
.chart-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}canvas{{width:100%;height:230px;border:1px solid var(--line);border-radius:14px;background:#fff}}
details{{margin-top:18px;border-top:1px solid var(--line);padding-top:14px}}summary{{cursor:pointer;font-weight:800}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}}th,td{{padding:8px;border-bottom:1px solid #e7edf2;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.method{{padding:24px;margin-top:18px}}.method-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}}.mini{{padding:15px;border:1px solid var(--line);border-radius:14px;background:#fbfcfd}}.mini b{{display:block;margin-bottom:5px}}
footer{{margin:30px 4px 0;color:var(--muted);font-size:12px}}
@media(max-width:980px){{.metrics{{grid-template-columns:repeat(3,1fr)}}.explainer,.compare,.chart-wrap{{grid-template-columns:1fr}}.method-grid{{grid-template-columns:1fr}}}}
@media(max-width:650px){{main{{padding:16px 10px 50px}}.hero{{padding:28px 22px;border-radius:22px}}.metrics{{grid-template-columns:1fr 1fr}}.candidates{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
<section class="hero">
  <div class="eyebrow">SeedMark · real-LLM scientific teaching report</div>
  <h1>A tiny keyed nudge in token choice can accumulate into a detectable signal.</h1>
  <p>Qwen supplies the real next-token logits. SeedMark changes only the sampling probabilities.
  Later, the detector reconstructs keyed scores from the observed token IDs—without receiving the original model distribution.</p>
  <div class="pills">
    <span class="pill">Model · {safe_model}</span>
    <span class="pill">Prompt · {safe_prompt}</span>
    <span class="pill">Top-k · {marked.top_k}</span>
    <span class="pill">Strength · {marked.strength:g}</span>
    <span class="pill">Temperature · {marked.temperature:g}</span>
  </div>
</section>

<section class="metrics">
  <div class="card metric"><small>Marked z-score</small><b>{marked.detection.z_score:.2f}</b><span class="status {detected_class}">{detected_label}</span></div>
  <div class="card metric"><small>Control z-score</small><b>{control.detection.z_score:.2f}</b><span class="status {control_class}">{control_label}</span></div>
  <div class="card metric"><small>Decision threshold</small><b>{marked.detection.threshold_z:.2f}</b></div>
  <div class="card metric"><small>Marked p-value</small><b>{marked.detection.p_value_one_sided:.2g}</b></div>
  <div class="card metric"><small>Generated tokens</small><b>{len(marked.generated_token_ids)}</b></div>
  <div class="card metric"><small>First-word seed</small><b style="font-size:18px">{html.escape(marked.first_word)}</b></div>
</section>

<section class="explainer">
  <div class="card explain-card">
    <h2>Generation</h2>
    <div class="formula">q(v) ∝ p(v) × exp(λ · (2u(v) − 1))</div>
    <div class="flow"><span class="node">Qwen logits</span><span class="arrow">→</span><span class="node">top-k p(v)</span><span class="arrow">→</span><span class="node">keyed u(v)</span><span class="arrow">→</span><span class="node">adjusted q(v)</span><span class="arrow">→</span><span class="node">sample</span></div>
  </div>
  <div class="card explain-card">
    <h2>Detection</h2>
    <div class="formula">z = Σ(uₜ − 0.5) / √(n / 12)</div>
    <p>The null hypothesis expects keyed scores to behave like uniform random values. Persistent positive correlation makes z grow.</p>
  </div>
</section>

{generation_card}
{detection_card}

<section class="compare">
  <div class="card output"><h3>Watermarked output</h3><pre>{safe_marked_text}</pre></div>
  <div class="card output"><h3>Matched control</h3><pre>{safe_control_text}</pre></div>
</section>

<section class="card lab">
  <div class="lab-head"><div><h2>Interactive token microscope</h2><p style="color:var(--muted);margin:0">Scrub through the recorded generation trace. No model is running in this page.</p></div>
  <div class="toolbar"><button id="play">▶ Play</button><input id="slider" type="range" min="0" value="0"><b id="step"></b></div></div>
  <div class="legend"><span><i class="dot teal"></i>u ≥ 0.5</span><span><i class="dot coral"></i>u &lt; 0.5</span><span><i class="dot pink"></i>selected/current token</span></div>
  <div id="sentenceLive" class="sentence-live"></div>
  <p id="explain"></p>
  <div id="cand" class="candidates"></div>
  <div class="chart-wrap"><canvas id="zChart" width="560" height="230"></canvas><canvas id="shareChart" width="560" height="230"></canvas></div>

  <details><summary>Raw token trace</summary>
    <div style="overflow:auto"><table><thead><tr><th>Step</th><th>Token</th><th>ID</th><th>Base p</th><th>Marked p</th><th>Δp</th><th>u</th><th>z</th></tr></thead><tbody id="traceTable"></tbody></table></div>
  </details>
</section>

<section class="card method">
  <h2>Scientific interpretation</h2>
  <div class="method-grid">
    <div class="mini"><b>What is demonstrated</b>Generation uses a real Transformer distribution, while the detector can score the resulting token sequence without the original logits.</div>
    <div class="mini"><b>What is not claimed</b>This is a deliberately simple SeedMark teaching algorithm. It is not Anthropic's production watermark, Google SynthID-Text, or C2PA.</div>
    <div class="mini"><b>What 1−p means</b>It is confidence against this detector's null model. It is not a posterior probability that a passage was written by AI.</div>
  </div>
</section>

<footer>Generated by SeedMark · files in this directory are designed to be portable and inspectable offline.</footer>
</main>

<script>
const M={marked_payload};
const C={control_payload};
const ASSETS={assets_payload};
const T=M.trace||[];
const slider=document.getElementById('slider'), stepEl=document.getElementById('step');
const cand=document.getElementById('cand'), explain=document.getElementById('explain');
const sentenceLive=document.getElementById('sentenceLive'), play=document.getElementById('play');
const zChart=document.getElementById('zChart'), shareChart=document.getElementById('shareChart');
slider.max=Math.max(0,T.length-1); let timer=null;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const pct=x=>(100*x).toFixed(2)+'%';
const priority=u=>u>=.5?'prioritised':'deprioritised';
function sentenceAt(i){{
  let prefix=esc(M.prompt);
  for(let j=0;j<i;j++) prefix+=esc(T[j].chosen_token_text);
  return prefix+'<span class="now">'+esc(T[i]?.chosen_token_text||'')+'</span>';
}}
function drawChart(canvas, values, opts={{}}){{
  const ctx=canvas.getContext('2d'), W=canvas.width,H=canvas.height,pad={{l:44,r:16,t:30,b:34}};
  ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#dbe4ec';ctx.strokeRect(.5,.5,W-1,H-1);
  ctx.fillStyle='#275073';ctx.font='bold 14px system-ui';ctx.fillText(opts.title||'',pad.l,20);
  const min=opts.min??Math.min(...values,0), max=opts.max??Math.max(...values,1);
  const X=i=>pad.l+(W-pad.l-pad.r)*(i/Math.max(1,values.length-1));
  const Y=v=>H-pad.b-(H-pad.t-pad.b)*((Math.max(min,Math.min(max,v))-min)/(max-min||1));
  ctx.strokeStyle='#91a1af';ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);ctx.lineTo(W-pad.r,H-pad.b);ctx.stroke();
  if(opts.threshold!==undefined){{
    const y=Y(opts.threshold);ctx.save();ctx.setLineDash([5,4]);ctx.strokeStyle='#dc5649';ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.restore();
    ctx.fillStyle='#dc5649';ctx.font='11px system-ui';ctx.fillText(opts.thresholdLabel||'threshold',pad.l+5,y-5);
  }}
  ctx.strokeStyle=opts.color||'#275073';ctx.lineWidth=2.5;ctx.beginPath();
  values.forEach((v,i)=>{{const x=X(i),y=Y(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke();
  if(values.length){{const x=X(values.length-1),y=Y(values[values.length-1]);ctx.fillStyle=opts.color||'#275073';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill()}}
  ctx.fillStyle='#657789';ctx.font='11px system-ui';ctx.fillText('tokens →',W-68,H-10);
}}
function histories(){{
  const zs=[],shares=[];let sum=0,prio=0;
  T.forEach((a,i)=>{{sum+=a.chosen_watermark_score;if(a.chosen_watermark_score>=.5)prio++;
    const n=i+1;zs.push((sum-.5*n)/Math.sqrt(n/12));shares.push(prio/n)}});
  return {{zs,shares}};
}}
const H=histories();
function render(i){{
  const a=T[i];if(!a)return;slider.value=i;stepEl.textContent=`${{i+1}}/${{T.length}}`;
  sentenceLive.innerHTML=sentenceAt(i);
  explain.innerHTML=`Current token <b>${{esc(a.chosen_token_text)}}</b> · <b>${{priority(a.chosen_watermark_score)}}</b> · base p=${{a.chosen_base_probability.toFixed(4)}} → marked p=${{a.chosen_generation_probability.toFixed(4)}} · u=${{a.chosen_watermark_score.toFixed(3)}} · cumulative z=${{a.cumulative_z.toFixed(2)}}`;
  cand.innerHTML=a.candidates.slice(0,10).map(c=>`<div class="candidate ${{c.chosen?'chosen':''}}">
    <div class="candidate-top"><b>${{esc(c.token_text||'∅')}} ${{c.chosen?'✓':''}}</b><span>#${{c.token_id}} · u=${{c.watermark_score.toFixed(3)}}</span></div>
    <div class="bars"><div class="bar"><i style="width:${{Math.min(100,c.base_probability*100*3)}}%"></i></div>
    <div class="bar mark"><i style="width:${{Math.min(100,c.generation_probability*100*3)}}%"></i></div></div>
    <small>${{priority(c.watermark_score)}} · ${{pct(c.base_probability)}} → ${{pct(c.generation_probability)}}</small></div>`).join('');
  drawChart(zChart,H.zs.slice(0,i+1),{{title:'Cumulative keyed-correlation z-score',min:-2,max:Math.max(4.5,M.detection.threshold_z+1, ...H.zs),threshold:M.detection.threshold_z,thresholdLabel:`z=${{M.detection.threshold_z}}`,color:'#275073'}});
  drawChart(shareChart,H.shares.slice(0,i+1),{{title:'Selected-token priority share',min:0,max:1,threshold:.5,thresholdLabel:'null 0.5',color:'#0d9488'}});
}}
slider.oninput=e=>render(+e.target.value);
play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='▶ Play';return}}
  play.textContent='❚❚ Pause';timer=setInterval(()=>{{let n=(+slider.value+1)%T.length;render(n)}},550)}};
document.getElementById('traceTable').innerHTML=T.map(a=>`<tr><td>${{a.position}}</td><td>${{esc(a.chosen_token_text)}}</td><td>${{a.chosen_token_id}}</td><td>${{a.chosen_base_probability.toFixed(5)}}</td><td>${{a.chosen_generation_probability.toFixed(5)}}</td><td>${{(a.chosen_generation_probability-a.chosen_base_probability).toFixed(5)}}</td><td>${{a.chosen_watermark_score.toFixed(4)}}</td><td>${{a.cumulative_z.toFixed(3)}}</td></tr>`).join('');
render(0);
</script>
</body>
</html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(doc, encoding="utf-8")
    return report_path
