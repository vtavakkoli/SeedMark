"""Polished standalone HTML reporting for real-LLM SeedMark experiments."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import html
import json
import math


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    return value


def _result_dict(result: Any) -> dict[str, Any]:
    payload = _jsonable(result)
    if not isinstance(payload, dict):
        raise TypeError("generation result must serialize to an object")
    return payload


def _confidence_from_z(z: float) -> float:
    return 1.0 - 0.5 * math.erfc(z / math.sqrt(2.0))


def _priority_share(result: Any) -> float:
    trace = getattr(result, "trace", ())
    if not trace:
        return 0.0
    return sum(step.chosen_watermark_score >= 0.5 for step in trace) / len(trace)


def _status(result: Any) -> tuple[str, str]:
    return ("Detected", "detected") if result.detection.detected else ("Not detected", "clear")


def _asset_card(*, label: str, title: str, gif: str, preview: str, caption: str) -> str:
    if gif:
        media = (
            f'<img class="viz" src="{html.escape(gif)}" alt="{html.escape(title)} animation">'
            '<div class="actions">'
            f'<a href="{html.escape(gif)}" download>Download GIF</a>'
            + (
                f'<a href="{html.escape(preview)}" target="_blank" rel="noopener">Open preview</a>'
                if preview else ""
            )
            + "</div>"
        )
    else:
        media = (
            '<div class="placeholder"><b>Animation disabled for this run.</b>'
            '<span>Run <code>seedmark qwen-demo</code> without <code>--no-gif</code>.</span></div>'
        )
    return (
        '<section class="card media">'
        f'<div class="kicker">{html.escape(label)}</div><h2>{html.escape(title)}</h2>'
        f'{media}<p class="caption">{html.escape(caption)}</p></section>'
    )


def write_qwen_report(
    output_dir: Path,
    marked: Any,
    control: Any,
    *,
    assets: dict[str, str] | None = None,
) -> Path:
    """Write traces, summary JSON, and a polished marked-vs-control HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = dict(assets or {})
    marked_data = _result_dict(marked)
    control_data = _result_dict(control)

    (output_dir / "generated_watermarked.txt").write_text(marked.text, encoding="utf-8")
    (output_dir / "generated_control.txt").write_text(control.text, encoding="utf-8")
    (output_dir / "watermarked-trace.json").write_text(
        json.dumps(marked_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "control-trace.json").write_text(
        json.dumps(control_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    marked_label, marked_class = _status(marked)
    control_label, control_class = _status(control)
    contrast_ok = bool(marked.detection.detected and not control.detection.detected)
    marked_share = _priority_share(marked)
    control_share = _priority_share(control)
    threshold_confidence = _confidence_from_z(marked.detection.threshold_z)

    summary = {
        "model": marked.model_name,
        "prompt": marked.prompt,
        "first_word": marked.first_word,
        "top_k": marked.top_k,
        "temperature": marked.temperature,
        "strength": marked.strength,
        "generated_tokens": len(marked.generated_token_ids),
        "watermarked_detection": _jsonable(marked.detection),
        "control_detection": _jsonable(control.detection),
        "comparison": {
            "watermarked_label": marked_label,
            "control_label": control_label,
            "expected_contrast_achieved": contrast_ok,
            "watermarked_priority_share": marked_share,
            "control_priority_share": control_share,
        },
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    generation_card = _asset_card(
        label="Generation animation",
        title="How the watermark nudges token choice",
        gif=assets.get("generation_gif", ""),
        preview=assets.get("generation_preview", ""),
        caption=(
            "The sentence uses a sliding recent-context window so a long article wraps cleanly and never overlaps. "
            "Only the token appended on the current frame is highlighted in pink."
        ),
    )
    detection_card = _asset_card(
        label="Detection animation",
        title="How marked output separates from the control",
        gif=assets.get("detection_gif", ""),
        preview=assets.get("detection_preview", ""),
        caption=(
            "The solid line is the watermarked z-curve, the dashed line is the control z-curve, and the red dashed "
            "line is the decision threshold. The lower cards show one-sided confidence (1−p) and prioritized-token share."
        ),
    )

    contrast_label = "Expected contrast achieved" if contrast_ok else "Review this run"
    contrast_class = "ok" if contrast_ok else "review"
    safe_prompt = html.escape(marked.prompt)
    safe_marked = html.escape(marked.text)
    safe_control = html.escape(control.text)

    doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeedMark · Watermark comparison report</title>
<style>
:root{{--bg:#f4f7fb;--paper:#fff;--ink:#17283a;--muted:#64758a;--line:#dce5ed;--blue:#275073;--teal:#0d9488;--tealSoft:#ddf7f1;--blueSoft:#e7f0f7;--pink:#db2777;--shadow:0 16px 50px rgba(31,55,79,.08)}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 7% 1%,#e3f1f4 0,transparent 28rem),radial-gradient(circle at 94% 4%,#fae7f1 0,transparent 27rem),var(--bg);font:15px/1.58 Inter,system-ui,sans-serif;color:var(--ink)}}main{{max-width:1260px;margin:auto;padding:28px 22px 76px}}.card{{background:rgba(255,255,255,.97);border:1px solid var(--line);border-radius:22px;box-shadow:0 10px 30px rgba(31,55,79,.05)}}
.hero{{background:linear-gradient(135deg,#173957,#244f70 56%,#35657c);color:#fff;border-radius:30px;padding:38px 42px;box-shadow:var(--shadow)}}.kicker{{font-size:11px;text-transform:uppercase;letter-spacing:.14em;font-weight:900;color:var(--teal)}}.hero .kicker{{color:#bcece5}}h1{{font-size:clamp(34px,5vw,60px);line-height:1.04;letter-spacing:-.045em;margin:8px 0 12px}}.hero p{{font-size:17px;color:#dceaf2;max-width:900px}}.pills{{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}}.pill{{padding:7px 11px;border-radius:999px;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.1);font-size:12px}}
.contrast{{display:flex;justify-content:space-between;gap:14px;align-items:center;margin:18px 0;padding:15px 18px;background:#fff;border:1px solid var(--line);border-radius:17px;font-weight:850}}.flow{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}.arrow{{color:#8a9aaa}}.ok,.review{{padding:7px 10px;border-radius:999px;font-size:12px}}.ok{{background:var(--tealSoft);color:#08766e}}.review{{background:#fff4d7;color:#8a5b00}}
.results{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.result{{padding:23px;position:relative;overflow:hidden}}.result:before{{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--teal)}}.result.control:before{{background:var(--blue)}}.head{{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}}.head h2{{margin:2px 0 4px}}.head p{{margin:0;color:var(--muted)}}.badge{{padding:7px 11px;border-radius:999px;font-size:12px;font-weight:900;white-space:nowrap}}.badge.detected{{background:var(--tealSoft);color:#08766e}}.badge.clear{{background:var(--blueSoft);color:var(--blue)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:17px}}.metric{{padding:11px;border:1px solid var(--line);border-radius:13px;background:#fbfcfd}}.metric small{{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;font-weight:900;letter-spacing:.07em}}.metric b{{font-size:18px}}
.prompt{{padding:20px 22px;margin:18px 0}}.prompt h2{{margin:0 0 7px}}code.block{{display:block;white-space:pre-wrap;overflow-wrap:anywhere;padding:13px;border:1px solid var(--line);border-radius:13px;background:#f5f8fa;color:#24425c}}
.media{{padding:24px;margin:18px 0}}.media h2{{margin:3px 0 13px;font-size:25px}}.viz{{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:16px;background:#f8fafc}}.caption{{color:var(--muted);margin:11px 2px 0}}.actions{{display:flex;gap:10px;margin-top:9px}}.actions a{{font-size:12px;font-weight:850;color:var(--blue);text-decoration:none;border:1px solid var(--line);padding:6px 9px;border-radius:9px}}.placeholder{{min-height:180px;border:1px dashed #b7c6d2;border-radius:16px;display:grid;place-content:center;text-align:center;color:var(--muted);gap:7px}}
.texts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}}.text{{padding:21px}}.text h3{{margin:0}}.textTitle{{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;background:#10293d;color:#edf4f7;padding:16px;border-radius:14px;max-height:360px;overflow:auto;font:13px/1.58 ui-monospace,monospace}}
.method{{padding:22px;margin-top:18px}}.methodGrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.mini{{padding:14px;border:1px solid var(--line);border-radius:13px;background:#fbfcfd}}.mini b{{display:block;margin-bottom:4px}}footer{{margin:26px 3px;color:var(--muted);font-size:12px}}@media(max-width:900px){{.results,.texts,.methodGrid{{grid-template-columns:1fr}}.contrast{{align-items:flex-start;flex-direction:column}}.metrics{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{main{{padding:14px 10px 46px}}.hero{{padding:27px 21px}}.head{{flex-direction:column}}}}
</style></head><body><main>
<section class="hero"><div class="kicker">SeedMark · matched real-LLM watermark experiment</div><h1>One prompt. Two generations. Clear detection contrast.</h1><p>The same Qwen prompt is generated with and without SeedMark. The same detector scores both sequences using token IDs, the first-word seed and the secret key—not Qwen logits.</p><div class="pills"><span class="pill">Model · {html.escape(marked.model_name)}</span><span class="pill">Top-k · {marked.top_k}</span><span class="pill">Strength · {marked.strength:g}</span><span class="pill">Tokens · {len(marked.generated_token_ids)}</span></div></section>
<section class="contrast"><div class="flow"><span>Watermarked output</span><span class="arrow">→</span><strong>{marked_label}</strong><span class="arrow">·</span><span>Control / without watermark</span><span class="arrow">→</span><strong>{control_label}</strong></div><span class="{contrast_class}">{contrast_label}</span></section>
<section class="results">
<article class="card result marked"><div class="head"><div><div class="kicker">Watermarked output</div><h2>Keyed sampling nudge enabled</h2><p>Expected to accumulate positive keyed correlation.</p></div><span class="badge {marked_class}">{marked_label}</span></div><div class="metrics"><div class="metric"><small>z-score</small><b>{marked.detection.z_score:.2f}</b></div><div class="metric"><small>1−p</small><b>{1.0-marked.detection.p_value_one_sided:.2%}</b></div><div class="metric"><small>Priority share</small><b>{marked_share:.1%}</b></div><div class="metric"><small>Threshold</small><b>{marked.detection.threshold_z:g}</b></div></div></article>
<article class="card result control"><div class="head"><div><div class="kicker">Control / without watermark</div><h2>Original Qwen sampling</h2><p>Matched run with no keyed probability nudge.</p></div><span class="badge {control_class}">{control_label}</span></div><div class="metrics"><div class="metric"><small>z-score</small><b>{control.detection.z_score:.2f}</b></div><div class="metric"><small>1−p</small><b>{1.0-control.detection.p_value_one_sided:.2%}</b></div><div class="metric"><small>Priority share</small><b>{control_share:.1%}</b></div><div class="metric"><small>Threshold</small><b>{control.detection.threshold_z:g}</b></div></div></article>
</section>
<section class="card prompt"><div class="kicker">Default demonstration prompt</div><h2>Plain-language article: What is AI?</h2><code class="block">{safe_prompt}</code></section>
{generation_card}{detection_card}
<section class="texts"><article class="card text"><div class="textTitle"><h3>Watermarked text</h3><span class="badge {marked_class}">{marked_label}</span></div><pre>{safe_marked}</pre></article><article class="card text"><div class="textTitle"><h3>Control text · without watermark</h3><span class="badge {control_class}">{control_label}</span></div><pre>{safe_control}</pre></article></section>
<section class="card method"><h2>Scientific interpretation</h2><div class="methodGrid"><div class="mini"><b>Generation</b>Real Qwen logits provide the top-k distribution; SeedMark changes only the marked run's sampling probabilities.</div><div class="mini"><b>Detection</b>The detector reconstructs keyed scores from observed token IDs. It does not need Qwen logits, hidden states or next-token probabilities.</div><div class="mini"><b>Meaning of 1−p</b>Confidence against this detector's null model, not a posterior probability that a passage was written by AI.</div></div></section>
<footer>Generated by SeedMark · traces, texts and visual assets are portable and inspectable offline.</footer>
</main></body></html>'''
    report_path = output_dir / "report.html"
    report_path.write_text(doc, encoding="utf-8")
    return report_path
