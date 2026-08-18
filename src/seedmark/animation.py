"""Pillow animations for the real-LLM SeedMark marked/control experiment.

Frames are built only from recorded generation traces. Long text is rendered with
an intentionally sliding recent-context window; the current token stays separate
from the history so it is the only token highlighted on each frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re

INK = (24, 40, 58)
MUTED = (97, 113, 132)
PANEL = (255, 255, 255)
CANVAS = (246, 249, 252)
LINE = (219, 226, 234)
PINK = (219, 39, 119)
PINK_SOFT = (252, 231, 243)
TEAL = (13, 148, 136)
TEAL_SOFT = (218, 245, 239)
CORAL = (220, 86, 73)
BLUE = (39, 80, 115)
BLUE_SOFT = (227, 237, 246)
GRAY = (132, 145, 159)
_TOKEN_UNITS = re.compile(r"\s+|\S+")


def sentence_parts(result: Any, step_index: int) -> tuple[str, str]:
    """Return completed history and the current token as two separate values."""
    if not 0 <= step_index < len(result.trace):
        raise IndexError("step_index outside generation trace")
    completed = result.prompt + "".join(step.chosen_token_text for step in result.trace[:step_index])
    return completed, result.trace[step_index].chosen_token_text


def recent_sentence_parts(result: Any, step_index: int, *, max_chars: int = 300) -> tuple[str, str]:
    """Return a left-clipped recent context while keeping the current token separate."""
    if max_chars < 40:
        raise ValueError("max_chars must be >= 40")
    completed, current = sentence_parts(result, step_index)
    completed = re.sub(r"\s+", " ", completed).strip()
    if len(completed) <= max_chars:
        return completed, current
    tail = completed[-max_chars:]
    boundary = re.search(r"\s", tail[: max(8, max_chars // 5)])
    if boundary:
        tail = tail[boundary.end():]
    return "… " + tail.lstrip(), current


def cumulative_statistics(result: Any, step_index: int) -> dict[str, float]:
    """Compute z, one-sided p/confidence and prioritized-token share through a step."""
    if not 0 <= step_index < len(result.trace):
        raise IndexError("step_index outside generation trace")
    scores = [step.chosen_watermark_score for step in result.trace[: step_index + 1]]
    n = len(scores)
    z = (sum(scores) - 0.5 * n) / math.sqrt(n / 12.0)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {
        "n": float(n),
        "z": z,
        "p_value": p,
        "confidence": 1.0 - p,
        "priority_share": sum(score >= 0.5 for score in scores) / n,
    }


def _history(result: Any, step_index: int) -> tuple[list[float], list[float], list[float]]:
    zs, confidences, shares, scores = [], [], [], []
    for step in result.trace[: step_index + 1]:
        scores.append(step.chosen_watermark_score)
        n = len(scores)
        z = (sum(scores) - 0.5 * n) / math.sqrt(n / 12.0)
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
        zs.append(z)
        confidences.append(1.0 - p)
        shares.append(sum(score >= 0.5 for score in scores) / n)
    return zs, confidences, shares


def _font(image_font: Any, size: int, *, bold: bool = False) -> Any:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for name in names:
        try:
            return image_font.truetype(name, size=size)
        except OSError:
            pass
    try:
        return image_font.load_default(size=size)
    except TypeError:  # pragma: no cover
        return image_font.load_default()


def _fonts() -> dict[str, Any]:
    try:
        from PIL import ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual assets require Pillow; install SeedMark with the real-llm extra") from exc
    return {
        "title": _font(ImageFont, 28, bold=True), "subtitle": _font(ImageFont, 14),
        "section": _font(ImageFont, 18, bold=True), "sentence": _font(ImageFont, 21),
        "small": _font(ImageFont, 12), "small_bold": _font(ImageFont, 12, bold=True),
        "tiny": _font(ImageFont, 10), "tiny_bold": _font(ImageFont, 10, bold=True),
        "badge": _font(ImageFont, 13, bold=True),
    }


def _text_width(draw: Any, text: str, font: Any) -> float:
    if not text:
        return 0.0
    box = draw.textbbox((0, 0), text, font=font)
    return float(box[2] - box[0])


def _panel(draw: Any, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline=LINE, width=1)


def _dash(draw: Any, xy: tuple[float, float, float, float], *, fill: Any, width: int = 2) -> None:
    x0, y0, x1, y1 = xy
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    p = 0.0
    while p < length:
        q = min(length, p + 7)
        draw.line((x0 + ux*p, y0 + uy*p, x0 + ux*q, y0 + uy*q), fill=fill, width=width)
        p += 12


def _split_fragment(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    if _text_width(draw, text, font) <= max_width:
        return [text]
    chunks, rest = [], text
    while rest:
        lo, hi, best = 1, len(rest), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _text_width(draw, rest[:mid], font) <= max_width:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        chunks.append(rest[:best])
        rest = rest[best:]
    return chunks


def _draw_recent_text(
    draw: Any, *, completed: str, current: str, x: int, y: int,
    max_width: int, font: Any, line_height: int, max_lines: int,
) -> None:
    """Word-wrap history and draw exactly the current token with a pink outline."""
    items: list[tuple[str, bool]] = [(u, False) for u in _TOKEN_UNITS.findall(completed)]
    match = re.match(r"^(\s*)(.*)$", current, re.S)
    leading, body = match.groups() if match else ("", current)
    if leading:
        items.append((leading, False))
    visible_current = body.replace("\n", "↵").replace("\t", "⇥") or "∅"
    items.append((visible_current, True))

    cx, cy, line = float(x), float(y), 0
    for raw, is_current in items:
        if not raw:
            continue
        if raw.isspace():
            width = _text_width(draw, " ", font)
            if cx + width <= x + max_width:
                cx += width
            continue
        for chunk in _split_fragment(draw, raw, font, max_width):
            width = _text_width(draw, chunk, font)
            if cx + width > x + max_width and cx > x:
                line += 1
                if line >= max_lines:
                    return
                cx, cy = float(x), float(y + line * line_height)
            if is_current:
                draw.rounded_rectangle(
                    (cx - 6, cy - 3, cx + width + 6, cy + line_height - 3),
                    radius=7, fill=PINK_SOFT, outline=PINK, width=2,
                )
                draw.text((cx, cy), chunk, font=font, fill=(157, 23, 77))
            else:
                draw.text((cx, cy), chunk, font=font, fill=INK)
            cx += width


def _draw_chart(
    draw: Any, values: list[float], *, box: tuple[int, int, int, int], title: str,
    metric: str, y_min: float, y_max: float, color: Any, fonts: dict[str, Any],
    threshold: float | None = None, threshold_label: str = "",
    reference: list[float] | None = None,
) -> None:
    """Draw a chart with a dedicated header band so titles/metrics never overlap."""
    _panel(draw, box)
    x0, y0, x1, y1 = box
    draw.text((x0 + 20, y0 + 14), title, font=fonts["section"], fill=BLUE)
    draw.text((x1 - 150, y0 + 17), metric, font=fonts["small_bold"], fill=color)
    px0, py0, px1, py1 = x0 + 62, y0 + 55, x1 - 24, y1 - 36
    draw.line((px0, py1, px1, py1), fill=GRAY, width=1)
    draw.line((px0, py0, px0, py1), fill=GRAY, width=1)
    if y_max <= y_min:
        y_max = y_min + 1
    total = max(len(values), len(reference or []), 1)
    def point(i: int, value: float) -> tuple[float, float]:
        xx = px0 + (px1 - px0) * i / max(1, total - 1)
        value = min(max(value, y_min), y_max)
        yy = py1 - (py1 - py0) * (value - y_min) / (y_max - y_min)
        return xx, yy
    if threshold is not None:
        ty = point(0, threshold)[1]
        _dash(draw, (px0, ty, px1, ty), fill=CORAL)
        draw.text((px0 + 5, ty - 15), threshold_label, font=fonts["tiny"], fill=CORAL)
    if reference:
        pts = [point(i, v) for i, v in enumerate(reference)]
        for a, b in zip(pts, pts[1:]):
            _dash(draw, (*a, *b), fill=GRAY)
    pts = [point(i, v) for i, v in enumerate(values)]
    for a, b in zip(pts, pts[1:]):
        draw.line((*a, *b), fill=color, width=3)
    if pts:
        x, y = pts[-1]
        draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
    draw.text((px1 - 95, py1 + 11), "generated tokens", font=fonts["tiny"], fill=MUTED)


def _draw_candidates(draw: Any, step: Any, *, box: tuple[int, int, int, int], fonts: dict[str, Any]) -> None:
    _panel(draw, box)
    x0, y0, x1, y1 = box
    draw.text((x0+22, y0+15), "Next-token distribution", font=fonts["section"], fill=BLUE)
    draw.text((x0+22, y0+42), "dashed = base Qwen · solid = after SeedMark · pink ring = selected", font=fonts["tiny"], fill=MUTED)
    candidates = sorted(step.candidates, key=lambda c: c.base_probability, reverse=True)[:20]
    if not candidates:
        return
    px0, py0, px1, py1 = x0+75, y0+72, x1-30, y1-52
    draw.line((px0, py1, px1, py1), fill=GRAY, width=1)
    draw.line((px0, py0, px0, py1), fill=GRAY, width=1)
    max_p = max(max(c.base_probability, c.generation_probability) for c in candidates) * 1.1 or 1.0
    def point(i: int, p: float) -> tuple[float, float]:
        return (
            px0 + (px1-px0)*i/max(1, len(candidates)-1),
            py1 - (py1-py0)*p/max_p,
        )
    base = [point(i, c.base_probability) for i, c in enumerate(candidates)]
    marked = [point(i, c.generation_probability) for i, c in enumerate(candidates)]
    for a, b in zip(base, base[1:]):
        _dash(draw, (*a, *b), fill=GRAY)
    for a, b in zip(marked, marked[1:]):
        draw.line((*a, *b), fill=BLUE, width=3)
    chosen = None
    for c, p in zip(candidates, marked):
        fill = TEAL if c.watermark_score >= 0.5 else CORAL
        draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4), fill=fill)
        if c.chosen:
            chosen = c
            draw.ellipse((p[0]-10,p[1]-10,p[0]+10,p[1]+10), outline=PINK, width=3)
    if chosen is not None:
        token = chosen.token_text.replace("\n", "↵").replace("\t", "⇥") or "∅"
        line = f"selected {token!r} · base {chosen.base_probability:.3f} → marked {chosen.generation_probability:.3f} · u={chosen.watermark_score:.3f}"
        draw.text((px0, py1+15), line, font=fonts["tiny_bold"], fill=PINK)


def _render_generation_frame(result: Any, step_index: int, *, width: int, height: int) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual assets require Pillow; install SeedMark with the real-llm extra") from exc
    image = Image.new("RGB", (width, height), CANVAS)
    draw, fonts = ImageDraw.Draw(image), _fonts()
    step = result.trace[step_index]
    completed, current = recent_sentence_parts(result, step_index, max_chars=185)
    stats = cumulative_statistics(result, step_index)
    zs, _, shares = _history(result, step_index)

    draw.text((42, 27), "SeedMark · how a small token nudge becomes a watermark", font=fonts["title"], fill=BLUE)
    draw.text((43, 67), f"{result.model_name} · step {step.position}/{len(result.trace)} · top-k {result.top_k} · strength {result.strength:g}", font=fonts["subtitle"], fill=MUTED)
    _draw_candidates(draw, step, box=(34,100,width-34,385), fonts=fonts)

    _panel(draw, (34,402,width-34,595))
    draw.text((58,422), "Sentence generation · sliding recent context", font=fonts["section"], fill=BLUE)
    draw.text((58,450), "The article wraps inside this panel; only the current token is highlighted.", font=fonts["small"], fill=MUTED)
    _draw_recent_text(draw, completed=completed, current=current, x=58, y=481, max_width=width-116, font=fonts["sentence"], line_height=31, max_lines=3)

    left, right = (34,615,width//2-10,height-38), (width//2+10,615,width-34,height-38)
    z_max, z_min = max(4.5, max(zs+[result.detection.threshold_z])+0.8), min(-2.0, min(zs)-0.5)
    _draw_chart(draw, zs, box=left, title="Detector signal", metric=f"z={stats['z']:.2f}", y_min=z_min, y_max=z_max, color=BLUE, fonts=fonts, threshold=result.detection.threshold_z, threshold_label=f"threshold z={result.detection.threshold_z:g}")
    _draw_chart(draw, shares, box=right, title="Prioritized-token share", metric=f"{stats['priority_share']:.1%}", y_min=0, y_max=1, color=TEAL, fonts=fonts, threshold=.5, threshold_label="chance 0.5")
    draw.text((42,height-23), "Teal/coral markers show keyed preference · pink ring/highlight marks only the selected current token.", font=fonts["tiny"], fill=MUTED)
    return image


def _render_detection_frame(marked: Any, control: Any | None, step_index: int, *, width: int, height: int) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual assets require Pillow; install SeedMark with the real-llm extra") from exc
    image = Image.new("RGB", (width, height), CANVAS)
    draw, fonts = ImageDraw.Draw(image), _fonts()
    completed, current = recent_sentence_parts(marked, step_index, max_chars=185)
    stats = cumulative_statistics(marked, step_index)
    zs, confidences, shares = _history(marked, step_index)
    control_zs: list[float] = []
    if control is not None and getattr(control, "trace", None):
        control_zs, _, _ = _history(control, min(step_index, len(control.trace)-1))
    detected = stats["z"] >= marked.detection.threshold_z

    draw.text((42,27), "SeedMark · watermark detection from token IDs", font=fonts["title"], fill=BLUE)
    draw.text((43,67), f"token IDs + first-word seed + secret key · token {step_index+1}/{len(marked.trace)} · no model probabilities", font=fonts["subtitle"], fill=MUTED)
    badge = "WATERMARK DETECTED" if detected else "SIGNAL BUILDING"
    badge_color, badge_fill = (TEAL, TEAL_SOFT) if detected else (BLUE, BLUE_SOFT)
    bw = max(220, int(_text_width(draw, badge, fonts["badge"])+38))
    draw.rounded_rectangle((width-bw-42,27,width-42,69), radius=17, fill=badge_fill, outline=badge_color, width=2)
    draw.text((width-bw-22,38), badge, font=fonts["badge"], fill=badge_color)

    _panel(draw, (34,100,width-34,302))
    draw.text((58,120), "Observed text · sliding recent context", font=fonts["section"], fill=BLUE)
    priority = "prioritized (u ≥ 0.5)" if marked.trace[step_index].chosen_watermark_score >= .5 else "not prioritized (u < 0.5)"
    draw.text((58,148), f"Current token is {priority}. Historical tokens are not highlighted.", font=fonts["small"], fill=MUTED)
    _draw_recent_text(draw, completed=completed, current=current, x=58, y=178, max_width=width-116, font=fonts["sentence"], line_height=31, max_lines=3)

    z_all = zs + control_zs + [marked.detection.threshold_z]
    _draw_chart(
        draw, zs, box=(34,320,width-34,575), title="Detection evidence · solid marked / dashed control",
        metric=f"marked z={stats['z']:.2f}", y_min=min(-2.0,min(z_all)-.5), y_max=max(4.5,max(z_all)+.8),
        color=BLUE, fonts=fonts, threshold=marked.detection.threshold_z,
        threshold_label=f"decision z={marked.detection.threshold_z:g}", reference=control_zs or None,
    )

    left, right = (34,595,width//2-10,height-38), (width//2+10,595,width-34,height-38)
    conf_threshold = _confidence_from_z(marked.detection.threshold_z)
    _draw_chart(draw, confidences, box=left, title="One-sided confidence (1 − p)", metric=f"{stats['confidence']:.2%}", y_min=0, y_max=1, color=PINK, fonts=fonts, threshold=conf_threshold, threshold_label=f"z threshold ≈ {conf_threshold:.2%}")
    _draw_chart(draw, shares, box=right, title="Prioritized-token share", metric=f"{stats['priority_share']:.1%}", y_min=0, y_max=1, color=TEAL, fonts=fonts, threshold=.5, threshold_label="null 0.5")
    draw.text((42,height-23), "1−p is confidence against the detector null, not a posterior probability that the text is AI-generated.", font=fonts["tiny"], fill=MUTED)
    return image


def _confidence_from_z(z: float) -> float:
    return 1.0 - 0.5 * math.erfc(z / math.sqrt(2.0))


def _save_gif(output_path: Path, frames: list[Any], *, frame_ms: int) -> Path:
    if frame_ms < 100:
        raise ValueError("frame_ms must be >= 100")
    if not frames:
        raise ValueError("cannot save an empty animation")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual assets require Pillow; install SeedMark with the real-llm extra") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adaptive = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", getattr(Image, "ADAPTIVE", 1))
    prepared = [frame.convert("P", palette=adaptive, colors=256) for frame in frames]
    durations = [frame_ms] * len(prepared)
    durations[-1] = max(1800, frame_ms*3)
    prepared[0].save(output_path, format="GIF", save_all=True, append_images=prepared[1:], duration=durations, loop=0, optimize=False, disposal=2)
    return output_path


def write_generation_gif(output_path: Path, result: Any, *, frame_ms: int = 650, width: int = 1200, height: int = 900) -> Path:
    """Create a token-by-token generation GIF from a recorded trace."""
    if not result.trace:
        raise ValueError("cannot animate an empty generation trace")
    return _save_gif(output_path, [_render_generation_frame(result,i,width=width,height=height) for i in range(len(result.trace))], frame_ms=frame_ms)


def write_detection_gif(output_path: Path, marked: Any, control: Any | None = None, *, frame_ms: int = 650, width: int = 1200, height: int = 900) -> Path:
    """Create a detector GIF with marked/control z-curves, confidence and share."""
    if not marked.trace:
        raise ValueError("cannot animate an empty generation trace")
    return _save_gif(output_path, [_render_detection_frame(marked,control,i,width=width,height=height) for i in range(len(marked.trace))], frame_ms=frame_ms)


def write_visual_assets(output_dir: Path, marked: Any, control: Any | None = None, *, frame_ms: int = 650, width: int = 1200, height: int = 900) -> dict[str, str]:
    """Write both GIFs and static previews, rendering frame sets sequentially."""
    if width < 800 or height < 650:
        raise ValueError("GIF dimensions must be at least 800x650")
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_frames = [_render_generation_frame(marked,i,width=width,height=height) for i in range(len(marked.trace))]
    generation_path = _save_gif(output_dir/"generation.gif", generation_frames, frame_ms=frame_ms)
    generation_preview = output_dir/"generation-preview.png"
    generation_frames[min(len(generation_frames)-1,max(0,len(generation_frames)*2//3))].save(generation_preview, format="PNG")
    del generation_frames
    detection_frames = [_render_detection_frame(marked,control,i,width=width,height=height) for i in range(len(marked.trace))]
    detection_path = _save_gif(output_dir/"detection.gif", detection_frames, frame_ms=frame_ms)
    detection_preview = output_dir/"detection-preview.png"
    detection_frames[-1].save(detection_preview, format="PNG")
    del detection_frames
    return {"generation_gif":generation_path.name,"detection_gif":detection_path.name,"generation_preview":generation_preview.name,"detection_preview":detection_preview.name}
