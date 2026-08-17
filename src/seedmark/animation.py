"""Animated scientific visualizations for real-LLM SeedMark experiments.

Visual assets are rendered entirely from the recorded generation trace.  No model
forward pass is repeated.  The generation animation explains *how* the watermark
nudges token choice, while the detection animation explains *how* the keyed
statistical signal accumulates after generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re


_TOKEN_UNITS = re.compile(r"\s+|\S+")

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
CORAL_SOFT = (253, 235, 231)
BLUE = (39, 80, 115)
BLUE_SOFT = (227, 237, 246)
VIOLET = (111, 66, 193)
GRAY = (132, 145, 159)
AMBER = (202, 122, 18)


def sentence_parts(result: Any, step_index: int) -> tuple[str, str]:
    """Return ``(completed_prefix, current_token)`` for one animation frame."""
    trace = result.trace
    if not 0 <= step_index < len(trace):
        raise IndexError("step_index outside generation trace")
    completed = result.prompt + "".join(step.chosen_token_text for step in trace[:step_index])
    return completed, trace[step_index].chosen_token_text


def cumulative_statistics(result: Any, step_index: int) -> dict[str, float]:
    """Return detector statistics through ``step_index`` (inclusive)."""
    if not 0 <= step_index < len(result.trace):
        raise IndexError("step_index outside generation trace")
    scores = [step.chosen_watermark_score for step in result.trace[: step_index + 1]]
    n = len(scores)
    score_sum = sum(scores)
    z = (score_sum - 0.5 * n) / math.sqrt(n / 12.0)
    p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
    priority_share = sum(score >= 0.5 for score in scores) / n
    return {
        "n": float(n),
        "z": z,
        "p_value": p_value,
        "confidence": 1.0 - p_value,
        "priority_share": priority_share,
    }


def _font(image_font: Any, size: int, *, bold: bool = False) -> Any:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return image_font.truetype(candidate, size=size)
        except OSError:
            pass
    try:
        return image_font.load_default(size=size)
    except TypeError:  # pragma: no cover
        return image_font.load_default()


def _text_width(draw: Any, text: str, font: Any) -> float:
    if not text:
        return 0.0
    box = draw.textbbox((0, 0), text, font=font)
    return float(box[2] - box[0])


def _display_token(token: str) -> str:
    return token.replace("\n", "↵").replace("\t", "⇥") or "∅"


def _round_panel(draw: Any, box: tuple[int, int, int, int], *, radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=PANEL, outline=LINE, width=1)


def _draw_sentence(
    draw: Any,
    *,
    completed: str,
    current: str,
    x: int,
    y: int,
    max_width: int,
    font: Any,
    line_height: int,
    max_lines: int = 4,
    history_scores: list[float] | None = None,
) -> None:
    """Draw sentence context and highlight exactly the current appended token."""
    max_chars = 520
    if len(completed) > max_chars:
        completed = "…" + completed[-max_chars:]

    units: list[tuple[str, str]] = [(u, "normal") for u in _TOKEN_UNITS.findall(completed)]
    match = re.match(r"^(\s*)(.*)$", current, flags=re.S)
    leading, body = match.groups() if match else ("", current)
    if leading:
        units.append((leading, "normal"))
    units.append((_display_token(body), "current"))

    cx, cy, line_no = float(x), float(y), 0
    for unit, kind in units:
        chunks = unit.split("\n") if "\n" in unit and kind != "current" else [unit]
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index:
                line_no += 1
                if line_no >= max_lines:
                    return
                cx, cy = float(x), float(y + line_no * line_height)
            if not chunk:
                continue
            width = _text_width(draw, chunk, font)
            if not chunk.isspace() and cx + width > x + max_width and cx > x:
                line_no += 1
                if line_no >= max_lines:
                    return
                cx, cy = float(x), float(y + line_no * line_height)
            if kind == "current":
                pad_x, pad_y = 7, 4
                draw.rounded_rectangle(
                    (cx - pad_x, cy - pad_y, cx + width + pad_x, cy + line_height - 5 + pad_y),
                    radius=8,
                    fill=PINK_SOFT,
                    outline=PINK,
                    width=2,
                )
                draw.text((cx, cy), chunk, font=font, fill=(157, 23, 77))
            else:
                draw.text((cx, cy), chunk, font=font, fill=INK)
            cx += width


def _draw_colored_token_sentence(
    draw: Any,
    result: Any,
    step_index: int,
    *,
    x: int,
    y: int,
    max_width: int,
    font: Any,
    line_height: int,
    max_lines: int = 5,
) -> None:
    """Draw prompt + generated tokens, coloring historical choices by PRF direction."""
    prompt = result.prompt
    tokens = [step.chosen_token_text for step in result.trace[: step_index + 1]]
    scores = [step.chosen_watermark_score for step in result.trace[: step_index + 1]]

    cx, cy, line_no = float(x), float(y), 0

    def draw_piece(text: str, fill: tuple[int, int, int], *, current: bool = False) -> bool:
        nonlocal cx, cy, line_no
        match = re.match(r"^(\s*)(.*)$", text, flags=re.S)
        leading, body = match.groups() if match else ("", text)
        if leading:
            for unit in _TOKEN_UNITS.findall(leading):
                cx += _text_width(draw, unit, font)
        visible = _display_token(body)
        width = _text_width(draw, visible, font)
        if cx + width > x + max_width and cx > x:
            line_no += 1
            if line_no >= max_lines:
                return False
            cx, cy = float(x), float(y + line_no * line_height)
        if current:
            draw.rounded_rectangle(
                (cx - 6, cy - 4, cx + width + 6, cy + line_height - 5 + 4),
                radius=7,
                fill=PINK_SOFT,
                outline=PINK,
                width=2,
            )
        draw.text((cx, cy), visible, font=font, fill=fill)
        cx += width
        return True

    for unit in _TOKEN_UNITS.findall(prompt):
        width = _text_width(draw, unit, font)
        if not unit.isspace() and cx + width > x + max_width and cx > x:
            line_no += 1
            if line_no >= max_lines:
                return
            cx, cy = float(x), float(y + line_no * line_height)
        draw.text((cx, cy), unit, font=font, fill=INK)
        cx += width

    for idx, (token, score) in enumerate(zip(tokens, scores)):
        fill = TEAL if score >= 0.5 else CORAL
        if not draw_piece(token, fill, current=idx == step_index):
            return


def _dashed_line(draw: Any, xy: tuple[float, float, float, float], *, fill: Any, width: int = 2, dash: int = 8, gap: int = 6) -> None:
    x0, y0, x1, y1 = xy
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    p = 0.0
    while p < length:
        q = min(length, p + dash)
        draw.line((x0 + ux * p, y0 + uy * p, x0 + ux * q, y0 + uy * q), fill=fill, width=width)
        p += dash + gap


def _plot_axes(draw: Any, box: tuple[int, int, int, int], *, title: str, x_label: str, y_label: str, fonts: dict[str, Any]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 31), title, font=fonts["section"], fill=BLUE)
    plot = (x0 + 58, y0 + 8, x1 - 15, y1 - 38)
    px0, py0, px1, py1 = plot
    draw.line((px0, py1, px1, py1), fill=GRAY, width=1)
    draw.line((px0, py0, px0, py1), fill=GRAY, width=1)
    draw.text((px0 + (px1 - px0) // 3, py1 + 13), x_label, font=fonts["tiny"], fill=MUTED)
    draw.text((px0 - 49, py0 - 2), y_label, font=fonts["tiny"], fill=MUTED)
    return plot


def _draw_candidate_distribution(draw: Any, step: Any, *, box: tuple[int, int, int, int], fonts: dict[str, Any]) -> None:
    plot = _plot_axes(
        draw,
        box,
        title="Next-token candidate distribution",
        x_label="candidate next tokens (ranked by base likelihood)",
        y_label="prob.",
        fonts=fonts,
    )
    candidates = sorted(step.candidates, key=lambda c: c.base_probability, reverse=True)[:20]
    if not candidates:
        return
    px0, py0, px1, py1 = plot
    max_p = max(c.base_probability for c in candidates) * 1.12 or 1.0

    def point(index: int, probability: float) -> tuple[float, float]:
        x = px0 + (px1 - px0) * index / max(1, len(candidates) - 1)
        y = py1 - (py1 - py0) * probability / max_p
        return x, y

    base_points = [point(i, c.base_probability) for i, c in enumerate(candidates)]
    marked_points = [point(i, c.generation_probability) for i, c in enumerate(candidates)]

    for a, b in zip(base_points, base_points[1:]):
        _dashed_line(draw, (*a, *b), fill=GRAY, width=2, dash=6, gap=4)
    for a, b in zip(marked_points, marked_points[1:]):
        draw.line((*a, *b), fill=BLUE, width=3)

    chosen_rank = 0
    for i, (candidate, mp) in enumerate(zip(candidates, marked_points)):
        fill = TEAL if candidate.watermark_score >= 0.5 else CORAL
        draw.ellipse((mp[0] - 4, mp[1] - 4, mp[0] + 4, mp[1] + 4), fill=fill)
        if candidate.chosen:
            chosen_rank = i
            draw.ellipse((mp[0] - 10, mp[1] - 10, mp[0] + 10, mp[1] + 10), outline=PINK, width=3)
            draw.line((mp[0] - 7, mp[1], mp[0] + 7, mp[1]), fill=PINK, width=2)
            draw.line((mp[0], mp[1] - 7, mp[0], mp[1] + 7), fill=PINK, width=2)

    lx, ly = px1 - 240, py0 + 8
    _dashed_line(draw, (lx, ly, lx + 30, ly), fill=GRAY, width=2)
    draw.text((lx + 38, ly - 8), "base model", font=fonts["tiny"], fill=MUTED)
    draw.line((lx, ly + 22, lx + 30, ly + 22), fill=BLUE, width=3)
    draw.text((lx + 38, ly + 14), "after SeedMark", font=fonts["tiny"], fill=BLUE)
    draw.ellipse((lx, ly + 40, lx + 8, ly + 48), fill=TEAL)
    draw.text((lx + 14, ly + 34), "u ≥ 0.5", font=fonts["tiny"], fill=TEAL)
    draw.ellipse((lx + 95, ly + 40, lx + 103, ly + 48), fill=CORAL)
    draw.text((lx + 109, ly + 34), "u < 0.5", font=fonts["tiny"], fill=CORAL)

    chosen = candidates[chosen_rank]
    draw.text(
        (px0, py1 + 14),
        f"selected: {_display_token(chosen.token_text)!r}  ·  base {chosen.base_probability:.3f}  →  marked {chosen.generation_probability:.3f}  ·  u={chosen.watermark_score:.3f}",
        font=fonts["tiny_bold"],
        fill=PINK,
    )


def _history(result: Any, step_index: int) -> tuple[list[float], list[float], list[float]]:
    zs: list[float] = []
    confidences: list[float] = []
    shares: list[float] = []
    scores: list[float] = []
    for step in result.trace[: step_index + 1]:
        scores.append(step.chosen_watermark_score)
        n = len(scores)
        z = (sum(scores) - 0.5 * n) / math.sqrt(n / 12.0)
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
        zs.append(z)
        confidences.append(1.0 - p)
        shares.append(sum(s >= 0.5 for s in scores) / n)
    return zs, confidences, shares


def _draw_line_chart(
    draw: Any,
    values: list[float],
    *,
    box: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    fill: tuple[int, int, int],
    fonts: dict[str, Any],
    title: str,
    x_label: str,
    y_label: str,
    threshold: float | None = None,
    threshold_label: str | None = None,
    dashed_reference: list[float] | None = None,
) -> None:
    plot = _plot_axes(draw, box, title=title, x_label=x_label, y_label=y_label, fonts=fonts)
    px0, py0, px1, py1 = plot
    if y_max <= y_min:
        y_max = y_min + 1.0

    def point(i: int, value: float, total: int) -> tuple[float, float]:
        x = px0 + (px1 - px0) * i / max(1, total - 1)
        clipped = min(max(value, y_min), y_max)
        y = py1 - (py1 - py0) * (clipped - y_min) / (y_max - y_min)
        return x, y

    if threshold is not None:
        ty = py1 - (py1 - py0) * (threshold - y_min) / (y_max - y_min)
        _dashed_line(draw, (px0, ty, px1, ty), fill=CORAL, width=2, dash=4, gap=4)
        if threshold_label:
            draw.text((px0 + 5, ty - 17), threshold_label, font=fonts["tiny"], fill=CORAL)

    if dashed_reference:
        ref_points = [point(i, value, len(dashed_reference)) for i, value in enumerate(dashed_reference)]
        for a, b in zip(ref_points, ref_points[1:]):
            _dashed_line(draw, (*a, *b), fill=GRAY, width=2, dash=6, gap=4)

    points = [point(i, value, len(values)) for i, value in enumerate(values)]
    for a, b in zip(points, points[1:]):
        draw.line((*a, *b), fill=fill, width=3)
    if points:
        x, y = points[-1]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill)


def _fonts() -> dict[str, Any]:
    try:
        from PIL import ImageFont
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual report assets require Pillow; install SeedMark with the real-llm extra") from exc
    return {
        "title": _font(ImageFont, 30, bold=True),
        "subtitle": _font(ImageFont, 16),
        "section": _font(ImageFont, 19, bold=True),
        "sentence": _font(ImageFont, 23),
        "small": _font(ImageFont, 13),
        "small_bold": _font(ImageFont, 13, bold=True),
        "tiny": _font(ImageFont, 11),
        "tiny_bold": _font(ImageFont, 11, bold=True),
        "metric": _font(ImageFont, 26, bold=True),
        "badge": _font(ImageFont, 15, bold=True),
    }


def _render_generation_frame(result: Any, step_index: int, *, width: int, height: int) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual report assets require Pillow; install SeedMark with the real-llm extra") from exc

    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    step = result.trace[step_index]
    completed, current = sentence_parts(result, step_index)
    stats = cumulative_statistics(result, step_index)
    zs, _, shares = _history(result, step_index)

    draw.text((42, 28), "SeedMark · how a small token nudge becomes a watermark", font=fonts["title"], fill=BLUE)
    draw.text(
        (43, 70),
        f"{result.model_name}  ·  step {step.position}/{len(result.trace)}  ·  top-k {result.top_k}  ·  strength {result.strength:g}",
        font=fonts["subtitle"],
        fill=MUTED,
    )

    _round_panel(draw, (34, 105, width - 34, 400))
    _draw_candidate_distribution(draw, step, box=(58, 145, width - 58, 382), fonts=fonts)

    _round_panel(draw, (34, 420, width - 34, 590))
    draw.text((58, 442), "Sentence generation", font=fonts["section"], fill=BLUE)
    draw.text((58, 470), "The exact token appended at this step is highlighted in pink.", font=fonts["small"], fill=MUTED)
    _draw_sentence(
        draw,
        completed=completed,
        current=current,
        x=58,
        y=505,
        max_width=width - 116,
        font=fonts["sentence"],
        line_height=34,
        max_lines=2,
    )

    left = (34, 625, width // 2 - 10, height - 38)
    right = (width // 2 + 10, 625, width - 34, height - 38)
    _round_panel(draw, left)
    _round_panel(draw, right)
    z_max = max(4.5, max(zs + [result.detection.threshold_z]) + 0.8)
    _draw_line_chart(
        draw,
        zs,
        box=(left[0] + 20, left[1] + 48, left[2] - 18, left[3] - 8),
        y_min=-2.0,
        y_max=z_max,
        fill=BLUE,
        fonts=fonts,
        title="Detector signal accumulating",
        x_label="generated tokens",
        y_label="z",
        threshold=result.detection.threshold_z,
        threshold_label=f"detection threshold z={result.detection.threshold_z:g}",
    )
    _draw_line_chart(
        draw,
        shares,
        box=(right[0] + 20, right[1] + 48, right[2] - 18, right[3] - 8),
        y_min=0.0,
        y_max=1.0,
        fill=TEAL,
        fonts=fonts,
        title="Selected-token priority share",
        x_label="generated tokens",
        y_label="share",
        threshold=0.5,
        threshold_label="chance baseline 0.5",
    )
    draw.text(
        (right[0] + 36, right[1] + 14),
        f"current: {stats['priority_share']:.1%}",
        font=fonts["small_bold"],
        fill=TEAL,
    )

    draw.text((42, height - 24), "Green/red = keyed preference direction · pink outline = token actually selected", font=fonts["tiny"], fill=MUTED)
    return image


def _render_detection_frame(marked: Any, control: Any | None, step_index: int, *, width: int, height: int) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual report assets require Pillow; install SeedMark with the real-llm extra") from exc

    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    fonts = _fonts()
    stats = cumulative_statistics(marked, step_index)
    zs, confidences, shares = _history(marked, step_index)
    control_zs: list[float] = []
    if control is not None and control.trace:
        upto = min(step_index, len(control.trace) - 1)
        control_zs, _, _ = _history(control, upto)

    detected = stats["z"] >= marked.detection.threshold_z
    status_fill = TEAL_SOFT if detected else BLUE_SOFT
    status_text = TEAL if detected else BLUE

    draw.text((42, 28), "SeedMark · how the watermark is detected without model probabilities", font=fonts["title"], fill=BLUE)
    draw.text(
        (43, 70),
        f"Detector sees token IDs + first-word seed + secret key  ·  token {step_index + 1}/{len(marked.trace)}",
        font=fonts["subtitle"],
        fill=MUTED,
    )
    badge = "WATERMARK SIGNAL DETECTED" if detected else "SIGNAL BUILDING"
    draw.rounded_rectangle((width - 330, 28, width - 42, 72), radius=18, fill=status_fill, outline=status_text, width=2)
    draw.text((width - 308, 39), badge, font=fonts["badge"], fill=status_text)

    _round_panel(draw, (34, 105, width - 34, 330))
    draw.text((58, 128), "Observed text", font=fonts["section"], fill=BLUE)
    draw.text(
        (58, 157),
        "Teal tokens align with the keyed preferred half (u ≥ 0.5); coral tokens do not. Current token is outlined.",
        font=fonts["small"],
        fill=MUTED,
    )
    _draw_colored_token_sentence(
        draw,
        marked,
        step_index,
        x=58,
        y=198,
        max_width=width - 116,
        font=fonts["sentence"],
        line_height=34,
        max_lines=3,
    )

    _round_panel(draw, (34, 350, width - 34, 610))
    z_all = zs + control_zs + [marked.detection.threshold_z]
    z_max = max(4.5, max(z_all) + 0.8)
    z_min = min(-2.0, min(z_all) - 0.5)
    _draw_line_chart(
        draw,
        zs,
        box=(58, 400, width - 58, 594),
        y_min=z_min,
        y_max=z_max,
        fill=BLUE,
        fonts=fonts,
        title="Cumulative keyed-correlation score",
        x_label="generated tokens",
        y_label="z",
        threshold=marked.detection.threshold_z,
        threshold_label=f"decision threshold z={marked.detection.threshold_z:g}",
        dashed_reference=control_zs if control_zs else None,
    )
    if control_zs:
        draw.text((width - 290, 380), "solid: watermarked", font=fonts["tiny_bold"], fill=BLUE)
        draw.text((width - 170, 380), "dashed: matched control", font=fonts["tiny"], fill=GRAY)

    left = (34, 630, width // 2 - 10, height - 38)
    right = (width // 2 + 10, 630, width - 34, height - 38)
    _round_panel(draw, left)
    _round_panel(draw, right)

    threshold_confidence = 1.0 - 0.5 * math.erfc(marked.detection.threshold_z / math.sqrt(2.0))
    _draw_line_chart(
        draw,
        confidences,
        box=(left[0] + 20, left[1] + 48, left[2] - 18, left[3] - 8),
        y_min=0.0,
        y_max=1.0,
        fill=PINK,
        fonts=fonts,
        title="One-sided detector confidence (1 − p)",
        x_label="generated tokens",
        y_label="1-p",
        threshold=threshold_confidence,
        threshold_label=f"z threshold ≈ {threshold_confidence:.2%}",
    )
    _draw_line_chart(
        draw,
        shares,
        box=(right[0] + 20, right[1] + 48, right[2] - 18, right[3] - 8),
        y_min=0.0,
        y_max=1.0,
        fill=TEAL,
        fonts=fonts,
        title="Share of selected tokens with u ≥ 0.5",
        x_label="generated tokens",
        y_label="share",
        threshold=0.5,
        threshold_label="null expectation 0.5",
    )

    draw.text(
        (left[0] + 34, left[1] + 16),
        f"z={stats['z']:.2f}   ·   p={stats['p_value']:.3g}   ·   1-p={stats['confidence']:.2%}",
        font=fonts["small_bold"],
        fill=PINK if detected else BLUE,
    )
    draw.text(
        (right[0] + 34, right[1] + 16),
        f"priority share={stats['priority_share']:.1%}",
        font=fonts["small_bold"],
        fill=TEAL,
    )
    draw.text(
        (42, height - 24),
        "1−p is a test confidence against the null, not a posterior probability that the text is AI-generated.",
        font=fonts["tiny"],
        fill=MUTED,
    )
    return image


def _save_gif(output_path: Path, frames: list[Any], *, frame_ms: int) -> Path:
    if frame_ms < 100:
        raise ValueError("frame_ms must be >= 100")
    if not frames:
        raise ValueError("cannot save an empty animation")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Visual report assets require Pillow; install SeedMark with the real-llm extra") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adaptive = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", getattr(Image, "ADAPTIVE", 1))
    prepared = [frame.convert("P", palette=adaptive, colors=256) for frame in frames]
    durations = [frame_ms] * len(prepared)
    durations[-1] = max(1800, frame_ms * 3)
    prepared[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=prepared[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output_path


def write_generation_gif(
    output_path: Path,
    result: Any,
    *,
    frame_ms: int = 650,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Create a token-by-token generation GIF from a recorded real-LLM trace."""
    if not result.trace:
        raise ValueError("cannot animate an empty generation trace")
    frames = [_render_generation_frame(result, i, width=width, height=height) for i in range(len(result.trace))]
    return _save_gif(output_path, frames, frame_ms=frame_ms)


def write_detection_gif(
    output_path: Path,
    marked: Any,
    control: Any | None = None,
    *,
    frame_ms: int = 650,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Create an animation of the detector accumulating keyed evidence."""
    if not marked.trace:
        raise ValueError("cannot animate an empty generation trace")
    frames = [
        _render_detection_frame(marked, control, i, width=width, height=height)
        for i in range(len(marked.trace))
    ]
    return _save_gif(output_path, frames, frame_ms=frame_ms)


def write_visual_assets(
    output_dir: Path,
    marked: Any,
    control: Any | None = None,
    *,
    frame_ms: int = 650,
    width: int = 1200,
    height: int = 900,
) -> dict[str, str]:
    """Write both GIFs plus static PNG previews and return relative asset names."""
    if width < 800 or height < 650:
        raise ValueError("GIF dimensions must be at least 800x650")
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_frames = [
        _render_generation_frame(marked, i, width=width, height=height)
        for i in range(len(marked.trace))
    ]
    detection_frames = [
        _render_detection_frame(marked, control, i, width=width, height=height)
        for i in range(len(marked.trace))
    ]
    generation_path = _save_gif(output_dir / "generation.gif", generation_frames, frame_ms=frame_ms)
    detection_path = _save_gif(output_dir / "detection.gif", detection_frames, frame_ms=frame_ms)

    generation_preview = output_dir / "generation-preview.png"
    detection_preview = output_dir / "detection-preview.png"
    generation_frames[min(len(generation_frames) - 1, max(0, len(generation_frames) * 2 // 3))].save(generation_preview, format="PNG")
    detection_frames[-1].save(detection_preview, format="PNG")

    return {
        "generation_gif": generation_path.name,
        "detection_gif": detection_path.name,
        "generation_preview": generation_preview.name,
        "detection_preview": detection_preview.name,
    }
