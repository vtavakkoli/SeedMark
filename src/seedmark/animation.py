"""Animated educational visualization for real-LLM SeedMark runs.

The GIF is intentionally derived only from the recorded generation trace.  It does
not re-run the model.  At each frame the sentence prefix is rendered normally and
the token selected at that generation step is highlighted, making it visually
clear which token is being added to the text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re


_TOKEN_UNITS = re.compile(r"\s+|\S+")


def sentence_parts(result: Any, step_index: int) -> tuple[str, str]:
    """Return ``(completed_prefix, current_token)`` for one animation frame.

    ``step_index`` is zero-based.  The current token is deliberately excluded from
    the prefix so a renderer can highlight exactly the token being appended.
    """
    trace = result.trace
    if not 0 <= step_index < len(trace):
        raise IndexError("step_index outside generation trace")
    completed = result.prompt + "".join(step.chosen_token_text for step in trace[:step_index])
    return completed, trace[step_index].chosen_token_text


def _font(image_font: Any, size: int, *, bold: bool = False) -> Any:
    """Load a portable font without requiring a repository-bundled font file."""
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return image_font.truetype(candidate, size=size)
        except OSError:
            pass
    try:
        return image_font.load_default(size=size)
    except TypeError:  # pragma: no cover - old Pillow fallback
        return image_font.load_default()


def _text_width(draw: Any, text: str, font: Any) -> float:
    if not text:
        return 0.0
    box = draw.textbbox((0, 0), text, font=font)
    return float(box[2] - box[0])


def _display_token(token: str) -> str:
    return token.replace("\n", "↵").replace("\t", "⇥") or "∅"


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
) -> None:
    """Draw recent generated context and highlight only the current token."""
    # Long runs stay readable by showing the most recent context instead of
    # shrinking the font until it becomes illegible.
    max_chars = 430
    if len(completed) > max_chars:
        completed = "…" + completed[-max_chars:]

    normal = (31, 41, 55)
    muted = (100, 116, 139)
    highlight_bg = (252, 231, 243)
    highlight_border = (236, 72, 153)
    highlight_text = (157, 23, 77)

    units: list[tuple[str, bool]] = [(u, False) for u in _TOKEN_UNITS.findall(completed)]
    # Keep leading whitespace outside the highlighted chip so the color marks the
    # lexical token itself rather than a blank rectangle.
    match = re.match(r"^(\s*)(.*)$", current, flags=re.S)
    leading, body = match.groups() if match else ("", current)
    if leading:
        units.append((leading, False))
    units.append((_display_token(body), True))

    cx, cy, line = float(x), float(y), 0
    for unit, highlighted in units:
        if "\n" in unit and not highlighted:
            chunks = unit.split("\n")
        else:
            chunks = [unit]
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index:
                line += 1
                if line >= max_lines:
                    return
                cx, cy = float(x), float(y + line * line_height)
            if not chunk:
                continue
            width = _text_width(draw, chunk, font)
            is_space = chunk.isspace()
            if not is_space and cx + width > x + max_width and cx > x:
                line += 1
                if line >= max_lines:
                    return
                cx, cy = float(x), float(y + line * line_height)
            if highlighted:
                pad_x, pad_y = 7, 4
                draw.rounded_rectangle(
                    (cx - pad_x, cy - pad_y, cx + width + pad_x, cy + line_height - 5 + pad_y),
                    radius=8,
                    fill=highlight_bg,
                    outline=highlight_border,
                    width=2,
                )
                draw.text((cx, cy), chunk, font=font, fill=highlight_text)
            else:
                draw.text((cx, cy), chunk, font=font, fill=muted if chunk == "…" else normal)
            cx += width


def _draw_candidate_rows(draw: Any, step: Any, *, x: int, y: int, width: int, fonts: dict[str, Any]) -> None:
    candidates = list(step.candidates)[:6]
    if not candidates:
        return
    max_prob = max(max(c.base_probability, c.generation_probability) for c in candidates) or 1.0
    label_w = 150
    bar_x = x + label_w
    bar_w = width - label_w - 68
    row_h = 34

    draw.text((x, y - 28), "Top candidates", font=fonts["section"], fill=(30, 41, 59))
    draw.text((bar_x, y - 25), "base", font=fonts["tiny"], fill=(109, 40, 217))
    draw.text((bar_x + 60, y - 25), "watermarked", font=fonts["tiny"], fill=(190, 24, 93))

    for idx, candidate in enumerate(candidates):
        ry = y + idx * row_h
        if candidate.chosen:
            draw.rounded_rectangle(
                (x - 8, ry - 4, x + width, ry + row_h - 4),
                radius=8,
                fill=(253, 242, 248),
                outline=(244, 114, 182),
                width=1,
            )
        token = _display_token(candidate.token_text)
        if len(token) > 18:
            token = token[:17] + "…"
        label = f"{token}  #{candidate.token_id}"
        draw.text((x, ry + 3), label, font=fonts["small_bold"] if candidate.chosen else fonts["small"], fill=(157, 23, 77) if candidate.chosen else (51, 65, 85))

        base_width = int(bar_w * candidate.base_probability / max_prob)
        marked_width = int(bar_w * candidate.generation_probability / max_prob)
        draw.rounded_rectangle((bar_x, ry + 5, bar_x + bar_w, ry + 10), radius=3, fill=(226, 232, 240))
        draw.rounded_rectangle((bar_x, ry + 5, bar_x + max(2, base_width), ry + 10), radius=3, fill=(124, 58, 237))
        draw.rounded_rectangle((bar_x, ry + 15, bar_x + bar_w, ry + 20), radius=3, fill=(226, 232, 240))
        draw.rounded_rectangle((bar_x, ry + 15, bar_x + max(2, marked_width), ry + 20), radius=3, fill=(236, 72, 153))
        draw.text((x + width - 57, ry + 3), f"u={candidate.watermark_score:.2f}", font=fonts["tiny"], fill=(100, 116, 139))


def _draw_z_gauge(draw: Any, *, z: float, threshold: float, x: int, y: int, width: int, font: Any) -> None:
    min_z, max_z = -2.0, 7.0
    clipped = min(max(z, min_z), max_z)
    gauge_x0, gauge_x1 = x, x + width
    draw.rounded_rectangle((gauge_x0, y, gauge_x1, y + 14), radius=7, fill=(226, 232, 240))
    zero_x = gauge_x0 + int((0.0 - min_z) / (max_z - min_z) * width)
    current_x = gauge_x0 + int((clipped - min_z) / (max_z - min_z) * width)
    threshold_x = gauge_x0 + int((threshold - min_z) / (max_z - min_z) * width)
    lo, hi = sorted((zero_x, current_x))
    draw.rounded_rectangle((lo, y, max(lo + 3, hi), y + 14), radius=7, fill=(236, 72, 153) if z >= threshold else (124, 58, 237))
    draw.line((threshold_x, y - 5, threshold_x, y + 19), fill=(15, 23, 42), width=2)
    draw.text((threshold_x - 26, y + 23), "z=3", font=font, fill=(71, 85, 105))


def _render_frame(result: Any, step_index: int, *, width: int, height: int) -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - guarded by real-llm extra
        raise RuntimeError("Animated GIF output requires Pillow; install SeedMark with the real-llm extra") from exc

    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": _font(ImageFont, 30, bold=True),
        "subtitle": _font(ImageFont, 16),
        "section": _font(ImageFont, 18, bold=True),
        "sentence": _font(ImageFont, 25),
        "small": _font(ImageFont, 13),
        "small_bold": _font(ImageFont, 13, bold=True),
        "tiny": _font(ImageFont, 11),
        "metric": _font(ImageFont, 27, bold=True),
    }

    step = result.trace[step_index]
    completed, current = sentence_parts(result, step_index)
    card = (255, 255, 255)
    border = (226, 232, 240)

    draw.text((44, 30), "SeedMark · live token generation", font=fonts["title"], fill=(23, 32, 51))
    draw.text(
        (45, 72),
        f"{result.model_name}   ·   step {step.position}/{len(result.trace)}   ·   top-k {result.top_k}   ·   strength {result.strength:g}",
        font=fonts["subtitle"],
        fill=(100, 116, 139),
    )

    # Sentence card: the key educational view requested by the project.
    draw.rounded_rectangle((40, 108, width - 40, 300), radius=20, fill=card, outline=border, width=1)
    draw.text((62, 129), "Sentence generation", font=fonts["section"], fill=(30, 41, 59))
    draw.text((62, 157), "The token being added right now is highlighted.", font=fonts["small"], fill=(100, 116, 139))
    _draw_sentence(
        draw,
        completed=completed,
        current=current,
        x=63,
        y=191,
        max_width=width - 126,
        font=fonts["sentence"],
        line_height=36,
        max_lines=3,
    )

    # Candidate microscope card.
    draw.rounded_rectangle((40, 320, 700, height - 42), radius=20, fill=card, outline=border, width=1)
    _draw_candidate_rows(draw, step, x=62, y=370, width=610, fonts=fonts)

    # Detection card.
    draw.rounded_rectangle((720, 320, width - 40, height - 42), radius=20, fill=card, outline=border, width=1)
    draw.text((744, 345), "Selected token", font=fonts["section"], fill=(30, 41, 59))
    token_label = _display_token(step.chosen_token_text)
    if len(token_label) > 22:
        token_label = token_label[:21] + "…"
    draw.text((744, 381), token_label, font=fonts["metric"], fill=(190, 24, 93))
    draw.text((744, 420), f"token id   {step.chosen_token_id}", font=fonts["small"], fill=(71, 85, 105))
    draw.text((744, 444), f"base p     {step.chosen_base_probability:.3f}", font=fonts["small"], fill=(71, 85, 105))
    draw.text((744, 468), f"marked p   {step.chosen_generation_probability:.3f}", font=fonts["small"], fill=(71, 85, 105))
    draw.text((744, 492), f"PRF score  {step.chosen_watermark_score:.3f}", font=fonts["small"], fill=(71, 85, 105))
    draw.text((744, 529), f"cumulative z = {step.cumulative_z:.2f}", font=fonts["small_bold"], fill=(30, 41, 59))
    _draw_z_gauge(draw, z=step.cumulative_z, threshold=result.detection.threshold_z, x=744, y=558, width=285, font=fonts["tiny"])

    draw.text((45, height - 27), "Real Qwen logits guide generation · detector later needs only token IDs + seed + secret key", font=fonts["tiny"], fill=(100, 116, 139))
    return image


def write_generation_gif(
    output_path: Path,
    result: Any,
    *,
    frame_ms: int = 650,
    width: int = 1100,
    height: int = 650,
) -> Path:
    """Create an animated GIF of a generation trace with the current token highlighted."""
    if frame_ms < 100:
        raise ValueError("frame_ms must be >= 100")
    if not result.trace:
        raise ValueError("cannot animate an empty generation trace")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - guarded by real-llm extra
        raise RuntimeError("Animated GIF output requires Pillow; install SeedMark with the real-llm extra") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_render_frame(result, index, width=width, height=height) for index in range(len(result.trace))]
    palette = getattr(Image, "Palette", None)
    adaptive = palette.ADAPTIVE if palette is not None else Image.ADAPTIVE
    gif_frames = [frame.convert("P", palette=adaptive, colors=128) for frame in frames]
    durations = [frame_ms] * len(gif_frames)
    durations[-1] = max(1800, frame_ms * 3)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output_path
