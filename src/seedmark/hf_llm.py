"""Real-LLM SeedMark adapter for Hugging Face Qwen models.

The watermark remains intentionally simple: the normalized first word supplies a
public seed, while HMAC-SHA256 over (seed, position, token-id) supplies a keyed
pseudorandom score. Generation tilts a real model's top-k probabilities toward
high-scoring token IDs. Detection needs the tokenizer and secret key, but never
the model logits or next-token probability distribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import html
import json
import math
from pathlib import Path
from typing import Any

from .core import DetectionResult, first_word_seed, normalize_word

_TWO64 = float(1 << 64)
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
QUALITY_MODEL = "Qwen/Qwen3.5-2B"


@dataclass(frozen=True, slots=True)
class HFCandidateTrace:
    token_id: int
    token_text: str
    base_probability: float
    generation_probability: float
    watermark_score: float
    chosen: bool


@dataclass(frozen=True, slots=True)
class HFStepTrace:
    position: int
    chosen_token_id: int
    chosen_token_text: str
    chosen_base_probability: float
    chosen_generation_probability: float
    chosen_watermark_score: float
    cumulative_z: float
    candidates: tuple[HFCandidateTrace, ...]


@dataclass(frozen=True, slots=True)
class HFGenerationResult:
    model_name: str
    prompt: str
    first_word: str
    text: str
    continuation: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    watermarked: bool
    temperature: float
    top_k: int
    strength: float
    rng_seed: int
    detection: DetectionResult
    trace: tuple[HFStepTrace, ...]


def token_id_score(secret_key: str, first_word: str, position: int, token_id: int) -> float:
    """Map a model token id to a reproducible U[0,1) keyed score."""
    if not secret_key:
        raise ValueError("secret_key must not be empty")
    if position < 1:
        raise ValueError("position must be >= 1")
    if token_id < 0:
        raise ValueError("token_id must be >= 0")
    seed = first_word_seed(first_word)
    message = seed + position.to_bytes(8, "big") + token_id.to_bytes(8, "big")
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _TWO64


def detect_token_ids(
    token_ids: list[int] | tuple[int, ...],
    *,
    secret_key: str,
    first_word: str,
    threshold_z: float = 3.0,
) -> DetectionResult:
    """Detect SeedMark correlation from observed token IDs; no logits are needed."""
    if threshold_z <= 0:
        raise ValueError("threshold_z must be > 0")
    if not token_ids:
        return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)
    observed = [
        token_id_score(secret_key, first_word, position, token_id)
        for position, token_id in enumerate(token_ids, start=1)
    ]
    n = len(observed)
    mean_score = sum(observed) / n
    z_score = (sum(observed) - 0.5 * n) / math.sqrt(n / 12.0)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return DetectionResult(n, mean_score, z_score, p_value, threshold_z, z_score >= threshold_z)


def _optional_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "Real-LLM support is optional. Install it with: pip install -e '.[real-llm]'"
        ) from exc
    return torch, AutoProcessor, AutoModelForMultimodalLM


def _choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenSeedMark:
    """Load one Qwen3.5 model and generate marked/unmarked matched samples."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        torch, AutoProcessor, AutoModelForMultimodalLM = _optional_stack()
        self.torch = torch
        self.model_name = model_name
        self.device = _choose_device(torch, device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("The selected processor does not expose a tokenizer")
        self.model = AutoModelForMultimodalLM.from_pretrained(model_name, torch_dtype="auto")
        self.model.to(self.device)
        if self.device == "cpu":
            # Float32 is slower/larger but avoids CPU kernels that may not support BF16.
            self.model.float()
        self.model.eval()

    def _encode(self, text: str) -> tuple[Any, Any]:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def generate(
        self,
        *,
        prompt: str = "Research is",
        max_new_tokens: int = 64,
        secret_key: str = "seedmark-demo-key",
        strength: float = 1.5,
        top_k: int = 20,
        temperature: float = 1.0,
        threshold_z: float = 3.0,
        rng_seed: int = 20260817,
        watermarked: bool = True,
    ) -> HFGenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if top_k < 2:
            raise ValueError("top_k must be >= 2")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if strength < 0:
            raise ValueError("strength must be >= 0")

        torch = self.torch
        seed_word = normalize_word(prompt)
        input_ids, attention_mask = self._encode(prompt)
        prompt_token_ids = tuple(int(x) for x in input_ids[0].detach().cpu().tolist())
        generated_ids: list[int] = []
        trace: list[HFStepTrace] = []
        scores_seen: list[float] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng_seed)

        with torch.inference_mode():
            for position in range(1, max_new_tokens + 1):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :].float() / temperature
                k = min(top_k, int(logits.shape[-1]))
                values, ids = torch.topk(logits, k=k)
                base = torch.softmax(values, dim=-1).detach().cpu()
                candidate_ids = [int(x) for x in ids.detach().cpu().tolist()]
                keyed_scores = [
                    token_id_score(secret_key, seed_word, position, token_id)
                    for token_id in candidate_ids
                ]
                score_tensor = torch.tensor(keyed_scores, dtype=torch.float32)
                tilted = base * torch.exp(strength * (2.0 * score_tensor - 1.0))
                marked = tilted / tilted.sum()
                generation = marked if watermarked else base
                chosen_index = int(torch.multinomial(generation, 1, generator=generator).item())
                chosen_id = candidate_ids[chosen_index]
                chosen_score = keyed_scores[chosen_index]
                scores_seen.append(chosen_score)
                cumulative_n = len(scores_seen)
                cumulative_z = (sum(scores_seen) - 0.5 * cumulative_n) / math.sqrt(cumulative_n / 12.0)

                candidate_trace = []
                for idx, token_id in enumerate(candidate_ids):
                    candidate_trace.append(HFCandidateTrace(
                        token_id=token_id,
                        token_text=self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                        base_probability=float(base[idx]),
                        generation_probability=float(generation[idx]),
                        watermark_score=keyed_scores[idx],
                        chosen=idx == chosen_index,
                    ))
                trace.append(HFStepTrace(
                    position=position,
                    chosen_token_id=chosen_id,
                    chosen_token_text=self.tokenizer.decode([chosen_id], clean_up_tokenization_spaces=False),
                    chosen_base_probability=float(base[chosen_index]),
                    chosen_generation_probability=float(generation[chosen_index]),
                    chosen_watermark_score=chosen_score,
                    cumulative_z=cumulative_z,
                    candidates=tuple(candidate_trace),
                ))
                generated_ids.append(chosen_id)
                next_id = torch.tensor([[chosen_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones_like(next_id)), dim=1)
                eos_ids = self.tokenizer.eos_token_id
                if eos_ids is not None and chosen_id == int(eos_ids):
                    break

        continuation = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        full_text = prompt + continuation
        detection = detect_token_ids(
            generated_ids,
            secret_key=secret_key,
            first_word=seed_word,
            threshold_z=threshold_z,
        )
        return HFGenerationResult(
            model_name=self.model_name,
            prompt=prompt,
            first_word=seed_word,
            text=full_text,
            continuation=continuation,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=tuple(generated_ids),
            watermarked=watermarked,
            temperature=temperature,
            top_k=top_k,
            strength=strength,
            rng_seed=rng_seed,
            detection=detection,
            trace=tuple(trace),
        )


def detect_text_with_tokenizer(
    *,
    model_name: str,
    text: str,
    prompt: str,
    secret_key: str,
    threshold_z: float = 3.0,
) -> DetectionResult:
    """Retokenize text using only the public tokenizer, then run the detector.

    No model weights are loaded. For exact reproducibility, the text must preserve the
    original generated tokenization; the saved generated_token_ids are authoritative.
    """
    _, AutoProcessor, _ = _optional_stack()
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("text does not tokenize with the supplied prompt as an exact prefix")
    generated_ids = full_ids[len(prompt_ids):]
    return detect_token_ids(
        generated_ids,
        secret_key=secret_key,
        first_word=normalize_word(prompt),
        threshold_z=threshold_z,
    )


def _result_dict(result: HFGenerationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["detection"] = asdict(result.detection)
    return payload


def write_qwen_report(output_dir: Path, marked: HFGenerationResult, control: HFGenerationResult) -> None:
    """Write JSON traces plus a standalone interactive HTML token microscope."""
    output_dir.mkdir(parents=True, exist_ok=True)
    marked_data, control_data = _result_dict(marked), _result_dict(control)
    (output_dir / "generated_watermarked.txt").write_text(marked.text, encoding="utf-8")
    (output_dir / "generated_control.txt").write_text(control.text, encoding="utf-8")
    (output_dir / "watermarked-trace.json").write_text(json.dumps(marked_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "control-trace.json").write_text(json.dumps(control_data, indent=2, ensure_ascii=False), encoding="utf-8")
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
        "note": "Real Qwen logits were used for generation; detection used only token IDs, first-word seed and secret key.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    payload = json.dumps(marked_data, ensure_ascii=False).replace("</", "<\\/")
    safe_model = html.escape(marked.model_name)
    safe_prompt = html.escape(marked.prompt)
    doc = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SeedMark Qwen report</title><style>
:root{{--bg:#f8fafc;--ink:#172033;--muted:#64748b;--pink:#ec4899;--violet:#7c3aed;--line:#e2e8f0}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#fff,#f8fafc 55%,#fdf2f8);font:15px/1.5 system-ui;color:var(--ink)}}main{{max-width:1180px;margin:auto;padding:28px 18px 70px}}h1{{font-size:clamp(30px,5vw,52px);letter-spacing:-.04em;margin:.2em 0}}.lead{{max-width:900px;color:#475569}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}.card{{background:white;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 10px 30px #0f172a0d}}.metric b{{display:block;font-size:28px}}.metric small{{color:var(--muted);text-transform:uppercase}}.toolbar{{display:flex;gap:10px;align-items:center}}button{{border:0;border-radius:10px;background:var(--violet);color:white;padding:9px 14px;font-weight:700}}input{{flex:1}}.candidate{{display:grid;grid-template-columns:190px 1fr 90px;gap:10px;align-items:center;margin:8px 0}}.bar{{height:9px;background:#e2e8f0;border-radius:20px;overflow:hidden;margin:3px 0}}.bar i{{display:block;height:100%;background:var(--violet)}}.bar.mark i{{background:var(--pink)}}.chosen{{font-weight:800;color:#be185d}}code{{background:#f1f5f9;padding:2px 6px;border-radius:6px}}.note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px;border-radius:9px}}pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:12px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr 1fr}}.candidate{{grid-template-columns:120px 1fr 70px}}}}</style></head><body><main>
<div style="font-weight:800;color:#7c3aed;text-transform:uppercase;letter-spacing:.12em;font-size:12px">SeedMark · real LLM experiment</div><h1>Qwen token probabilities + keyed pseudorandom watermark</h1><p class="lead">Model <code>{safe_model}</code> produced real next-token logits. Prompt: <code>{safe_prompt}</code>. The detector never receives those logits: it scores only the observed token IDs against the first-word seed and secret key.</p>
<div class="grid"><div class="card metric"><small>Marked z</small><b>{marked.detection.z_score:.2f}</b></div><div class="card metric"><small>Control z</small><b>{control.detection.z_score:.2f}</b></div><div class="card metric"><small>Top-k</small><b>{marked.top_k}</b></div><div class="card metric"><small>Tokens</small><b>{len(marked.generated_token_ids)}</b></div></div>
<section class="card"><h2>Watermarked output</h2><pre>{html.escape(marked.text)}</pre></section>
<section class="card"><h2>Interactive token microscope</h2><div class="toolbar"><button id="play">▶ Play</button><input id="slider" type="range" min="0" value="0"><b id="step"></b><span id="z"></span></div><p id="explain" class="note"></p><div id="cand"></div></section>
<section class="card"><h2>What this proves</h2><p>Generation uses a real LLM distribution. Detection does not: it reconstructs the keyed score from <code>(first word, position, token ID)</code>. This is a research teaching baseline, not SynthID, Anthropic's production system, or C2PA.</p></section>
<script>const D={payload},T=D.trace,s=document.getElementById('slider');s.max=Math.max(0,T.length-1);let timer=null;function esc(x){{return String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}function pct(x){{return Math.min(100,x*100).toFixed(2)+'%'}}function render(i){{const a=T[i];if(!a)return;s.value=i;step.textContent=`step ${{a.position}}/${{T.length}}`;z.textContent=`cumulative z=${{a.cumulative_z.toFixed(2)}}`;explain.innerHTML=`chosen token <b>${{esc(a.chosen_token_text)}}</b> · id <b>${{a.chosen_token_id}}</b> · keyed score <b>${{a.chosen_watermark_score.toFixed(3)}}</b>`;cand.innerHTML=a.candidates.map(c=>`<div class="candidate ${{c.chosen?'chosen':''}}"><span>${{esc(c.token_text)}} <small>#${{c.token_id}}</small>${{c.chosen?' ✓':''}}</span><div><div class="bar"><i style="width:${{pct(c.base_probability)}}"></i></div><div class="bar mark"><i style="width:${{pct(c.generation_probability)}}"></i></div></div><span>u=${{c.watermark_score.toFixed(3)}}</span></div>`).join('')}}s.oninput=e=>render(+e.target.value);play.onclick=()=>{{if(timer){{clearInterval(timer);timer=null;return}}timer=setInterval(()=>render((+s.value+1)%T.length),450)}};render(0);</script></main></body></html>'''
    (output_dir / "report.html").write_text(doc, encoding="utf-8")
