"""Chat-specific presentation layer for the real-Qwen SeedMark report.

The core report renderer stays reusable for arbitrary HF traces. This wrapper
adds the semantics of the public chat demo: a user question, two assistant
answers, and explicit metadata showing that Qwen's native chat template was used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import html
import json
import re

from .reporting import write_qwen_report


def write_chat_report(
    output_dir: Path,
    marked: Any,
    control: Any,
    *,
    question: str,
    system_prompt: str,
    assets: dict[str, str] | None = None,
) -> Path:
    """Write the standard scientific report and make its presentation chat-first."""
    report_path = write_qwen_report(output_dir, marked, control, assets=assets)
    document = report_path.read_text(encoding="utf-8")

    safe_question = html.escape(question.strip())
    safe_system = html.escape(system_prompt.strip())
    chat_panel = f'''<section class="card prompt">
  <div class="kicker">Real chat input · Qwen native chat template</div>
  <h2>User asks the AI</h2>
  <div style="display:flex;justify-content:flex-end;margin:14px 0 16px">
    <div style="max-width:78%;background:#275073;color:white;padding:13px 16px;border-radius:18px 18px 4px 18px;font-size:17px;box-shadow:0 8px 22px rgba(39,80,115,.16)">{safe_question}</div>
  </div>
  <p style="margin:0;color:#64758a">Qwen receives this as a real <strong>user</strong> message. The system instruction requests a short plain-language article; SeedMark is applied only while the <strong>assistant</strong> answer is generated.</p>
  <details style="margin-top:12px"><summary style="cursor:pointer;font-weight:800;color:#275073">Show system instruction</summary><code class="block" style="margin-top:9px">{safe_system}</code></details>
</section>'''

    document, count = re.subn(
        r'<section class="card prompt">.*?</section>',
        chat_panel,
        document,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("could not locate the prompt card in generated Qwen report")

    replacements = {
        "One prompt. Two generations. Clear detection contrast.":
            "One chat question. Two assistant answers. Clear detection contrast.",
        "The same Qwen prompt is generated with and without SeedMark.":
            "The same Qwen chat request is answered with and without SeedMark.",
        ">Watermarked text<": ">Assistant · watermarked<",
        ">Control text · without watermark<": ">Assistant · control<",
        "Default demonstration prompt": "Real chat input",
    }
    for old, new in replacements.items():
        document = document.replace(old, new)

    report_path.write_text(document, encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mode"] = "chat"
    summary["chat"] = {
        "question": question.strip(),
        "system_prompt": system_prompt.strip(),
        "native_chat_template": True,
        "thinking_disabled": True,
        "assistant_only_outputs": True,
        "seed_source": "first normalized word of the user question",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path
