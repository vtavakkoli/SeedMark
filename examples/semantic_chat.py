"""Minimal complete-answer semantic watermark example.

Requires: pip install -e '.[real-llm]'
"""

from seedmark.semantic_chat import SemanticChatQwenSeedMark


model = SemanticChatQwenSeedMark(
    model_name="Qwen/Qwen3.5-0.8B",
    semantic_device="cpu",
)

question = "What is AI?"
common = {
    "question": question,
    "semantic_scope": "answer",
    "bucket_count": 32,
    "max_answer_passes": 4,
}
marked = model.generate(**common, watermarked=True)
control = model.generate(**common, watermarked=False)

print("=== Watermarked answer ===")
print(marked.text)
print(
    f"bucket={marked.semantic_bucket} margin={marked.semantic_margin:.4f} "
    f"attempts={marked.answer_key_attempts} detected={marked.detection.detected} "
    f"z={marked.detection.z_score:.3f}"
)

print("\n=== Control answer ===")
print(control.text)
print(
    f"detected={control.detection.detected} "
    f"z={control.detection.z_score:.3f}"
)
