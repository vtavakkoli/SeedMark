"""Minimal semantic self-key watermark example.

Requires: pip install -e '.[real-llm]'
"""

from seedmark.semantic_chat import SemanticChatQwenSeedMark


model = SemanticChatQwenSeedMark(
    model_name="Qwen/Qwen3.5-0.8B",
    semantic_device="cpu",
)

question = "What is AI?"
marked = model.generate(question=question, watermarked=True)
control = model.generate(question=question, watermarked=False)

print("=== Watermarked answer ===")
print(marked.text)
print(
    f"detected={marked.detection.detected} "
    f"z={marked.detection.z_score:.3f} "
    f"p={marked.detection.p_value_one_sided:.3g}"
)

print("\n=== Control answer ===")
print(control.text)
print(
    f"detected={control.detection.detected} "
    f"z={control.detection.z_score:.3f} "
    f"p={control.detection.p_value_one_sided:.3g}"
)
