"""Demonstrate the central SeedMark claim: detection needs no LM distribution."""

from seedmark import ToyBigramLM, WatermarkConfig, detect_tokens, generate_text

config = WatermarkConfig(secret_key="example-key", strength=1.5, top_k=8, threshold_z=3.0)
result = generate_text(ToyBigramLM(), first_word="research", length=100, config=config, watermarked=True, rng_seed=7)

# At detection time the language model and its probability table are not supplied.
detection = detect_tokens(list(result.tokens), secret_key="example-key", first_word=result.tokens[0], threshold_z=3.0)

print(result.text)
print(f"z={detection.z_score:.3f}, p={detection.p_value_one_sided:.3g}, detected={detection.detected}")
