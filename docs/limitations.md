# Limitations and scientific boundaries

SeedMark is a transparent statistical teaching prototype, not a production provenance system.

1. **Not a vendor implementation.** It is not Anthropic's production watermark, Google SynthID-Text, C2PA, or a compatible detector.
2. **Distortionary sampling.** Exponential probability tilting changes the conditional token distribution.
3. **Toy word-level model.** Real LLM tokenization and probability geometry are much more complex.
4. **Approximate null.** The z-test treats keyed scores as independent uniform values; the repository therefore also reports an empirical unwatermarked baseline.
5. **Short text has low power.** Few tokens provide little statistical evidence.
6. **Editing can weaken the signal.** Insertions, deletions, replacements, and reordering change positions or scores.
7. **First-word fragility is intentional.** Changing the first word changes the seed and invalidates the remaining score stream.
8. **The key matters.** Without a secret key, outsiders can predict favorable tokens and forge this demonstration watermark.
9. **No semantic provenance claim.** Detection means correlation with this keyed rule under this test, not proof of authorship or truth.
10. **Thresholds require calibration.** A deployed detector would require representative domain, length, language, decoding, editing, and adversarial calibration.

For research beyond the teaching demo, add real tokenizer-level distributions, repeated-context handling, multiple keys, domain-shift experiments, paraphrase/edit robustness, calibration by length, quality metrics, and published watermarking baselines.
