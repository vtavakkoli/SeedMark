# Limitations and scientific boundaries

SeedMark is a transparent statistical teaching prototype, not a production provenance system.

1. **Not a vendor implementation.** It is not Anthropic's production watermark, Google SynthID-Text, C2PA, or a compatible detector.
2. **Distortionary sampling.** Exponential probability tilting changes the conditional token distribution.
3. **Toy word-level baseline.** The original baseline uses a toy model; real LLM tokenization and probability geometry are much more complex.
4. **Approximate null.** The z-test treats keyed scores as independent uniform values; the repository therefore also reports an empirical unwatermarked baseline.
5. **Short text has low power.** Few tokens provide little statistical evidence.
6. **Editing can weaken the token signal.** Insertions, deletions, replacements, and reordering alter the observed carrier tokens.
7. **Original-mode first-word fragility is intentional.** In the baseline construction, changing the seed word invalidates the remaining score stream.
8. **Semantic mode improves re-synchronisation, not invulnerability.** It derives later keys from coarse semantic buckets and resets offsets at sentence boundaries, but sufficiently strong paraphrasing, translation, sentence splitting/merging, or semantic drift can still change buckets and remove evidence.
9. **Semantic detection needs the same encoder configuration.** The detector must use the same semantic encoder model, bucket count, context-window setting, tokenizer, and secret key as generation.
10. **Semantic bucket boundaries are not guaranteed invariants.** A meaning-preserving edit can still cross a nearest-anchor boundary, especially when the recorded bucket margin is small.
11. **The key matters.** Without a secret key, outsiders can predict favorable tokens and forge this demonstration watermark.
12. **No semantic provenance claim.** Detection means correlation with this keyed rule under this test, not proof of authorship, truth, or model identity.
13. **Thresholds require calibration.** A deployed detector would require representative domain, length, language, decoding, editing, and adversarial calibration.
14. **Adaptive attacks are not solved.** Repeated detector access, model-assisted rewriting, or optimization against the detector may reduce or forge the signal.

For research beyond the teaching demo, benchmark tokenizer-level generation across multiple models, repeated-context handling, multiple keys, domain shift, multilingual text, paraphrase/edit/translation robustness, calibration by length, quality metrics, detector ROC/AUC, false-positive rates, and published watermarking baselines.

See [`semantic-watermark.md`](semantic-watermark.md) for the experimental semantic self-key design and its proposed robustness evaluation.
