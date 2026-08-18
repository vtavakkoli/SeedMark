# Limitations and scientific boundaries

SeedMark is a transparent statistical teaching prototype, not a production provenance system.

1. **Not a vendor implementation.** It is not Anthropic's production watermark, Google SynthID-Text, C2PA, or a compatible detector.
2. **Distortionary sampling.** Exponential probability tilting changes the conditional token distribution.
3. **Toy word-level baseline.** The original baseline uses a toy model; real LLM tokenization and probability geometry are much more complex.
4. **Approximate null.** The z-test treats keyed scores as approximately independent uniform values; representative unwatermarked calibration is still required.
5. **Short text has low power.** Few tokens provide little statistical evidence.
6. **Editing can weaken the carrier signal.** Insertions, deletions, replacements, and reordering remove or alter scored token IDs even when the semantic key remains stable.
7. **Original-mode first-word fragility is intentional.** In the baseline construction, changing the seed word invalidates the remaining score stream.
8. **Semantic mode improves robustness, not invulnerability.** Complete-answer mode derives one coarse bucket from the answer semantics; paragraph mode re-keys only at paragraph boundaries. Strong paraphrasing or semantic drift can still move text into another bucket.
9. **Complete-answer mode is multi-pass.** It generates an ordinary semantic draft, derives a bucket, and generates the marked final answer. The final answer is accepted only when it maps to the same bucket. This costs additional inference and can require retries.
10. **Answer-key stabilization is not guaranteed.** Fine bucket counts or stronger watermark distortion may make the final answer cross a semantic boundary. SeedMark raises an error after the configured maximum attempts instead of silently returning an unreconstructable key.
11. **Paragraph mode depends on paragraph structure.** Blank-line insertion, deletion, paragraph splitting, or merging can change paragraph-level re-synchronisation. Sentence punctuation alone does not re-key the watermark.
12. **Token occurrence counters are more edit-tolerant, not edit-invariant.** Inserting an unrelated token does not shift every later score, but adding/removing the same token ID changes later occurrence numbers for that token.
13. **Semantic detection needs matching configuration.** The detector must use the same semantic encoder, bucket count, semantic scope, tokenizer, and secret key; paragraph mode also needs the same paragraph-context setting.
14. **Semantic bucket boundaries are not guaranteed invariants.** A meaning-preserving edit can still cross a nearest-anchor boundary, especially when the recorded bucket margin is small.
15. **The key matters.** Without a secret key, outsiders could predict favorable carrier tokens and attempt forgery.
16. **No semantic provenance claim.** Detection means correlation with this keyed rule under the tested assumptions, not proof of authorship, truth, or model identity.
17. **Thresholds require calibration.** A deployed detector would require representative domain, length, language, decoding, editing, and adversarial calibration.
18. **Adaptive attacks are not solved.** Repeated detector access, model-assisted rewriting, or optimization against the detector may reduce or forge the signal.

For research beyond the teaching demo, benchmark multiple generator models, languages, answer lengths, semantic encoders, bucket counts, paraphrase/edit/translation attacks, answer-key stabilization rates, quality metrics, ROC/AUC, false-positive rates, and published watermarking baselines.

See [`semantic-watermark.md`](semantic-watermark.md) for the experimental complete-answer and paragraph semantic-key designs and their proposed robustness evaluation.
