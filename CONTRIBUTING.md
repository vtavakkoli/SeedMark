# Contributing

SeedMark welcomes reproducible contributions that preserve the distinction between a teaching prototype and production watermark systems.

## Development

1. Use Python 3.13.
2. Install with `python -m pip install -e .` or set `PYTHONPATH=src`.
3. Run `python -m unittest discover -s tests -v`.
4. Regenerate a report with `seedmark experiment --output-dir results/run`.
5. Record random seeds and configuration for new experiments.

## Scientific changes

For changes that alter the detector or generator, include the mathematical assumption being changed, a matched unwatermarked control, a reproducible seed, at least one false-positive metric, tests, and an update to `docs/limitations.md` when claims or boundaries change.

Do not describe SeedMark results as evidence about a vendor's private production watermark unless the experiment actually evaluates that system with an authorized detector.
