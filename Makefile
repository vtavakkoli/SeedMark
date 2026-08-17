.PHONY: test experiment docker clean

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

experiment:
	PYTHONPATH=src python -m seedmark.cli experiment --output-dir results/run --trials 300 --length 80

docker:
	docker compose up --build

clean:
	rm -rf results/run results/ci
