FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

COPY src ./src
COPY tests ./tests
COPY examples ./examples

ENTRYPOINT ["python", "-m", "seedmark.cli"]
CMD ["experiment", "--output-dir", "/workspace/results/run"]
