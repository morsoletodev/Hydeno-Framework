FROM python:3.12 AS builder
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y git
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .
COPY src/ ./src/


FROM python:3.12-slim AS runner
WORKDIR /app/DataPipeline

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src/ /app/DataPipeline/src
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/DataPipeline:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT [ "python", "-m", "src" ]
