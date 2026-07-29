FROM python:3.14-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv only in the build stage.
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# Install the actual project using uv
RUN uv sync --frozen --no-dev

FROM python:3.14-slim AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ONNX Runtime requires the OpenMP runtime, but compilers and headers stay out
# of the final image to reduce image pull and scale-out time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

# Schema migration is an explicit deployment job (the Compose `migrate`
# service), so application replicas can start concurrently without contending
# on Alembic locks.
# Containers must listen on every interface so sibling services (the frontend
# reverse proxy) can reach the API over the Compose network.
CMD ["python", "main.py", "--host", "0.0.0.0"]
