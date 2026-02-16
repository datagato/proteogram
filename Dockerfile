# syntax=docker/dockerfile:1.6

############################
# Builder: install deps with uv
############################
FROM python:3.11-slim AS builder

# System deps (add more if you need: git, build-essential, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install uv
# (uv is a single binary; pip install also works, but this is usually faster/cleaner)
RUN pip install --no-cache-dir uv

# Copy only dependency metadata first for better layer caching
COPY pyproject.toml ./
# If you have a lockfile, copy it too:
COPY uv.lock ./

# Create a virtualenv and sync deps from lockfile
# --frozen: fail if lockfile doesn't match pyproject
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --active --frozen --no-dev

############################
# Runtime: copy venv + code
############################
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy repo contents
COPY . /app

# Make sure imports work from repo root
ENV PYTHONPATH=/app

# Default help
CMD ["python", "-c", "print('Container ready. Try: python scripts/measure_similarity.py')"]
