# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: build TA-Lib C library + Python wheel
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile TA-Lib from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and build TA-Lib C library (v0.4.0)
WORKDIR /tmp
RUN wget -q https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install

# Install Python packages into a clean prefix
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r /tmp/requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: lean runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Copy compiled TA-Lib shared library
COPY --from=builder /usr/local/lib/libta_lib* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib  /usr/local/include/ta-lib

# Refresh linker cache so Python can find libta_lib.so
RUN ldconfig

# Copy installed Python packages
COPY --from=builder /install /usr/local

# App source
WORKDIR /app
COPY trading_agent.py api.py ./

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# gunicorn: 2 workers, 120 s timeout (Yahoo Finance fetch can be slow)
CMD ["gunicorn", "api:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
