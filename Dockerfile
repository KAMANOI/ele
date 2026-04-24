FROM python:3.11-slim

# ── System deps + ClamAV antivirus ───────────────────────────────────────────
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        clamav \
        clamav-freshclam \
    && rm -rf /var/lib/apt/lists/*

# Update virus signatures during image build
# (signatures are also refreshed on each container startup via CMD)
RUN freshclam --quiet || true

# ── Python app ────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

# ── Startup: refresh signatures then launch ───────────────────────────────────
EXPOSE 8000
CMD freshclam --quiet 2>/dev/null || true && \
    uvicorn ele.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
