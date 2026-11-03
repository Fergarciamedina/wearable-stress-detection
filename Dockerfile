# Multi-arch CPU-compatible base image (Python 3.11 Slim on Debian Bookworm)
FROM python:3.11-slim

# Prevent interactive prompts and Python buffer delays
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install minimal build tools and system dependencies for scipy/neurokit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, tests, and Makefile
COPY pyproject.toml .
COPY Makefile .
COPY reference/ ./reference/
COPY tests/ ./tests/
COPY src/ ./src/

# Install the package locally in editable/standard mode
RUN pip install --no-cache-dir -e .

# Expose Streamlit default port
EXPOSE 8501

# Default environment variables for container data mapping
ENV WEARABLE_DATA_DIR=/app/data \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Healthcheck to verify dashboard accessibility
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit dashboard by default
ENTRYPOINT ["streamlit", "run", "src/act/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
