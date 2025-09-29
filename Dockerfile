# Multi-stage Docker build for MEWS
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash mews
USER mews
WORKDIR /home/mews/app

# Copy requirements first for better caching
COPY --chown=mews:mews requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY --chown=mews:mews requirements-dev.txt ./
RUN pip install --user --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY --chown=mews:mews . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Expose ports
EXPOSE 8501

# Default command for development
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=mews:mews src/ ./src/
COPY --chown=mews:mews main.py streamlit_app.py ./
COPY --chown=mews:mews config/ ./config/

# Create necessary directories
RUN mkdir -p data models outputs logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Production command
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
