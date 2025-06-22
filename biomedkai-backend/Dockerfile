# Use Python 3.11 slim image as base for better performance
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies required for biomedical libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libpq-dev \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash biomedkai && \
    chown -R biomedkai:biomedkai /app
USER biomedkai

# Copy requirements first for better Docker layer caching
COPY --chown=biomedkai:biomedkai requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Add local Python packages to PATH
ENV PATH="/home/biomedkai/.local/bin:${PATH}"

# Copy the application code
COPY --chown=biomedkai:biomedkai . .

# Create necessary directories
RUN mkdir -p logs data models cache

# Expose the port (adjust based on your application)
EXPOSE 8000

# Health check for the application
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command
CMD ["python", "start.py"]