FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "scikit-learn>=1.3.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0"

# Install the package itself (enables relative imports)
RUN pip install --no-cache-dir -e .

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
USER user

# Expose port that matches app_port in README.md
EXPOSE 7860

# Start the environment server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
