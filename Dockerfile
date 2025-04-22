FROM python:3.11.7-slim
# Set working directory
WORKDIR /app
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1
# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Copy requirements first for better caching
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY . .
# Create workspace directory for Snowflake key
RUN mkdir -p /workspace && chmod 755 /workspace
# Expose the port the app runs on
EXPOSE 8080
# Command to run the application with increased timeout
CMD ["gunicorn", "--config", "gunicorn_config.py", "main:app"]
