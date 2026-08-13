FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Command to launch FastAPI server on port 7860
CMD ["uvicorn", "VeriDex_WebApp.app:app", "--host", "0.0.0.0", "--port", "7860"]
