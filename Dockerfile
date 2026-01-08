FROM python:3.9-slim

WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source code
COPY backend/ ./

EXPOSE 8000

# Run the FastAPI server with proper settings
CMD ["python", "-u", "-m", "uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
