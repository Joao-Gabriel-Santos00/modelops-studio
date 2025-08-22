FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (better caching)
COPY ../../requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy trainer code
COPY . .

# Default command (can be overridden in docker-compose)
CMD ["python", "train.py"]
