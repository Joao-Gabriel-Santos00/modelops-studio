FROM python:3.13-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements from build context (services/model_server)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy app code (docker-compose mounts repo over /app during development)
COPY ./services/model_server/ .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
