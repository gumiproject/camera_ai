FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "6000", "app:app"]