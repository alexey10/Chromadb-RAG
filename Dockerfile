# Use official Python base
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and app
COPY pyproject.toml requirements.txt ./
# If you don't have a requirements file, create one listing:
# fastapi uvicorn[standard] chromadb openai pandas beautifulsoup4 requests
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
