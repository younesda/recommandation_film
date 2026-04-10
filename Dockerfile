FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY scripts/run_api.py ./scripts/run_api.py
COPY hosting ./hosting

EXPOSE 8000

CMD ["python", "scripts/run_api.py"]
