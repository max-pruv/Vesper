FROM python:3.12-slim

WORKDIR /app

# Build deps for ta (technical analysis) and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

CMD ["python", "-m", "vesper.main"]
