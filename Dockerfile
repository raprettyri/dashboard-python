FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 8002

CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8002", "api.index:app"]