# Small official Python image
FROM python:3.12-slim

# No .pyc files, and print logs straight away (useful in `docker logs` and on Render)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# 1) Install dependencies first. This layer is cached, so rebuilds are fast
#    unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy the application code and the trained model
COPY app ./app
COPY model ./model

# Run as a normal user instead of root
RUN useradd --create-home appuser
USER appuser

# Local default is 8000. Render sets PORT itself, so the app listens on whatever PORT is.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
