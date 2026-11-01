FROM mcr.microsoft.com/playwright/python:v1.56.0-jammy

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
    && scrapling install

COPY backend /app/backend
COPY keyword-nlp-output /app/keyword-nlp-output
COPY frontend/build /app/frontend/build

ENV FRONTEND_DIR=/app/frontend/build
ENV KEYWORD_NLP_OUTPUT_DIR=/app/keyword-nlp-output
ENV PORT=8010

EXPOSE 8010

CMD ["python", "/app/backend/main.py"]
