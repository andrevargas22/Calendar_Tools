FROM python:3.10-slim
WORKDIR /app

# common/google_calendar.py relies on the container's system timezone (bare
# astimezone()); Cloud Run defaults to UTC, so fix it here instead of there.
ENV TZ=America/Sao_Paulo
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY agent/requirements.txt agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

COPY common/ common/
COPY agent/ agent/

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# --workers 1: session state in telegram_bot.py is an in-process dict, not shared across workers.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 0 agent.src.telegram_bot:app"]
