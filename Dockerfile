FROM python:3.10-slim
WORKDIR /app

COPY agent/requirements.txt agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

COPY common/ common/
COPY agent/ agent/

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# --workers 1 is required, not just a performance knob: telegram_bot.py
# keeps per-chat session state (including pending confirmations) in a plain
# module-level dict, unsynchronized across processes. More workers would
# silently scatter a single conversation's state across independent dicts.
# --timeout 0 disables gunicorn's own worker timeout; Cloud Run's request
# timeout is the real ceiling.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 0 agent.src.telegram_bot:app"]
