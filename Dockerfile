# Personal AI - zero-dependency, runs on the Python standard library.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Data persists in /data; mount a volume to keep it across runs.
ENV PERSONAL_AI_HOME=/data
VOLUME ["/data"]

# Web UI port.
EXPOSE 8000

# Default to the web UI. For the terminal chat, override with:
#   docker run -it <image> python run.py
CMD ["python", "run.py", "--web", "8000"]
