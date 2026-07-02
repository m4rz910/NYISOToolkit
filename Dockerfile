FROM python:3.11-slim

WORKDIR /app

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# Install the library itself first (leverages Docker layer caching)
COPY setup.py MANIFEST.in README.md /app/
COPY nyisotoolkit /app/nyisotoolkit
RUN pip install --no-cache-dir -e .

# Docker/Postgres-only extra dependency, kept out of setup.py
COPY docker/sync/requirements.txt /app/docker/sync/requirements.txt
RUN pip install --no-cache-dir -r /app/docker/sync/requirements.txt

COPY docker/sync/sync.py /app/docker/sync/sync.py

ENTRYPOINT ["python", "/app/docker/sync/sync.py"]
