FROM python@sha256:4ba18b066cee17f2696cf9a2ba564d7d5eb05a91d6a949326780aa7c6912160d
RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/reference-requirements.txt
RUN pip install --no-cache-dir -r /opt/reference-requirements.txt && pip freeze > /opt/reference-freeze.txt
