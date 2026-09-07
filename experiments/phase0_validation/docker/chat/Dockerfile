FROM python:3.11-slim

# build-essential: some of kraken 4.3.13's older transitive deps (e.g. scikit-image) don't ship
# prebuilt wheels for this Python version and need to compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the kraken release current when CHAT_models was published (DOI 10.5281/zenodo.8325545,
# Sept 2023) - the latest kraken (7.x) produced degenerate/near-constant predictions from these
# .mlmodel weights (confirmed by testing), likely an inference-pipeline behavior change since 2023.
RUN pip install --no-cache-dir kraken==4.3.13 pillow

WORKDIR /workspace
ENTRYPOINT ["python", "scripts/run_chat.py"]
