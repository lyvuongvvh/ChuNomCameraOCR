FROM python:3.10-slim

# build-essential: kraken 4.3.13's older transitive deps (e.g. scikit-image 0.19.3) don't ship
# prebuilt wheels for this Python version and need to compile from source - same reason
# experiments/phase0_validation/docker/chat/Dockerfile needs it.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Confirmed compatible in one environment (see experiments/phase4_detection/README.md's
# "kraken + TF coexistence check"): pip's resolver settles on numpy==1.23.5, which satisfies
# both kraken's and tensorflow-cpu==2.10.0's constraints - the two were never combined before
# this project reached Phase 4 (Phase 0 ran them in separate containers/Python versions).
RUN pip install --no-cache-dir \
    'tensorflow-cpu==2.10.0' \
    'kraken==4.3.13' \
    'shapely' \
    'pillow'

WORKDIR /workspace
