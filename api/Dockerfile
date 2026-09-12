FROM python:3.10-slim

# build-essential: kraken 4.3.13's older transitive deps (e.g. scikit-image 0.19.3) don't ship
# prebuilt wheels for this Python version and need to compile from source - same reason
# experiments/phase0_validation/docker/chat/Dockerfile and experiments/phase4_detection/docker/
# Dockerfile need it.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Build context is the repo root (docker build -f api/Dockerfile .), not api/ itself - every
# COPY below is written relative to that, including this one.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The API imports nomnaocr_lib/translate_lib/lm/detect_lib from experiments/ at runtime via
# api/_pipeline_paths.py (reused in place, not duplicated) - both this api/ directory and the
# relevant experiments/ subtrees need to be present in the image. web/ (Phase 5's frontend) is
# served via api/main.py's StaticFiles mount, so it needs to be present too, or that mount fails
# at startup.
COPY api/ /app/api/
COPY web/ /app/web/
COPY experiments/phase0_validation/nomnaocr_lib/ /app/experiments/phase0_validation/nomnaocr_lib/
COPY experiments/phase2b_postcorrection/lm/ /app/experiments/phase2b_postcorrection/lm/
COPY experiments/phase3_translation/translate_lib/ /app/experiments/phase3_translation/translate_lib/
COPY experiments/phase4_detection/detect_lib/ /app/experiments/phase4_detection/detect_lib/

# Data/weights (recognizer .h5, char_lm.json, best_lambda.json, reading_dict.json, the Rung 1
# detection checkpoint) are large binaries kept out of git (see .gitignore) - mount the local
# experiments/ tree at the SAME path api/config.py's defaults already expect, rather than baking
# them into the image or introducing a separate directory layout + env vars just for Docker:
#   docker run --rm -p 8000:8000 -v "$(pwd)/experiments:/app/experiments" nomocr-api

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
