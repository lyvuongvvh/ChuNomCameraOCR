# NomNaOCR's requirements.txt pins tensorflow==2.10.0 (Keras 2 API). Newer TensorFlow ships
# Keras 3, which changes enough (KerasTensor.get_shape(), .h5 weight loading behavior) that
# faithfully reproducing the pretrained CRNNxCTC weights needs this exact major/minor version.
# TF 2.10 requires Python <=3.10.
FROM python:3.10-slim

RUN pip install --no-cache-dir tensorflow-cpu==2.10.0 pillow "numpy<2"

WORKDIR /workspace
ENTRYPOINT ["python", "scripts/run_nomnaocr.py"]
