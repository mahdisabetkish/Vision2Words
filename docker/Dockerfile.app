# syntax=docker/dockerfile:1.7
#
# Gradio demo image: runs app/app.py on CPU so anyone can try the captioner
# with one command. The trained decoders are downloaded from the Hugging Face
# model repo on first start (see MODEL_REPO in app/app.py), so the image
# contains no weights. Build from the repo root:
#   docker build -f docker/Dockerfile.app -t vision2words-app .
# Run:
#   docker run --rm -p 7860:7860 vision2words-app
# then open http://localhost:7860

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN useradd --create-home --uid 1000 v2w

WORKDIR /app

# CPU-only torch keeps the image a few hundred MB instead of several GB.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY app/ app/
RUN pip install --no-cache-dir ".[app]" \
    && rm -rf /root/.cache

# Gradio listens on localhost by default, which isn't reachable from outside
# a container. Binding to all interfaces is what makes -p 7860:7860 work.
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

USER v2w
EXPOSE 7860
CMD ["python", "app/app.py"]
