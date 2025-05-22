
# to build locally - docker build -t  cancer-cls .
# # run locally - docker run -it --env-file .env --rm  --gpus all -p 8000:8000 cancer-cls
FROM ubuntu:24.04

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_PROJECT=/app

# System deps:
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends g++ build-essential \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . /app

EXPOSE ${APP_PORT}

ENTRYPOINT ["bash", "-c", "cd /app && uv run fastapi run --host ${APP_HOST} --port ${APP_PORT}"]
