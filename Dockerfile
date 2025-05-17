
# to build locally - docker build -t  cancer-cls .
# run locally - docker run -it --env-file .env --rm  --gpus all -p 5000:5000 cancer-cls
FROM ubuntu:24.04

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
# System deps:
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends g++ build-essential \
        curl \
        # git \
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

RUN uv sync --no-dev

EXPOSE 8000

# RUN uv run dvc repro
CMD ["bash", "-c", "uv run fastapi dev --host 0.0.0.0 --port 8000"]
# CMD ["bash", "-c", "uv run dvc repro --force  && uv run fastapi dev --host 0.0.0.0 --port 5000"]





# to build locally - docker build --no-cache -t cancer-cls .
