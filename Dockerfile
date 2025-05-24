
# to build locally - docker build -t  cancer-cls .
# # run locally - docker run -it --env-file .env --rm  --gpus all -p 8000:8000 cancer-cls
FROM python:3.12-slim

ENV APP_HOST=0.0.0.0
ENV APP_PORT=5000
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV UV_PROJECT=/app

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends g++ build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR ${UV_PROJECT}

# Copy your application code
COPY ./app /app/app
COPY ./app/pyproject.toml .
COPY ./src src
COPY ./configs configs


EXPOSE ${APP_PORT}

ENTRYPOINT ["bash", "-c", "cd ${UV_PROJECT} && uv run fastapi run --host ${APP_HOST} --port ${APP_PORT}"]
