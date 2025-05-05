# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter

# FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
FROM ubuntu:25.04

WORKDIR /code

# ENV UV_PROJECT_ENVIRONMENT="/code/.venv"

# System deps:
RUN apt-get update -y && \
    apt-get upgrade -y  \
    && apt-get install curl ffmpeg libsm6 libxext6  -y 

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/



