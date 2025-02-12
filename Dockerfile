# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter

# FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /code

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# ARG USERNAME=ultron
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID

# Create the user
# RUN apt-get update && apt-get install -y sudo \
#     && groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# System deps:
RUN apt-get update -y && \
    apt-get upgrade -y  \
    && apt-get install curl ffmpeg libsm6 libxext6  -y 


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ENV VIRTUAL_ENV="/code/.venv" 
ENV PATH="/root/.local/bin/:$PATH"
# COPY pyproject.toml uv.lock /code/
# RUN uv sync --active  --frozen


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME
ENV PATH="/root/.local/bin:${PATH}"

# COPY . /code/
# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["--ip=0.0.0.0","--allow-root","--no-browser"]

