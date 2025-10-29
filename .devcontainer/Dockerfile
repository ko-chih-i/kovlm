FROM nvidia/cuda:12.8.0-base-ubuntu22.04

SHELL [ "bash", "-c" ]

RUN apt update && \
    apt install -yq \
        ffmpeg \
        build-essential \
        wget \
        curl \
        git \
        sudo

# git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
    sudo apt-get install -yq git-lfs && \
    git lfs install

# Miniconda
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH


RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /app
COPY . /app
RUN /opt/conda/bin/conda create -n sdvlm python=3.10 -y && \
    /opt/conda/bin/conda run -n sdvlm pip install --upgrade pip && \
    /opt/conda/bin/conda run -n sdvlm pip install -e . && \
    /opt/conda/bin/conda run -n sdvlm pip install -e ".[train]"

SHELL ["conda", "run", "-n", "sdvlm", "/bin/bash", "-c"]


