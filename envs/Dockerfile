FROM nvcr.io/nvidian/pytorch:18.12.1-py3 as base

RUN apt-get -y update -qq && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        gcc-5 \
        cmake \
        htop \
        curl \
        git \
        libomp-dev \
        libsm6 \
        libssl-dev \
        libxrender-dev \
        libxext-dev \
        iproute2 \
        module-init-tools \
        vim \
        ssh \
        wget \
        vim \
        zip \
    && \
    rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

FROM base as conda
RUN pip install --upgrade pip && \
    pip install --upgrade cloudpickle gym[atari] opencv-python psutil tqdm tensorflow tensorboard tensorboardX && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
