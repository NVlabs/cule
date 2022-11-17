FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US

RUN apt-get -y update -qq && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        gcc \
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
        python3.9 \
        python3-dev \
        python3-setuptools \
        python3-pip \
        vim \
        ssh \
        wget \
        vim \
        zip \
    && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.9 /usr/bin/python
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

RUN pip install --upgrade cython \
                          cloudpickle \
                          gym[atari] \
                          opencv-python \
                          psutil \
                          torch==1.11.0 \
                          torchvision==0.12.0 \
                          tqdm

RUN git clone -b master --recursive https://github.com/NVLabs/cule && \
    cd cule && \
    python setup.py install
