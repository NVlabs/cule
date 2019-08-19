FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

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
        python3.6 \
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
    ln -s /usr/bin/python3.6 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

RUN pip install --upgrade cython \
                          cloudpickle \
                          gym[atari] \
                          opencv-python \
                          psutil \
                          tensorflow \
                          tensorboard \
                          tensorboardX \
                          torch==1.2.0 \
                          torchvision==0.4.0 \
                          tqdm && \
    git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install --upgrade pip && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN git clone -b develop --recursive https://github.com/NVLabs/cule && \
    cd cule && \
    python setup.py install
