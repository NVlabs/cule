FROM nvcr.io/nvidian/pytorch:20.12-py3

RUN pip install --upgrade cloudpickle \
                          gym[atari] \
                          opencv-python \
                          psutil \
                          tqdm \
                          wandb
ADD . /cule
RUN cd /cule && \
    python setup.py install
