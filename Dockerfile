FROM mmrl/dl-pytorch

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

COPY requirements.txt /main_dir/

WORKDIR /main_dir

RUN pip install --upgrade pip \
    &&  pip install -r requirements.txt

COPY utils ./utils
COPY assets ./assets
COPY evaluate.py .
COPY generate.py .
COPY train.py .
COPY tune.py .

USER root

CMD ["python", "tune.py"]

# docker build -t=don_word_sim:0.1 .
# docker run --shm-size=8gb --rm --gpus all don_word_sim:0.1
# you can increase /dev/shm size by passing '--shm-size=10.24gb' to 'docker run'
