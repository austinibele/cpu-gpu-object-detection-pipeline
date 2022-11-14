FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /workspace

COPY services/gpu_train/requirements.txt .

RUN apt-get update -y && apt-get -y install libboost-filesystem-dev libboost-dev ffmpeg libsm6 libxext6 g++

RUN pip install -r requirements.txt

COPY . .
