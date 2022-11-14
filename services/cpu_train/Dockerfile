FROM python:3.9-slim-buster

WORKDIR /workspace

COPY services/cpu_train/requirements.txt .

RUN apt-get update -y && apt-get -y install libboost-filesystem-dev libboost-dev ffmpeg libsm6 libxext6 g++

RUN pip install -r requirements.txt

COPY . .
