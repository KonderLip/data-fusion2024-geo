FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace