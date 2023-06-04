FROM nvidia/cuda:11.6.2-base-ubuntu20.04

LABEL maintainer="mohammad.buet@gmail.com"
LABEL build_date="2023-06-04"
LABEL build_version="v1.0"

COPY . /app

RUN apt update && \
    apt install -y python3-pip && \
    cd /app && \ 
    pip3 install -r requirements.txt

WORKDIR /app

# docker run -it --gpus all --name customer-churn -p 8888:8888 -v "$(pwd)":/app -w /app  nvidia/cuda:11.6.2-base-ubuntu20.04
# https://qiita.com/KEINOS/items/4d8800b38aa6580b66f4
# jupyter notebook --port 8888 --no-browser --allow-root --ip 0.0.0.0