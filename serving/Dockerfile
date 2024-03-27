FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ENV base_path /workspace/
COPY . ${base_path}

RUN apt-get update -y && \
    apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${base_path}/SwimSwap/insightface_func/models \
    ${base_path}/SwimSwap/parsing_model/checkpoint \
    ${base_path}/SwimSwap/arcface_model 

ADD https://storage.makezenerator.com:9000/voice2face-public/model_resource/swimswap/79999_iter.pth ${base_path}/SwimSwap/parsing_model/checkpoint/
ADD https://storage.makezenerator.com:9000/voice2face-public/model_resource/swimswap/antelope.zip ${base_path}/SwimSwap/insightface_func/models/
RUN unzip ${base_path}/SwimSwap/insightface_func/models/antelope.zip -d ${base_path}/SwimSwap/insightface_func/models/ && \
    rm ${base_path}/SwimSwap/insightface_func/models/antelope.zip

RUN pip install --upgrade pip && \
    pip install -r ${base_path}/requirements.txt

WORKDIR ${base_path}
ENTRYPOINT ["gunicorn", "-w", "1", "-b", "0.0.0.0:5050", "app:app", "--reload"]
