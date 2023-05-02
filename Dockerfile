FROM runpod/pytorch:3.10-2.0.0-117

RUN apt-get update && apt-get install -y git wget

WORKDIR /

RUN pip install runpod

RUN git clone https://github.com/FluttyProger/runpod-sd-trt.git

WORKDIR /runpod-sd-trt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN mkdir -p models/Deliberate/engine/

RUN wget -O models/Deliberate/engine/clip.plan https://huggingface.co/FluttyProger/Deliberate-trt-rtx4090/resolve/main/engine/clip.plan

RUN wget -O models/Deliberate/engine/unet.plan https://huggingface.co/FluttyProger/Deliberate-trt-rtx4090/resolve/main/engine/unet.plan

RUN wget -O models/Deliberate/engine/vae.plan https://huggingface.co/FluttyProger/Deliberate-trt-rtx4090/resolve/main/engine/vae.plan

RUN wget -O models/Deliberate/engine/vae_encoder.plan https://huggingface.co/FluttyProger/Deliberate-trt-rtx4090/resolve/main/engine/vae_encoder.plan

ENV CUDA_MODULE_LOADING=LAZY
CMD python -u handler.py