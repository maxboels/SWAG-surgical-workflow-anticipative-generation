FROM nvcr.io/nvidia/pytorch:23.01-py3
ARG USER_ID
ARG GROUP_ID
ARG USER

ENV TZ=Europe/London

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# cv2 and other dependencies
RUN apt-get update && apt-get install -y sudo \
    libgl1-mesa-glx \
    ffmpeg \
    libimage-exiftool-perl \
    build-essential

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install hydra-core --upgrade
COPY ./requirements_mb.txt .
RUN pip3 install -r requirements_mb.txt


RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

WORKDIR /nfs/home/$USER
EXPOSE 8888