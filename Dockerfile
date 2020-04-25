FROM phusion/baseimage:0.11
WORKDIR /openprotein

# install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  clang \
  cmake \
  curl \
  git \
  libc6-dev \
  make \
  python3 \
  python3-pip \
  python3-setuptools \
  python3-dev \
  build-essential \
  default-jre \
  autoconf \
  autogen \
  libtool \
  shtool \
  autopoint \
  software-properties-common

RUN apt install python3.7 python3.7-dev -y

RUN python3 -m pip install wheel
RUN python3 -m pip install pipenv

COPY . /openprotein

RUN pipenv install

RUN echo "pipenv shell" >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]