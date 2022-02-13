FROM ubuntu:20.04
ENV username=eric

RUN apt -y update && DEBIAN_FRONTEND=noninteractive apt install -y \
  python3-numpy \
  python3-matplotlib \
  python3-pandas \
  python3-sklearn \
  python3-sklearn-pandas \
  python3-pip

RUN pip3 install \
  notebook \
  tensorflow

RUN useradd -rm -d /home/${username} -s /bin/bash -g root -G sudo -u 1000 ${username} 
RUN  echo '${username}:${username}' | chpasswd

EXPOSE 22
