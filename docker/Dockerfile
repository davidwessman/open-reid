# https://hub.docker.com/r/nvidia/cuda/
FROM nvidia/cuda:9.0-runtime-ubuntu16.04
LABEL authors="David Wessman <david.wesmn@gmail.com>, Pontus Andersson <pontus94a@gmail.com>"

# Set arg in build arguments, see section ARGS on:
# https://docs.docker.com/compose/compose-file/#dockerfile
ARG user_id=9001
ARG group_id=9002
ARG http_proxy
ARG no_proxy
ARG https_proxy=${http_proxy}
ENV LANG C.UTF-8

# Setup apt config if http_proxy is set
COPY create_apt_conf.sh /home/create_apt_conf.sh
RUN /home/create_apt_conf.sh

RUN apt-get update
RUN apt-get install -y software-properties-common vim git ca-certificates curl sudo \
                       cmake build-essential libssl-dev zlib1g-dev libbz2-dev \
                       libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
                       xz-utils tk-dev

RUN useradd --shell /bin/bash -u ${user_id} -o -c "" -m user
RUN groupadd -g ${group_id} pyusers
RUN usermod -g pyusers user
RUN echo "user:user" | chpasswd && adduser user sudo

# Cleanup proxy settings
RUN rm /etc/apt/apt.conf

USER user
WORKDIR /home/user
ENV HOME /home/user

RUN git clone https://github.com/yyuu/pyenv.git .pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV PYTHON_CONFIGURE_OPTS="--enable-shared"
RUN pyenv install 3.6.5
RUN pyenv global 3.6.5
RUN pyenv rehash

# Update pip
RUN python3.6 -m pip install pip --upgrade

# Pytorch
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip install numpy scipy torchvision six h5py Pillow scikit-learn metric-learn matplotlib

WORKDIR /home/user/open-reid
CMD ["bash"]
