FROM tensorflow/tensorflow:2.15.0-gpu-jupyter AS base
# warning: 2.15 uses python 3.11 which is experimental in Ray
# FROM tensorflow/tensorflow:2.13.0-gpu-jupyter as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing \
    && apt-get --no-install-recommends install -y \
        git \
        gdb \
        curl \
        ca-certificates \
        build-essential \
        host \
        vim-gtk3 \
        sudo \
        man \
        tmux \
        net-tools \
        cmake \
        openssh-server \
        iproute2 \
        inetutils-ping \
        python3-dev \
        python-is-python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# useradd with home dir created
# RUN addgroup --gid 100 usersd
RUN useradd -m --uid 1025 --gid 100 developer
# give "developer" user the password "developer"
RUN echo "developer:developer" | chpasswd
# add USER developer to GROUP sudo
RUN adduser developer sudo

# Use Tini to init container, install code-server for use on ACEHub
FROM base AS base2
ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

RUN curl -fsSL https://code-server.dev/install.sh | sh

# install VS Code extensions
RUN code-server --install-extension ms-python.python \
                --install-extension vscode-icons-team.vscode-icons \
                --install-extension ms-toolsai.jupyter \
                --install-extension gruntfuggly.todo-tree

RUN git clone https://github.com/samoshkin/tmux-config.git && ./tmux-config/install.sh

# install python libraries
FROM base2 AS develop
WORKDIR /
COPY requirements.txt /
RUN python -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# set user and container startup
FROM develop AS final
# is refreshing library cache necessary? Image tries to do it on startup and fails as developer user
# RUN /sbin/ldconfig.real
# become desired user
WORKDIR /home/developer
COPY install_sc2.sh /home/developer/
RUN chmod +x ./install_sc2.sh
RUN ./install_sc2.sh

USER developer
WORKDIR /opt/project
# Tensorboard port
EXPOSE 6006
# What to do on container start: run code-server through tini
ENTRYPOINT ["/tini", "-g", "--"]
CMD ["code-server", "--auth", "none", "--port", "8888", "--host", "0.0.0.0"]
