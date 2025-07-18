# FROM tensorflow/tensorflow:2.13.0-gpu-jupyter
# warning: 2.15 uses python 3.11 which is experimental in Ray
FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing \
    && apt-get --no-install-recommends install -y \
        git \
        gdb \
        curl \
        ca-certificates \
        build-essential \
        host \
        vim \
        man \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use Tini to init container
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "-g", "--"]

RUN curl -fsSL https://code-server.dev/install.sh | sh

# install VS Code extensions
RUN code-server --install-extension ms-python.python \
                --install-extension vscode-icons-team.vscode-icons \
                --install-extension ms-toolsai.jupyter \
                --install-extension gruntfuggly.todo-tree

CMD ["code-server", "--auth", "none", "--port", "8888", "--host", "0.0.0.0"]

WORKDIR /

COPY requirements.txt /
RUN python -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
WORKDIR /opt/project

EXPOSE 6006