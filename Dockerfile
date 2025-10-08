# syntax=docker/dockerfile:1.7  # BuildKit syntax
FROM ubuntu:22.04

########################################################################
# ---------- 1. build-time arguments ----------------------------------
########################################################################
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME
ARG CONTAINER_NAME

########################################################################
# ---------- 2. base OS preparation -----------------------------------
########################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

# Do not install recommended/suggested packages
RUN echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/00-docker
# RUN echo 'APT::Get::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
# RUN echo 'APT::Get::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker
# RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
# RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

# Base packages
RUN apt-get update && apt install -y --no-install-recommends \
        aptitude bc build-essential binutils bash cmake ccache cpio ca-certificates curl dos2unix file flex git gpg less \
        dosfstools mtools libedit-dev libffi8 libsm6 libxext6 \
        libstdc++6 libncurses5-dev libtinfo5 libxml2-dev libz-dev libzstd-dev libssl-dev \
        locales lsb-release mc ninja-build openssh-client p7zip-full \
        pkg-config python3 python3-dev python3-pip python3-setuptools python-is-python3 \
        rsync sudo tree valgrind vim wget zlib1g-dev unzip uuid-dev zip \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Locales
RUN locale-gen en_US.UTF-8 ru_RU.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US:en

# python -> python3 symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

########################################################################
# ---------- 3. Python dependencies -----------------------------------
########################################################################
RUN pip3 install --no-cache-dir \
        cython numpy psutil scipy pyyaml \
        sympy typer typing_extensions

# Create working directory
WORKDIR /data

########################################################################
# ---------- 4. Build and install toolchain -------------------------------
########################################################################
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV SLEEP_BETWEEN_DOWNLOADS=5
ENV ZSTD_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    TOOLDIR=/opt/riscv \
    RISCV_CFLAGS="-march=rv64gcv_zvfh -mabi=lp64d -O3" \
    SRCDIR=/usr/src/toolchain

RUN mkdir -p $SRCDIR /opt/riscv && \
    cd $SRCDIR && git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git && cd vortex && \
    $SRCDIR/vortex/ci/install_dependencies.sh && \
    mkdir build && cd build && \
    ../configure --xlen=64 --tooldir=$TOOLDIR --prefix=$TOOLDIR && \
    sed -i.bak -E \
    -e '/# ADDED_BY_DOCKER_PATCH/d' \
    -e 's|^REPOSITORY=.*|REPOSITORY=https://raw.githubusercontent.com/vortexgpgpu/vortex-toolchain-prebuilt/master|' \
    -e '0,/^set -x$/s//set -x\nset -o pipefail # ADDED_BY_DOCKER_PATCH/' \
    -e '/^OSVERSION=/a\
SLEEP_BETWEEN_DOWNLOADS=${SLEEP_BETWEEN_DOWNLOADS:=3} # ADDED_BY_DOCKER_PATCH' \
   -e 's|wget[[:space:]]+\$REPOSITORY/|curl -fL --retry 20 --retry-all-errors --retry-delay 2 -O $REPOSITORY/|g' \
   -e '/^[[:space:]]*curl .* \$REPOSITORY\//a\
    sleep "$SLEEP_BETWEEN_DOWNLOADS" # ADDED_BY_DOCKER_PATCH' \
   ./ci/toolchain_install.sh && \
   awk 'BEGIN{print "#!/bin/sh"} /^export (RISCV|LLVM|PATH)=/{print}' ./ci/toolchain_env.sh \
        > /etc/profile.d/40-vortex.sh && \
    chmod +x /etc/profile.d/40-vortex.sh
# Optionally also bake PATH non-interactively:
ENV PATH="/opt/riscv/riscv64-gnu-toolchain/bin:/opt/riscv/riscv32-gnu-toolchain/bin:/opt/riscv/llvm-vortex/bin:${PATH}"

RUN cd $SRCDIR/vortex/build && \
    ./ci/toolchain_install.sh --all && \
    . ./ci/toolchain_env.sh && \
    make -s && make install

########################################################################
# ---------- 5. user & sudo -------------------------------------------
########################################################################
RUN set -eux; \
    if ! getent group "${GROUP_ID}" >/dev/null; then \
      groupadd -g "${GROUP_ID}" "${USER_NAME}"; \
    fi; \
    if ! getent passwd "${USER_ID}" >/dev/null; then \
      useradd -u "${USER_ID}" -g "${GROUP_ID}" -m -s /bin/bash "${USER_NAME}"; \
    fi; \
    mkdir -p /data && chown -R "${USER_ID}:${GROUP_ID}" /data; \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}; \
    chmod 0440 /etc/sudoers.d/${USER_NAME}

########################################################################
# ---------- 6. shell tweaks for the user -----------------------------
########################################################################
# ~/.bashrc: aliases & helper functions only
RUN mkdir -p /home/${USER_NAME} && \
    tee -a /home/${USER_NAME}/.bashrc >/dev/null <<'BASH'
# aliases
alias la='ls -A'
alias l='ls -CF'
alias ll='ls -alhFp'
alias hcg='cat ~/.bash_history | grep '
alias 7zip='7za a -t7z -m0=lzma2:d1536m:fb273:mf=bt4:lc4:pb2 -mx=9 -myx=9 -ms=4g -mqs=on -mmt=8 '
alias cls='clear;clear'
alias gcrs='git clone --recurse-submodules '
alias gprs='git pull  --recurse-submodules '

# coloured prompt
export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "

# helper: print PATH line by line
path() {
    local IFS=:
    printf '%s\n' ${PATH}
}

# helper: print LD_LIBRARY_PATH
libs() {
    local IFS=:
    printf '%s\n' ${LD_LIBRARY_PATH}
}
source /usr/src/toolchain/vortex/build/ci/toolchain_env.sh
BASH

# ~/.bash_profile: source .bashrc for login shells
RUN tee /home/${USER_NAME}/.bash_profile >/dev/null <<'BASH'
# Load user aliases and functions
[ -f ~/.bashrc ] && . ~/.bashrc
BASH

RUN chown -R ${USER_ID}:${GROUP_ID} /home/${USER_NAME} 

########################################################################
# ---------- 7. working dir & entrypoint ------------------------------
########################################################################
WORKDIR /data
USER ${USER_NAME}

CMD ["/bin/bash"]
