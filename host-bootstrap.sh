#!/usr/bin/env bash

set -euo pipefail

#--- helper: run as root (use sudo if needed)
if [[ $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    exec sudo -E bash "$0" "$@"
  else
    echo "Please run as root or install sudo." >&2
    exit 1
  fi
fi

USER_NAME="${SUDO_USER:-${USER:-$(id -un)}}"

#--- detect OS
source /etc/os-release
ID_LIKE_LOWER="$(echo "${ID_LIKE:-}" | tr '[:upper:]' '[:lower:]')"
ID_LOWER="$(echo "${ID}" | tr '[:upper:]' '[:lower:]')"
VERSION_ID_MAJOR="${VERSION_ID%%.*}"

echo "Detected: ID=${ID} VERSION_ID=${VERSION_ID}."

install_docker_ubuntu() {
  apt-get update
  apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  UBUNTU_CODENAME="$(. /etc/os-release; echo "${VERSION_CODENAME}")"
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    ${UBUNTU_CODENAME} stable" > /etc/apt/sources.list.d/docker.list

  apt-get update
  apt-get install -y --no-install-recommends \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  systemctl enable --now docker
}

install_docker_rhel9() {
  dnf -y install dnf-plugins-core ca-certificates curl
  dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
  dnf -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
}

case "${ID_LOWER}" in
  ubuntu)
    if [[ "${VERSION_ID_MAJOR}" != "22" && "${VERSION_ID_MAJOR}" != "24" ]]; then
      echo "Warning: this script targets Ubuntu 22.04/24.04; continuing for ${VERSION_ID}."
    fi
    install_docker_ubuntu
    ;;
  rhel|redhat)
    if [[ "${VERSION_ID_MAJOR}" != "9" ]]; then
      echo "Warning: this script targets RHEL 9; continuing for ${VERSION_ID}."
    fi
    install_docker_rhel9
    ;;
  *)
    # Try ID_LIKE when ID is unexpected (e.g., Rocky/Alma -> rhel)
    if [[ "${ID_LIKE_LOWER}" == *"debian"* || "${ID_LIKE_LOWER}" == *"ubuntu"* ]]; then
      install_docker_ubuntu
    elif [[ "${ID_LIKE_LOWER}" == *"rhel"* || "${ID_LIKE_LOWER}" == *"fedora"* ]]; then
      install_docker_rhel9
    else
      echo "Unsupported OS: ID=${ID} ID_LIKE=${ID_LIKE:-} VERSION_ID=${VERSION_ID}" >&2
      exit 2
    fi
    ;;
esac

#--- add current user to docker group
if ! getent group docker >/dev/null; then
  groupadd docker
fi
usermod -aG docker "${USER_NAME}"

#--- quick smoke tests
echo "Docker version:"
docker --version || true
echo "Compose plugin version:"
docker compose version || true

echo
echo "All set. You may need to log out/in (or run 'newgrp docker') for group changes to take effect."
echo "Next steps:"
echo "  0) If you installed Docker first time, re-login or run:  newgrp docker"
echo "  1) ./build-docker.sh"
echo "  2) ./run-docker.sh"
