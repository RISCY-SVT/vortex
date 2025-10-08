#!/usr/bin/env bash
set -eo pipefail

###############################################################################
# Detect proper Compose command (Compose V2 vs. V1)                           #
###############################################################################
# Prefer 'docker compose' (CLI plugin, Compose V2). Fallback to 'docker-compose'
# if the plugin is not available (old standalone binary, Compose V1).

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed. Run script 'host-bootstrap.sh' to install it." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not accessible for user '$USER'."
  echo "Run:  newgrp docker   (or re-login)   and retry."
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "Error: Docker Compose is not installed." >&2
    exit 1
fi

###############################################################################
# Update .env with current user / group IDs                                   #
###############################################################################
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
USER_NAME=$USER

echo "Updating .env with USER_ID=$CURRENT_UID, GROUP_ID=$CURRENT_GID, USER_NAME=$USER_NAME"

# Create .env if missing, then update or append keys
touch .env
grep -q '^USER_ID='   .env && sed -i "s/^USER_ID=.*/USER_ID=$CURRENT_UID/"   .env || echo "USER_ID=$CURRENT_UID"   >> .env
grep -q '^GROUP_ID='  .env && sed -i "s/^GROUP_ID=.*/GROUP_ID=$CURRENT_GID/" .env || echo "GROUP_ID=$CURRENT_GID"  >> .env
grep -q '^USER_NAME=' .env && sed -i "s/^USER_NAME=.*/USER_NAME=$USER_NAME/" .env || echo "USER_NAME=$USER_NAME" >> .env

###############################################################################
# Ensure data directory exists                                                #
###############################################################################
if [ ! -d "./data" ]; then
    mkdir -p ./data
    echo "Created data directory"
fi

###############################################################################
# Build Docker image                                                          #
###############################################################################
echo "Cleaning unused build cache …"
docker builder prune -f

echo "Building Docker image using '$COMPOSE build' …"
$COMPOSE build

echo
echo "Docker image built successfully."
echo "Run the container with ./run-docker.sh"
