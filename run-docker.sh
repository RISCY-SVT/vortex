#!/usr/bin/env bash
set -eEo pipefail

###############################################################################
# Load environment and figure out paths                                       #
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    # shellcheck disable=SC1090
    source "${SCRIPT_DIR}/.env"
else
    echo "Error: .env file not found in ${SCRIPT_DIR}" >&2
    exit 1
fi

###############################################################################
# Detect proper Compose command (Compose V2 vs. V1)                           #
###############################################################################
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    echo "Error: Docker Compose is not installed." >&2
    exit 1
fi

###############################################################################
# Verify required variables                                                   #
###############################################################################
: "${CONTAINER_NAME:?CONTAINER_NAME is not set in .env}"

###############################################################################
# Start the container if necessary                                            #
###############################################################################
# Check if the container is already running
if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container '${CONTAINER_NAME}' is already running."
else
    # # Remove a stopped container with the same name
    # if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    #     echo "Removing stopped container '${CONTAINER_NAME}' …"
    #     docker rm "${CONTAINER_NAME}"
    # fi

    echo "Starting container '${CONTAINER_NAME}' …"
    $COMPOSE up -d
fi

###############################################################################
# Connect to the running container                                            #
###############################################################################
echo "Connecting to container '${CONTAINER_NAME}' …"
docker exec -ti "${CONTAINER_NAME}" /bin/bash
