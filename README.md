# Vortex Toolchain Container

## Overview
This repository provides a Docker-based workflow for building and packaging the Vortex RISC-V GPU toolchain. The supplied Dockerfile compiles the full stack inside an Ubuntu 22.04 image, while helper scripts bootstrap Docker on the host, create the image, and attach to a persistent development container.

## Repository Layout
- `Dockerfile` – builds the toolchain and prepares runtime helpers under `/opt/riscv`.
- `docker-compose.yml` – defines the `vortex` service, mounts `./data` to `/data`, and injects per-user IDs.
- `host-bootstrap.sh` – installs Docker and the compose plugin on supported hosts with elevated permissions.
- `build-docker.sh` – refreshes `.env`, cleans builder cache, and runs `docker compose build`.
- `run-docker.sh` – ensures the container is running and opens an interactive shell session.
- `rsync_csi_to_M.sh` – optional helper to mirror the repo to a shared mount.
- `data/` – workspace for build outputs; ignored by Git so you can safely persist artifacts.

## Getting Started
1. **Bootstrap Docker (first time only):** `./host-bootstrap.sh`
2. **Build the toolchain image:** `./build-docker.sh`
3. **Launch / attach to the container:** `./run-docker.sh`

The first run of `build-docker.sh` writes `.env` with host-specific IDs. 

## Common Tasks
- Tail container logs: `docker compose logs vortex -f`
- Open a second shell: `docker exec -ti $CONTAINER_NAME /bin/bash`
- Rebuild from scratch after base changes: `docker builder prune -f && ./build-docker.sh`

## Development Notes
- Keep new automation scripts at the repository root and follow the strict Bash style used by existing files (`#!/usr/bin/env bash`, `set -euo pipefail`).
- Store any persistent build output or toolchain artifacts in `data/`; keep secrets outside the repository and inject them at runtime.
- Refer to `NOTES.md` for contributor workflow details, coding conventions, and pull-request expectations.

## Support
Issues or feature requests should include the commands executed, relevant log excerpts from `docker compose logs vortex`, and host environment details (OS, Docker version). Contributions that introduce new tests or CI steps are highly encouraged.
