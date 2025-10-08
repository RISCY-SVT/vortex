# Repository Guidelines

## Project Structure & Module Organization
The repo centres on containerizing the Vortex RISC-V toolchain. `Dockerfile` builds the Ubuntu 22.04 image, installs dependencies, compiles the toolchain into `/opt/riscv`, and writes helper env scripts. `docker-compose.yml` names the service `vortex`, mounts `./data` to `/data`, and injects user IDs via args. Host helpers (`host-bootstrap.sh`, `build-docker.sh`, `run-docker.sh`, `rsync_csi_to_M.sh`) live at the root; add new automation beside them with actionable file names. Keep persistent build outputs under `data/`, and avoid committing heavy binaries or secrets.

## Build, Test, and Development Commands
Use `./host-bootstrap.sh` once per machine to install Docker plus the compose plugin (runs with sudo if needed). `./build-docker.sh` refreshes `.env`, prunes stale build cache, and executes `docker compose build`. `./run-docker.sh` ensures the container is up and opens an interactive shell; it requires `.env` to define `CONTAINER_NAME`. Helpful diagnostics include `docker compose logs vortex -f` for streaming output and `docker exec -ti $CONTAINER_NAME bash` for extra sessions.

## Coding Style & Naming Conventions
Scripts should target Bash, start with `#!/usr/bin/env bash`, and enable strict mode (`set -euo pipefail`). Match the existing two-space indentation in YAML and shell blocks, keep 120-character lines, and reserve ALL_CAPS for exported variables (`USER_ID`, `TOOLDIR`). Descriptive lowercase names suit locals (`script_dir`). Run `shellcheck` or `bash -n` before submitting.

## Testing Guidelines
We rely on smoke tests rather than formal suites. After changing the Dockerfile or orchestration scripts, rebuild and launch the stack (`./build-docker.sh && ./run-docker.sh`) and confirm tool binaries appear on the `PATH` inside the container. Capture any regressions with `docker compose logs vortex` and note manual checks in your PR description. Propose CI additions when you introduce repeatable validation steps.

## Commit & Pull Request Guidelines
History is still forming, so adopt Conventional Commits in the imperative mood (e.g., `chore(container): sync env defaults`). Keep each commit focused on one concern. Pull requests must summarize the change, list verification commands, mention follow-up tasks, and link issues when relevant. Attach screenshots only when they add clarity (UI, dashboards).

## Environment & Security Notes
`.env` stays untracked; it stores host-specific IDs and the container name. Never bake credentials or keys into the Docker image—pass them as secrets, bind mounts, or environment overrides at runtime. When updating toolchain sources, prefer pinned tags or checksums and document any new network requirements.
