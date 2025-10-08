#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

rsync -avz --no-perms --delete "${SCRIPT_DIR}/"  /mnt/hgfs/M/RISCY/"${SCRIPT_DIR##*/}"

echo
echo "Files successfully synced. Current time: $(date  +'%F %T %Z')"
date +%Y-%m-%d_%H-%M-%S
