#!/usr/bin/env bash
# Builds the JOSS draft PDF (paper.pdf in the repository root) with the official
# Open Journals toolchain (openjournals/inara). Uses podman or docker if available,
# and on Windows falls back to podman inside WSL.
# Usage: bash resources/scripts/build_paper.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image=docker.io/openjournals/inara

if command -v podman >/dev/null; then
  podman run --rm --userns=keep-id --volume "$root:/data:Z" --env JOURNAL=joss "$image"
elif command -v docker >/dev/null; then
  docker run --rm --user "$(id -u):$(id -g)" --volume "$root:/data" --env JOURNAL=joss "$image"
elif command -v wsl.exe >/dev/null; then
  # Git Bash: run podman in WSL on the same checkout (C:\... -> /mnt/c/...).
  export MSYS_NO_PATHCONV=1
  wsl_root="$(wsl.exe -d "${WSL_DISTRO:-FedoraLinux-44}" --exec wslpath -a "$(cygpath -w "$root")" | tr -d '\r')"
  wsl.exe -d "${WSL_DISTRO:-FedoraLinux-44}" --exec podman run --rm --userns=keep-id \
    --volume "$wsl_root:/data:Z" --env JOURNAL=joss "$image"
else
  echo "Need podman, docker, or WSL with podman." >&2
  exit 1
fi

echo "Wrote $root/paper.pdf"
