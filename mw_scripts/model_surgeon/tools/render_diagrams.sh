#!/usr/bin/env bash
# Render the model-surgery architecture diagrams.
#
#   ./render_diagrams.sh                 # all 8 -> ./exports
#   ./render_diagrams.sh v3              # just one
#   ./render_diagrams.sh --formats png,svg,pdf
#
# matplotlib only exists inside the training container, and that container runs
# as root against an NFS export that root-squashes, so it cannot write next to
# the script. Render to the container's /tmp, then copy the results back out.
set -euo pipefail

HOST=${RENDER_HOST:-gpu-31}
CONTAINER=${RENDER_CONTAINER:-mellum2_3xh200}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The repo is bind-mounted into the container; map this directory onto that path
# rather than hard-coding one, so moving the script does not break rendering.
SCRIPT=${HERE/#\/net\/michaelw-vm\/srv\/nfs\/michaelw-data\/ws//workspace/ws}/draw_surgery_arch.py
OUT_DIR=$HERE/exports
REMOTE_OUT=/tmp/surgery_exports.$$

ssh "$HOST" "docker exec '$CONTAINER' python '$SCRIPT' --out-dir '$REMOTE_OUT' $*"

mkdir -p "$OUT_DIR"
ssh "$HOST" "docker exec '$CONTAINER' tar cf - -C '$REMOTE_OUT' ." | tar xf - -C "$OUT_DIR"
ssh "$HOST" "docker exec '$CONTAINER' rm -rf '$REMOTE_OUT'"

echo "wrote to $OUT_DIR"
ls -1 "$OUT_DIR"
