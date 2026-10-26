#!/bin/bash
set -Eeuo pipefail
trap 'rc=$?; echo "Container launch failed at line $LINENO (exit $rc)" >&2; exit "$rc"' ERR
cd "${STUDY_ROOT:?}"
: "${TRM_CONTAINER:?}" "${TRM_PYTHON:?}"
[[ -r "$TRM_CONTAINER" ]] || { echo "Cannot read container: $TRM_CONTAINER" >&2; exit 2; }
if [[ "$(uname -m)" != aarch64 && "$TRM_CONTAINER" = *arm* ]]; then
  echo 'ARM container requires a Kairos aarch64 GH200 node.' >&2
  exit 2
fi
binds=(--bind /work --bind "$STUDY_ROOT:$STUDY_ROOT")
exec apptainer exec --nv "${binds[@]}" --pwd "$STUDY_ROOT" "$TRM_CONTAINER" "$TRM_PYTHON" "$@"
