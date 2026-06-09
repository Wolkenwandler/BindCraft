#!/bin/bash
# entrypoint-gpu.sh
# Runs inside the activated micromamba "base" env (via micromamba's _entrypoint.sh).
# 1) Ensures AlphaFold2 weights exist on the mounted params volume.
# 2) Waits for the Rosetta service to be reachable (relax/score run there).
# 3) Execs the given CMD (worker.py by default).
set -euo pipefail

PARAMS_DIR="${PARAMS_DIR:-/app/params}"
PARAMS_URL="${PARAMS_URL:-https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar}"
PARAMS_MARKER="${PARAMS_DIR}/params_model_5_ptm.npz"

mkdir -p "${PARAMS_DIR}"

if [ -f "${PARAMS_MARKER}" ]; then
  echo "[entrypoint] AlphaFold2 weights already present at ${PARAMS_DIR}"
else
  echo "[entrypoint] AlphaFold2 weights missing -> downloading (~5.3 GB) to ${PARAMS_DIR}"
  TAR="${PARAMS_DIR}/alphafold_params_2022-12-06.tar"
  wget -q --show-progress -O "${TAR}" "${PARAMS_URL}"
  [ -s "${TAR}" ] || { echo "[entrypoint] ERROR: download produced empty file"; exit 1; }
  tar tf "${TAR}" >/dev/null 2>&1 || { echo "[entrypoint] ERROR: corrupt weights archive"; exit 1; }
  tar -xf "${TAR}" -C "${PARAMS_DIR}"
  [ -f "${PARAMS_MARKER}" ] || { echo "[entrypoint] ERROR: weights not found after extraction"; exit 1; }
  rm -f "${TAR}"
  echo "[entrypoint] AlphaFold2 weights ready"
fi

# Wait for the Rosetta service (functions/rosetta_client also retries, but fail fast here
# with a clear message if it never comes up).
if [ "${BINDCRAFT_MODE:-}" = "gpu" ]; then
  ROSETTA_URL="${ROSETTA_URL:-http://rosetta:8000}"
  echo "[entrypoint] waiting for Rosetta service at ${ROSETTA_URL} ..."
  for i in $(seq 1 60); do
    if python - "$ROSETTA_URL" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(sys.argv[1].rstrip('/') + '/health', timeout=10)
PY
    then
      echo "[entrypoint] Rosetta service is up"
      break
    fi
    sleep 5
    [ "$i" -eq 60 ] && echo "[entrypoint] WARNING: Rosetta service not reachable yet; continuing (client will retry)"
  done
fi

exec "$@"
