#!/bin/bash
# entrypoint-gpu.sh
# Runs inside the activated micromamba "base" env (via micromamba's _entrypoint.sh).
# AF2 weights are baked into the image at /app/params (see Dockerfile.gpu), so there is
# no runtime download. This entrypoint just sanity-checks the weights, waits for the
# Rosetta service, then execs the given CMD (worker.py by default).
set -euo pipefail

PARAMS_DIR="${PARAMS_DIR:-/app/params}"
if [ ! -f "${PARAMS_DIR}/params_model_5_ptm.npz" ]; then
  echo "[entrypoint] ERROR: AF2 weights not found in ${PARAMS_DIR}. They should be baked into"
  echo "             the image at build time; rebuild docker/Dockerfile.gpu."
  exit 1
fi
echo "[entrypoint] AF2 weights present at ${PARAMS_DIR}"

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
