#!/bin/bash
# smoke_test.sh — post-build validation for the three-image BindCraft setup.
#
# Validates everything that does NOT require a GPU, so you can confirm the split
# works right after building, even on a CPU-only machine:
#   1. dependency isolation — each image imports only its own stack
#   2. the Rosetta RPC boundary end-to-end — start the service, /health, real pr_relax
#
# GPU visibility and the full design loop need a GPU and are documented separately
# (see README-docker.md "构建后测试").
#
# Usage:
#   docker compose build                 # or pull prebuilt images and set *_IMAGE in .env
#   bash docker/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."

GPU_IMAGE="${GPU_IMAGE:-bindcraft-gpu:local}"
ROSETTA_IMAGE="${ROSETTA_IMAGE:-bindcraft-rosetta:local}"
ORCH_IMAGE="${ORCH_IMAGE:-bindcraft-orchestrator:local}"
PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

echo "== 1. dependency isolation =="
docker run --rm "$ORCH_IMAGE"    python -c "import pandas; print('orchestrator ok')" >/dev/null 2>&1 \
    && ok "orchestrator: pandas imports" || bad "orchestrator import"
docker run --rm "$ROSETTA_IMAGE" python -c "import pyrosetta; import functions" >/dev/null 2>&1 \
    && ok "rosetta: pyrosetta + functions (no jax) import" || bad "rosetta import"
# the GPU image must NOT contain pyrosetta, and must import the package in gpu mode
docker run --rm "$GPU_IMAGE" python -c "import jax, colabdesign, functions" >/dev/null 2>&1 \
    && ok "gpu: jax + colabdesign + functions import" || bad "gpu import"
docker run --rm "$GPU_IMAGE" sh -c '! python -c "import pyrosetta" 2>/dev/null' \
    && ok "gpu: pyrosetta is absent (as intended)" || bad "gpu unexpectedly has pyrosetta"

echo "== 2. Rosetta RPC boundary (no GPU needed) =="
cleanup() { docker compose down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker compose up -d rosetta >/dev/null 2>&1 || { bad "could not start rosetta service"; exit 1; }

echo -n "  waiting for /health "
healthy=0
for _ in $(seq 1 40); do
  if docker compose exec -T rosetta python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health',timeout=5)" >/dev/null 2>&1; then
    healthy=1; break
  fi
  echo -n "."; sleep 5
done
echo ""
[ "$healthy" = 1 ] && ok "rosetta /health responds" || { bad "rosetta /health timeout"; exit 1; }

# Real pr_relax over HTTP on the bundled example PDB, exchanged via the shared volume.
docker compose exec -T rosetta sh -c 'mkdir -p /workspace/_smoke && cp /app/example/PDL1.pdb /workspace/_smoke/in.pdb' >/dev/null 2>&1
docker compose exec -T rosetta python - <<'PY' >/tmp/bc_relax.out 2>&1
import json, urllib.request
payload = json.dumps({"pdb_file": "/workspace/_smoke/in.pdb",
                      "relaxed_pdb_path": "/workspace/_smoke/out.pdb"}).encode()
req = urllib.request.Request("http://localhost:8000/pr_relax", data=payload,
                             headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=1800).read().decode())
PY
if grep -q '"ok": true' /tmp/bc_relax.out && \
   docker compose exec -T rosetta test -s /workspace/_smoke/out.pdb >/dev/null 2>&1; then
  ok "pr_relax over HTTP produced a relaxed PDB"
else
  bad "pr_relax RPC failed — service response:"; sed 's/^/      /' /tmp/bc_relax.out
fi

echo ""
echo "== summary: ${PASS} passed, ${FAIL} failed =="
[ "$FAIL" = 0 ]
