---
name: bindcraft-rosetta-service
description: Use when running BindCraft protein binder design and you need the PyRosetta scoring backend — starting/verifying the bindcraft:rosetta container, or debugging pr_relax / score_interface / interface_dG / libgfortran errors. Required dependency before bindcraft-gpu-design in gpu mode.
---

# BindCraft Rosetta Scoring Service

## Overview

The `ai4science/bindcraft:rosetta` image is the **CPU-only PyRosetta backend** for BindCraft. In the split (Docker) architecture the GPU design loop runs in `BINDCRAFT_MODE=gpu` and forwards every PyRosetta call (`pr_relax`, `score_interface`, `unaligned_rmsd`, `align_pdbs`) over HTTP to this service. It has **no jax**; the GPU image has **no PyRosetta**. They cooperate over a shared `/workspace` volume.

**This service must be running and healthy before `bindcraft-gpu-design` starts in gpu mode.** Start it first.

## When to Use

- Standing up the scoring backend for a full BindCraft design run (the GPU worker depends on it).
- A GPU container is stuck at `waiting for Rosetta service`.
- Debugging relax/interface scoring: `interface_dG`, `interface_nres`, `pr_relax`, `score_interface`.
- Errors: `libgfortran.so.5: cannot open shared object file`, dalphaball failures.

## Quick Reference

| Item | Value |
|------|-------|
| Image | `docker.1ms.run/ai4science/bindcraft:rosetta` |
| Internal port | `8000` |
| Health | `GET /health` → `{"ok": true, "result": "ok"}` |
| Env | `ROSETTA_PORT=8000`, `DALPHABALL_PATH=/app/functions/DAlphaBall.gcc` |
| Shared volume | mount host dir at `/workspace` (same path as GPU container) |
| BINDCRAFT_MODE | `rosetta` (set by image) |

## Start the service

```bash
docker network create bindcraft-net 2>/dev/null || true

docker run -d --name bindcraft-rosetta \
  --network bindcraft-net \
  -v /path/to/workspace:/workspace \
  -p 18000:8000 \
  docker.1ms.run/ai4science/bindcraft:rosetta

# verify (give it ~8s to boot PyRosetta)
sleep 8 && curl -s http://localhost:18000/health     # -> {"ok": true, "result": "ok"}
```

The GPU container reaches it by container name: `-e ROSETTA_URL=http://bindcraft-rosetta:8000` (same `--network`). The `-p 18000:8000` host mapping is only for your own curl checks.

## Critical Rules

1. **Same shared volume, same mount path.** PDB files cross between GPU and rosetta as *file paths over the shared volume, never over the wire*. Both containers must mount the same host dir at `/workspace`, and every `design_path` / PDB path must live under `/workspace`.
2. **Scale with replicas, not threads.** PyRosetta is not thread-safe; each process serializes internally. For more CPU throughput run more replicas (`ROSETTA_REPLICAS` in compose), not more threads.
3. **Start before the GPU worker.** The GPU entrypoint polls `${ROSETTA_URL}/health` for up to ~5 min, then continues (the client retries), but relax/score will fail until this is up.

## Common Mistakes

| Symptom | Fix |
|---------|-----|
| `python: executable file not found` running a custom cmd | micromamba image — use `--entrypoint /usr/local/bin/_entrypoint.sh` then `python ...`, or `/opt/conda/bin/python` directly |
| `libgfortran.so.5: cannot open shared object file` | needs `LD_LIBRARY_PATH=/opt/conda/lib` (baked into current image; old images: add `-e LD_LIBRARY_PATH=/opt/conda/lib`) |
| GPU worker `FileNotFound` on a PDB | path not under `/workspace`, or the two containers mounted different host dirs |
| healthcheck never passes in compose | healthcheck must use absolute `/opt/conda/bin/python` (bare `python` isn't on PATH without entrypoint activation) |

## Related Skills

- **bindcraft-gpu-design** — the GPU design engine that calls this service. Run the rosetta service first.
- **bindcraft-batch-orchestrator** — batch runs; `docker compose up` starts rosetta automatically (healthcheck-gated) before the GPU worker.
