---
name: bindcraft-batch-orchestrator
description: Use when running BindCraft on many targets at once — batch-queuing target JSONs, aggregating per-target results, or standing up the full three-image pipeline with docker compose. Covers the bindcraft:orchestrator image (enqueue + aggregate) and the combined_final_stats.csv output.
---

# BindCraft Batch Orchestrator

## Overview

The `ai4science/bindcraft:orchestrator` image is the **bookkeeping layer** for batch BindCraft runs: it has no jax and no PyRosetta (just pandas). It does two jobs, controlled by env vars:

- **Enqueue** (`ENQUEUE=1`): copy target JSONs from `${BATCH_IN}` into `${QUEUE}/pending/`.
- **Aggregate** (`WAIT=1`): wait for jobs to land in `${QUEUE}/done/`, then merge every target's `final_design_stats.csv` into `${WORKSPACE}/combined_final_stats.csv` (each row prefixed with the target name).

The GPU worker (`bindcraft-gpu-design` in `worker` mode) atomically claims each pending job via `os.rename` and runs the unmodified `bindcraft.py`. The orchestrator never touches GPUs.

## When to Use

- "Run BindCraft over these N targets and give me a combined results table."
- Standing up the whole pipeline (rosetta + gpu-worker + orchestrator) with one command.
- Triggers: batch design, multiple targets, queue, combined_final_stats.csv, aggregate.

## Quick Reference

| Item | Value |
|------|-------|
| Image | `docker.1ms.run/ai4science/bindcraft:orchestrator` |
| Entry | `python -u /app/orchestrator.py` (plain python image — `python` works directly) |
| Inputs | `${BATCH_IN}/*.json` (schema = `settings_target/*.json`; `design_path` under `/workspace`) |
| Queue | `${QUEUE}/{pending,processing,done,failed,logs}/` |
| Output | `${WORKSPACE}/combined_final_stats.csv` |
| Env | `BATCH_IN`, `QUEUE`, `WORKSPACE`, `ENQUEUE`, `WAIT`, `POLL` |

## Recommended: full pipeline via docker compose

`docker-compose.yml` wires all three images on a shared `workspace` volume; rosetta is healthcheck-gated before the GPU worker starts.

```bash
# 1. drop one or more target JSONs in ./batch_inputs/ (design_path -> /workspace/designs/<name>/)
mkdir -p batch_inputs
cp docker/examples/PDL1_smoke.json batch_inputs/

# 2. (optional) .env:  WORKERS=1  ROSETTA_REPLICAS=1  CUDA=12.4
#    prebuilt images: GPU_IMAGE / ROSETTA_IMAGE / ORCH_IMAGE

# 3. run — orchestrator enqueues, gpu-worker designs, orchestrator aggregates
docker compose up --build

# 4. rosetta is a long-running service; stop it when done
docker compose down
```

Result lands in the `workspace` volume at `combined_final_stats.csv` and per-target `designs/<name>/`. Export:

```bash
docker run --rm -v bindcraft_workspace:/w -v "$PWD/out":/out alpine cp -r /w/designs /out/
```

## Standalone orchestrator (enqueue / aggregate separately)

```bash
# enqueue only
docker run --rm -v "$HOST":/workspace -v "$PWD/batch_inputs":/batch_inputs:ro \
  -e BATCH_IN=/batch_inputs -e QUEUE=/workspace/queue -e WORKSPACE=/workspace \
  -e ENQUEUE=1 -e WAIT=0 \
  docker.1ms.run/ai4science/bindcraft:orchestrator

# aggregate only (after jobs are in queue/done/)
docker run --rm -v "$HOST":/workspace \
  -e QUEUE=/workspace/queue -e WORKSPACE=/workspace -e ENQUEUE=0 -e WAIT=1 -e POLL=2 \
  docker.1ms.run/ai4science/bindcraft:orchestrator
```

## Critical Rules

1. **`design_path` must be under `/workspace`** in every target JSON — that's the shared volume all three images mount.
2. **Single GPU → `WORKERS=1`.** Scale GPU throughput with `WORKERS` (cards), CPU/rosetta throughput with `ROSETTA_REPLICAS`.
3. **Smoke runs prove plumbing, not quality.** `PDL1_smoke.json` + `no_filters.json` checks the pipeline connects; real designs need hundreds–thousands of trajectories.

## Common Mistakes

| Symptom | Fix |
|---------|-----|
| `combined_final_stats.csv` empty | jobs never reached `queue/done/`, or `final_design_stats.csv` missing under each `design_path` |
| worker idles, no jobs claimed | nothing enqueued — confirm `batch_inputs/*.json` and the enqueue step ran |
| paths unresolved across containers | all services must mount the same host dir / named volume at `/workspace` |

## Related Skills

- **bindcraft-rosetta-service** — the scoring backend; `compose up` starts it first (healthcheck-gated).
- **bindcraft-gpu-design** — the per-target design engine the worker runs; use it directly (API/MCP) for single targets instead of batching.
