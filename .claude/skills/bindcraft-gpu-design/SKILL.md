---
name: bindcraft-gpu-design
description: Use when designing a de novo protein binder with BindCraft — hallucinating a binder backbone against a target, predicting a binder-target complex, or running the full design loop on a GPU. Covers the bindcraft:gpu image in worker/api/mcp/all modes and its REST + MCP interfaces. This is the main BindCraft design engine; it needs bindcraft-rosetta-service running first.
---

# BindCraft GPU Design Engine

## Overview

The `ai4science/bindcraft:gpu` image is the **GPU compute core** of BindCraft: AlphaFold2 hallucination + ProteinMPNN redesign + AF2 re-prediction. It has jax/colabdesign and the AF2 weights baked in at `/app/params`, but **no PyRosetta** — it forwards relax/score over HTTP to the **bindcraft-rosetta-service** (`BINDCRAFT_MODE=gpu`). A full design run therefore needs the rosetta service up first.

One image, four run modes via `START_MODE`:

| START_MODE | What runs | Ports |
|------------|-----------|-------|
| `worker` (default) | original file-queue batch worker (`worker.py`) | — |
| `api` | FastAPI design server | 42001 |
| `mcp` | FastMCP tool server (needs an API reachable via `BINDCRAFT_API_URL`) | 32210 |
| `all` | API + MCP in one container | 42001 + 32210 |

## When to Use

- "Use BindCraft to design a binder against <target PDB>."
- Hallucinate a backbone / predict a complex / run the full hallucinate→MPNN→validate→filter loop.
- Stand up the BindCraft REST API or MCP server for remote/agent calls.
- Triggers: binder design, hallucinate, ColabDesign, AF2, ProteinMPNN, i_pTM, pLDDT, PDL1.

## Prerequisites (do this first)

1. **Start the rosetta service** — see **bindcraft-rosetta-service**. Get its URL (e.g. `http://bindcraft-rosetta:8000` on a shared `--network`).
2. **Shared volume**: mount the *same* host dir at `/workspace` on both containers. Put the target PDB and `design_path` under `/workspace`.
3. **GPU**: host needs nvidia-container-toolkit; pass `--gpus all`. One complex can need ~32 GB VRAM — keep one worker per GPU.

## Run modes

### API + MCP server (recommended for interactive / remote use)

```bash
docker run -d --name bindcraft-api \
  --network bindcraft-net --gpus all \
  -v /path/to/workspace:/workspace \
  -e START_MODE=all \
  -e ROSETTA_URL=http://bindcraft-rosetta:8000 \
  -p 42001:42001 -p 32210:32210 \
  docker.1ms.run/ai4science/bindcraft:gpu

# startup takes ~20-30s (loads AF2 models). Then:
curl -s http://localhost:42001/health
```

### REST endpoints (port 42001)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | GPU availability + status |
| `/api/hallucinate_binder` | POST | backbone hallucination only (~5 min) |
| `/api/predict_complex` | POST | predict binder-target complex for a sequence (+ rosetta relax) |
| `/api/run_design` | POST | full design loop (can run hours) |

Hallucinate example:

```bash
curl -X POST http://localhost:42001/api/hallucinate_binder \
  -H "Content-Type: application/json" \
  -d '{
    "binder_name": "MyBinder",
    "starting_pdb": "/workspace/input_pdbs/target.pdb",
    "chains": "A",
    "target_hotspot_residues": "56",
    "length": 80,
    "seed": 42,
    "design_path": "/workspace/designs/test/"
  }'
# -> {status, design_name, trajectory_pdb, sequence, metrics:{plddt,...}, elapsed_seconds}
```

Feed the returned `sequence` + `trajectory_pdb` into `/api/predict_complex` to validate. Full request schemas and a Python MCP client are in `README-docker.md` ("API / MCP 远程调用").

### MCP tools (port 32210, `streamable-http` at `/mcp`)

`bindcraft_health`, `hallucinate_binder`, `predict_binder_complex`, `run_binder_design`. Point a client at `http://<host>:32210/mcp`.

### Single canonical CLI run (no API)

Inside the GPU env (or `START_MODE=worker` with a queue), the unmodified entry point is:

```bash
python -u ./bindcraft.py \
  --settings ./settings_target/PDL1.json \
  --filters  ./settings_filters/default_filters.json \
  --advanced ./settings_advanced/default_4stage_multimer.json
```

## Critical Rules

1. **Rosetta first.** Without a healthy rosetta service, relax/score (and thus `predict_complex`/`run_design`) fail.
2. **All paths under `/workspace`.** `starting_pdb`, `trajectory_pdb`, `design_path` must resolve inside the shared volume the rosetta container also mounts.
3. **One worker per GPU.** Don't oversubscribe a single GPU.

## Common Mistakes

| Symptom | Fix |
|---------|-----|
| `bind: address already in use` on 42001/32210 | host port taken — map an alternate host port `-p 42003:42001` |
| API stuck / 500 from `predict_complex` | rosetta service not reachable; check `ROSETTA_URL` and that both share `--network` + `/workspace` |
| `python: executable file not found` for a custom command | micromamba image — use `--entrypoint /usr/local/bin/_entrypoint.sh` then `python ...` |
| jax shows only CPU | forgot `--gpus all`, or no nvidia-container-toolkit on host |
| non-JSON-serializable response error | use the current image; the API converts jax/numpy metrics to plain JSON (`_make_json_safe`) |

## Related Skills

- **bindcraft-rosetta-service** — REQUIRED dependency; start it before this. A full design = rosetta + this GPU engine.
- **bindcraft-batch-orchestrator** — for batching many targets via `docker compose` instead of calling the API per target.
