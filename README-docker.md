# BindCraft — Docker / 单机批量运行(三镜像)

把 BindCraft 拆成**三个独立环境**,单机用 Docker Compose 协同,支持**批量输入、批量计算**。科学代码仅做 3 处最小、可逆、行为中性的改动(用 `BINDCRAFT_MODE` 环境变量切换;不设时本地/SLURM 行为完全不变)。

## 架构

| 镜像 | 环境 | 职责 | GPU |
|------|------|------|:---:|
| `bindcraft-orchestrator` | `python:3.10-slim`(无 jax/pyrosetta) | 批量入队、监控、跨 target 聚合 | 否 |
| `bindcraft-gpu` | jax + colabdesign + biopython(**无 pyrosetta**) | 跑 `bindcraft.py` 设计循环;Rosetta 调用走 HTTP | 是 |
| `bindcraft-rosetta` | pyrosetta + DAlphaBall + dssp(**无 jax**) | HTTP 服务:relax / score_interface / align / rmsd | 否 |

### 协同方式

GPU 进程(`BINDCRAFT_MODE=gpu`)把对 Rosetta 的 5 个调用(`pr_relax`、`score_interface`、`unaligned_rmsd`、`align_pdbs`、`pr.init`)改为 HTTP 转发到 `rosetta` 服务。**PDB 文件经共享 `/workspace` 卷传递,网线上只走文件路径**——所以 `gpu-worker` 与 `rosetta` 必须把 workspace 卷挂在相同路径 `/workspace`,且 target 的 `design_path` 要在 `/workspace` 下。

```
workspace 卷(orchestrator / gpu-worker / rosetta 共享):
  queue/pending|processing|done|failed|logs/   文件队列(os.rename 原子认领,多副本安全)
  designs/<name>/                              各 target 产物 + 中间 PDB
  combined_final_stats.csv                     聚合结果
```

> AF2 权重(约 5.3GB)在**构建阶段直接打包进 `bindcraft-gpu` 镜像**(`/app/params`),运行时无需下载、无需挂卷。代价:gpu 镜像很大(CUDA + jax + 5.3GB 权重,约 10GB+),构建慢、推送 Docker Hub 耗时。下载层在 `COPY . /app` 之前,改代码不会让它失效。

### 涉及的代码改动(env-gated,可逆)

- `functions/__init__.py`、`functions/colabdesign_utils.py`:按 `BINDCRAFT_MODE`(local/gpu/rosetta)选择导入 `pyrosetta_utils`(本地)还是 `rosetta_client`(HTTP 客户端),rosetta 模式跳过 colabdesign。
- `functions/generic_utils.py`:`import jax` 改为 `check_jax_gpu()` 内惰性导入(让无 jax 的 Rosetta 镜像能导入)。
- 新增 `functions/rosetta_client.py`(stdlib HTTP 客户端 shim)、`docker/rosetta_service.py`(stdlib HTTP 服务)。

`bindcraft.py` **未改动**:`pr.init(...)` 在 gpu 模式下命中 `rosetta_client` 的 no-op shim,真正的 `pr.init` 在 Rosetta 服务里执行。

## 前置条件

- Docker + Docker Compose。
- GPU 运行需宿主装 NVIDIA 驱动 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## 使用步骤

1. **批量输入**:在仓库根目录建 `batch_inputs/`,放入一个或多个 target 设置 JSON(schema 同 `settings_target/*.json`),`design_path` 指向 `/workspace`:
   ```json
   {
     "design_path": "/workspace/designs/PDL1/",
     "binder_name": "PDL1",
     "starting_pdb": "/app/example/PDL1.pdb",
     "chains": "A",
     "target_hotspot_residues": "56",
     "lengths": [65, 150],
     "number_of_final_designs": 100
   }
   ```

2. **(可选)`.env`**:
   ```
   WORKERS=1            # gpu-worker 副本;单 GPU 保持 1
   ROSETTA_REPLICAS=1   # Rosetta 服务副本;多 worker 时可加(compose DNS 轮询)
   CUDA=12.4
   ```

3. **构建并运行**:
   ```bash
   docker compose up --build
   ```
   先起 `rosetta`(healthcheck 通过后)→ `gpu-worker` 认领 job、跑 `bindcraft.py`(权重已在镜像内,relax/score 走 rosetta)→ `orchestrator` 聚合出 `/workspace/combined_final_stats.csv`。

4. **结束**:`gpu-worker`/`orchestrator` 完成后会退出,但 `rosetta` 是常驻服务,需手动停止:
   ```bash
   docker compose down
   ```

5. **取产物**(workspace 卷的 `designs/<name>/`):
   ```bash
   docker run --rm -v bindcraft_workspace:/w -v "$PWD/out":/out alpine cp -r /w/designs /out/
   ```

## 扩容与单卡注意

- 一个 binder 复合物可能需约 32GB 显存。**单 GPU 保持 `WORKERS=1`**;多 GPU 设为卡数。
- Rosetta 是 CPU 瓶颈:多 GPU worker 时把 `ROSETTA_REPLICAS` 调大,compose 服务 DNS 自动轮询。PyRosetta 非线程安全,每个 rosetta 进程内部串行(加锁),靠多副本并发。

## GitHub Actions

`.github/workflows/docker-build.yml`(约定沿用 Common_project):

- `push`(main / `v*`)与手动 `workflow_dispatch` 触发。
- **直接构建并推送**三个镜像到 **Docker Hub**:
  - `<DOCKER_USERNAME>/bindcraft:gpu`(+ `:gpu-<shortsha>`)
  - `<DOCKER_USERNAME>/bindcraft:rosetta`(+ `:rosetta-<shortsha>`)
  - `<DOCKER_USERNAME>/bindcraft:orchestrator`(+ `:orchestrator-<shortsha>`)
- `secrets.DOCKER_USERNAME` / `secrets.DOCKER_PASSWORD` 登录;激进磁盘清理(gpu 镜像大)+ buildx GHA 缓存;`max-parallel: 1` 串行构建。
- `validate` job:hadolint + `docker compose config`。

**需在仓库 Settings → Secrets 配置**:`DOCKER_USERNAME`、`DOCKER_PASSWORD`(Docker Hub access token)。

> ⚠️ GitHub runner 没有 GPU,CI 只构建/推送镜像,不能跑流水线。实际运行在 GPU 机器上 `docker compose up`。

别的机器拉镜像运行,`.env` 指向 Docker Hub:
```
GPU_IMAGE=<DOCKER_USERNAME>/bindcraft:gpu
ROSETTA_IMAGE=<DOCKER_USERNAME>/bindcraft:rosetta
ORCH_IMAGE=<DOCKER_USERNAME>/bindcraft:orchestrator
```

## 本地验证

```bash
# 构建
docker build -f docker/Dockerfile.orchestrator -t bindcraft-orchestrator:local .
docker build -f docker/Dockerfile.rosetta      -t bindcraft-rosetta:local .
docker build -f docker/Dockerfile.gpu          -t bindcraft-gpu:local .      # 较慢

# 依赖隔离(三镜像各自只装自己的栈)
docker run --rm bindcraft-gpu:local      python -c "import jax, colabdesign; print('ok, no pyrosetta needed')"
docker run --rm bindcraft-rosetta:local  python -c "import pyrosetta; print('ok, no jax needed')"
docker run --rm bindcraft-orchestrator:local python -c "import pandas; print('ok')"

# GPU 可见(需 nvidia-container-toolkit)
docker run --rm --gpus all bindcraft-gpu:local python -c "import jax; print(jax.devices())"
```

## 构建后测试

镜像构建好后,按下面四级逐步验证。**Level 1–2 不需要 GPU**,可在任意机器上确认三镜像拆分与 RPC 边界正确;Level 3–4 需要 GPU。

### Level 1 — 依赖隔离自检(无 GPU)

每个镜像构建时已内置 `RUN ... import functions` 自检;若 `docker build` 成功即说明三态导入(local/gpu/rosetta)无误。可随时复跑上面"本地验证"里的导入命令,确认:
- gpu 镜像有 jax/colabdesign、**无 pyrosetta**;
- rosetta 镜像有 pyrosetta、能在 `BINDCRAFT_MODE=rosetta` 下导入 `functions`(**不拉 jax**);
- orchestrator 仅 pandas。

### Level 2 — Rosetta RPC 边界(无 GPU,关键)

这是三镜像拆分最核心的新增链路:GPU 进程 → HTTP → Rosetta 服务,PDB 经共享卷交接。一条命令端到端验证(启动 rosetta 服务、`/health`、用内置 `example/PDL1.pdb` 真跑一次 `pr_relax`、校验产物,然后清理):

```bash
docker compose build            # 或拉取镜像并在 .env 设置 *_IMAGE
bash docker/smoke_test.sh
```

脚本输出 `summary: N passed, 0 failed` 即通过。它覆盖了依赖隔离 + `pr_relax` over HTTP 的真实往返,**不需要 GPU**——是验证本次重构是否正确最快、最有价值的一步。

### Level 3 — GPU 可见性(需 GPU)

```bash
docker run --rm --gpus all bindcraft-gpu:local python -c "import jax; print(jax.devices())"
# 期望打印包含 gpu/cuda 的设备列表;若为空,检查宿主 nvidia-container-toolkit。
```

### Level 4 — 端到端批量冒烟(需 GPU,较慢)

用内置的精简靶标(`docker/examples/PDL1_smoke.json`,只求 1 个最终设计、binder 长度很短)跑通完整链路:

```bash
mkdir -p batch_inputs
cp docker/examples/PDL1_smoke.json batch_inputs/

# 用无过滤器加快出结果(可选,通过 DEFAULT_FILTERS 覆盖)
WORKERS=1 DEFAULT_FILTERS=/app/settings_filters/no_filters.json docker compose up --build
```

期间应观察到:
1. `rosetta` 服务 healthy;
2. `gpu-worker` 认领 `PDL1_smoke.json`、`bindcraft.py` 启动设计循环(权重已打包进镜像);
3. 日志里 relax/score 阶段经 rosetta 服务执行(不在 gpu 容器内导入 pyrosetta);
4. 产物落到 workspace 卷 `designs/PDL1_smoke/`;
5. `orchestrator` 打印每靶汇总并生成 `/workspace/combined_final_stats.csv`。

结束后:
```bash
docker compose down                      # 停掉常驻的 rosetta 服务
# 查看队列处理情况与日志
docker run --rm -v bindcraft_workspace:/w alpine sh -c "ls -R /w/queue && echo '---' && tail -n 40 /w/queue/logs/*.log"
# 导出产物
docker run --rm -v bindcraft_workspace:/w -v "$PWD/out":/out alpine cp -r /w/designs /out/
```

> 提示:冒烟跑只为验证链路连通,不代表能产出合格 binder(真实运行常需数百~数千条 trajectory)。若只想验证"重构没破坏科学逻辑",Level 2 + Level 4 的链路连通性即足够;严格的科学回归应在单机 `local` 模式(不设 `BINDCRAFT_MODE`)与拆分模式间对比同一 seed 的输出。

### 常见排查

| 现象 | 排查 |
|------|------|
| `gpu-worker` 卡在 "waiting for Rosetta service" | `docker compose logs rosetta`;确认 healthcheck 通过、`ROSETTA_URL=http://rosetta:8000` |
| RPC 报 `FileNotFound` | target 的 `design_path` 未在 `/workspace` 下,或 rosetta 与 gpu-worker 未挂同一 workspace 卷 |
| rosetta 启动报 dalphaball/dssp 相关错误 | 确认 `Dockerfile.rosetta` 里两个二进制已 `chmod +x`、`DALPHABALL_PATH` 正确 |
| 多 worker 时 relax 很慢 | 提高 `ROSETTA_REPLICAS`(PyRosetta 单进程内串行,靠多副本并发) |

## API / MCP 远程调用

GPU 镜像支持三种运行模式,通过 `START_MODE` 环境变量切换:

| START_MODE | 说明 | 端口 |
|------------|------|------|
| `worker` (默认) | 原有批量处理模式,向后兼容 | - |
| `api` | FastAPI 服务,包装核心设计管线 | 42001 |
| `mcp` | FastMCP 服务,暴露 MCP 工具供远程客户端调用 | 32210 |
| `all` | 同一容器内同时启动 API + MCP | 42001 + 32210 |

### 架构

```
┌──────────────────────┐     HTTP      ┌─────────────────────┐     MCP/HTTP     ┌──────────────────────┐
│  functions/*.py      │◀──────────────│  api_bindcraft.py   │◀────────────────│  server_bindcraft.py │
│  (AF2/MPNN/Rosetta)  │              │  FastAPI :42001     │                 │  FastMCP :32210      │
└──────────────────────┘              └─────────────────────┘                 └──────────────────────┘
```

API 层直接调用 BindCraft 核心函数(`binder_hallucination`, `mpnn_gen_sequence`, `predict_binder_complex` 等);MCP 层定义 tool,内部通过 HTTP 调用 API 层。Rosetta 调用仍然走 HTTP 转发到 Rosetta 服务容器。

### 启动方式

**前提**: Rosetta 服务已运行,且 API 容器能访问它(同网络 + 共享卷)。

```bash
# 1. 启动 Rosetta 服务
docker run -d --name rosetta --network host \
  -v /path/to/workspace:/workspace \
  ai4science/bindcraft:rosetta

# 2. 启动 API + MCP (推荐)
docker run -d --name bindcraft-api --network host --gpus all \
  -v /path/to/workspace:/workspace \
  -e START_MODE=all \
  -e ROSETTA_URL=http://localhost:8000 \
  ai4science/bindcraft:gpu

# 或仅启动 API
docker run -d --name bindcraft-api --network host --gpus all \
  -v /path/to/workspace:/workspace \
  -e START_MODE=api \
  -e ROSETTA_URL=http://localhost:8000 \
  ai4science/bindcraft:gpu
```

**重要**: PDB 文件通过共享卷传递,Roestta 和 GPU 容器必须将 workspace 挂载到相同路径(推荐 `/workspace`),且设计产物的 `design_path` 需在 `/workspace` 下。

### API 端点

#### `GET /health`

健康检查,返回 GPU 可用性和服务状态。

```bash
curl http://localhost:42001/health
```

响应:
```json
{
  "status": "healthy",
  "service": "BindCraft API",
  "gpu_available": true,
  "gpu_devices": ["cuda:0", "cuda:1"],
  "bindcraft_mode": "gpu"
}
```

#### `POST /api/hallucinate_binder`

运行 binder backbone hallucination(仅 AF2 反向传播,不含 MPNN 重设计)。

```bash
curl -X POST http://localhost:42001/api/hallucinate_binder \
  -H "Content-Type: application/json" \
  -d '{
    "binder_name": "MyBinder",
    "starting_pdb": "/workspace/input_pdbs/target.pdb",
    "chains": "A",
    "target_hotspot_residues": "56,120",
    "length": 80,
    "seed": 42,
    "design_path": "/workspace/designs/test/"
  }'
```

请求参数:

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `binder_name` | string | 是 | 设计名称前缀 |
| `starting_pdb` | string | 是 | 靶标蛋白 PDB 文件路径(需在共享卷内) |
| `chains` | string | 是 | 靶标链 ID,如 `"A"` |
| `target_hotspot_residues` | string | 是 | 热点残基编号,逗号分隔,如 `"56,120"` |
| `length` | int | 是 | binder 长度(残基数) |
| `seed` | int | 否 | 随机种子(不指定则随机生成) |
| `design_path` | string | 是 | 输出目录(需在共享卷内) |
| `advanced_settings` | dict | 否 | 高级协议参数覆盖 |

响应:
```json
{
  "status": "success",
  "design_name": "MyBinder_l80_s42",
  "trajectory_pdb": "/workspace/designs/test/Trajectory/MyBinder_l80_s42.pdb",
  "terminated": "",
  "sequence": "SEEARDRFLRKMKPVFEEHVWRFRQMPNPTD...",
  "metrics": {
    "plddt": 0.91,
    "ptm": 0.85,
    "i_ptm": 0.78,
    "pae": 0.18,
    "i_pae": 0.15,
    "loss": 4.52,
    "helix": 1.02
  },
  "elapsed_seconds": 330.8
}
```

#### `POST /api/predict_complex`

对给定的 binder 序列预测其与靶标蛋白的复合物结构(AF2 多模板预测 + Rosetta FastRelax)。

```bash
curl -X POST http://localhost:42001/api/predict_complex \
  -H "Content-Type: application/json" \
  -d '{
    "binder_sequence": "SEEARDRFLRK...",
    "target_pdb": "/workspace/input_pdbs/target.pdb",
    "target_chain": "A",
    "binder_length": 80,
    "trajectory_pdb": "/workspace/designs/test/Trajectory/MyBinder_l80_s42.pdb",
    "design_path": "/workspace/designs/test/",
    "design_name": "pred_1"
  }'
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `binder_sequence` | string | 是 | Binder 氨基酸序列(单字母大写) |
| `target_pdb` | string | 是 | 靶标蛋白 PDB 路径 |
| `target_chain` | string | 是 | 靶标链 ID |
| `binder_length` | int | 是 | Binder 残基数 |
| `trajectory_pdb` | string | 是 | Hallucination 产生的 trajectory PDB 路径(用作模板) |
| `design_path` | string | 是 | 输出目录 |
| `design_name` | string | 是 | 预测名称 |
| `advanced_settings` | dict | 否 | AF2 预测参数覆盖 |
| `filters` | dict | 否 | 逐模型过滤阈值 |

响应:
```json
{
  "status": "success",
  "design_name": "pred_1",
  "sequence": "SEEARDRFLRK...",
  "models": {
    "1": {
      "pLDDT": 0.79,
      "pTM": 0.64,
      "i_pTM": 0.25,
      "pAE": 0.46,
      "i_pAE": 0.65,
      "pdb": "/workspace/designs/test/MPNN/pred_1_model1.pdb",
      "relaxed_pdb": "/workspace/designs/test/MPNN/Relaxed/pred_1_model1.pdb"
    },
    "2": {
      "pLDDT": 0.85,
      "pTM": 0.58,
      "i_pTM": 0.06,
      "pAE": 0.61,
      "i_pAE": 0.91,
      "pdb": "/workspace/designs/test/MPNN/pred_1_model2.pdb",
      "relaxed_pdb": "/workspace/designs/test/MPNN/Relaxed/pred_1_model2.pdb"
    }
  }
}
```

#### `POST /api/run_design`

运行完整的 BindCraft 设计管线(hallucination → relax → score → MPNN → validate → filter),循环直至产出足够的合格设计。

```bash
curl -X POST http://localhost:42001/api/run_design \
  -H "Content-Type: application/json" \
  -d '{
    "target_settings": {
      "design_path": "/workspace/designs/PDL1/",
      "binder_name": "PDL1",
      "starting_pdb": "/workspace/input_pdbs/PDL1.pdb",
      "chains": "A",
      "target_hotspot_residues": "56",
      "lengths": [65, 150],
      "number_of_final_designs": 10
    },
    "filters": {"pLDDT": 0.85, "i_pTM": 0.70}
  }'
```

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `target_settings` | dict | 是 | 靶标设置,结构同 `settings_target/*.json` |
| `advanced_settings` | dict | 否 | 高级协议参数(不提供则用默认 4-stage multimer) |
| `filters` | dict | 否 | 过滤阈值(不提供则用默认;设 `null` 禁用某项过滤) |

> **注意**: 完整设计管线可能运行数小时。建议先用 `hallucinate_binder` + `predict_complex` 分步调试参数,确认无误后再用 `run_design` 批量生产。

### MCP 工具

MCP 服务(FastMCP, `streamable-http` 传输)提供 4 个工具,可用任意 MCP 客户端调用。

MCP 服务端点: `http://<host>:32210/mcp`

#### 工具列表

| 工具 | 说明 | 超时 |
|------|------|------|
| `bindcraft_health` | 健康检查 + GPU 状态 | 10s |
| `run_binder_design` | 完整设计管线 | 24h |
| `hallucinate_binder` | Binder backbone hallucination | 2h |
| `predict_binder_complex` | 复合物结构预测 | 2h |

#### MCP 客户端示例 (Python)

```python
import asyncio, json
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

MCP_URL = "http://localhost:32210/mcp"

async def main():
    transport = streamablehttp_client(MCP_URL)
    read, write, get_session_id = await transport.__aenter__()
    session_ctx = ClientSession(read, write)
    session = await session_ctx.__aenter__()
    await session.initialize()

    # 健康检查
    response = await session.call_tool("bindcraft_health", arguments={})
    print(response.content[0].text)

    # Hallucinate backbone
    response = await session.call_tool(
        "hallucinate_binder",
        arguments={
            "binder_name": "test",
            "starting_pdb": "/workspace/input_pdbs/PDL1.pdb",
            "chains": "A",
            "target_hotspot_residues": "56",
            "length": 80,
            "seed": 42,
            "design_path": "/workspace/designs/test/",
        }
    )
    result = json.loads(response.content[0].text)
    print(f"pLDDT: {result['metrics']['plddt']}")
    trajectory_pdb = result["trajectory_pdb"]
    sequence = result["sequence"]

    # 预测复合物结构
    response = await session.call_tool(
        "predict_binder_complex",
        arguments={
            "binder_sequence": sequence,
            "target_pdb": "/workspace/input_pdbs/PDL1.pdb",
            "target_chain": "A",
            "binder_length": 80,
            "trajectory_pdb": trajectory_pdb,
            "design_path": "/workspace/designs/test/",
            "design_name": "predict_1",
        }
    )
    result = json.loads(response.content[0].text)
    for m, stats in result["models"].items():
        print(f"Model {m}: pLDDT={stats['pLDDT']}, i_pTM={stats['i_pTM']}")

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

asyncio.run(main())
```

#### 在 Claude Code 中使用

如果 MCP 服务部署在可通过 HTTP 访问的地址,可在 Claude Code 中配置为 MCP 工具:

```json
{
  "mcpServers": {
    "bindcraft": {
      "type": "streamable-http",
      "url": "http://<host>:32210/mcp"
    }
  }
}
```

### 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `START_MODE` | `worker` | 容器启动模式: `worker` / `api` / `mcp` / `all` |
| `API_PORT` | `42001` | API 服务监听端口 |
| `MCP_PORT` | `32210` | MCP 服务监听端口 |
| `BINDCRAFT_API_URL` | `http://localhost:42001` | MCP 服务连接 API 的地址(仅 mcp 模式需要) |
| `BINDCRAFT_MODE` | `gpu` | 依赖栈模式(gpu 镜像固定为 `gpu`) |
| `ROSETTA_URL` | `http://rosetta:8000` | Rosetta 服务地址 |

### 验证

```bash
# API 健康检查
curl http://localhost:42001/health

# 完整链路验证: Hallucination → Predict
curl -X POST http://localhost:42001/api/hallucinate_binder \
  -H "Content-Type: application/json" \
  -d '{"binder_name":"smoke","starting_pdb":"/workspace/input_pdbs/target.pdb","chains":"A","target_hotspot_residues":"1","length":50,"seed":1,"design_path":"/workspace/smoke/"}'

# 用输出的 trajectory_pdb 和 sequence 继续测试 predict_complex ...
```

