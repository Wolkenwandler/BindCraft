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
params 卷(仅 gpu-worker):                      AF2 权重(首次运行自动下载约 5.3GB)
```

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
   先起 `rosetta`(healthcheck 通过后)→ `gpu-worker` 下载 AF2 权重、认领 job、跑 `bindcraft.py`(relax/score 走 rosetta)→ `orchestrator` 聚合出 `/workspace/combined_final_stats.csv`。

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
