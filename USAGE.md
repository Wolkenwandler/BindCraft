# BindCraft 使用文档

## 概述

BindCraft 是一个基于深度学习的蛋白质binder从头设计管线，通过 AlphaFold2 反向传播（ColabDesign）生成binder骨架，ProteinMPNN 重新设计序列，AF2 预测验证，PyRosetta 界面打分，最终筛选出通过配置过滤条件的高质量设计。

## 环境要求

- **GPU**: NVIDIA GPU (推荐 ≥32GB VRAM)，需 nvidia-container-toolkit
- **Docker**: 已安装且可用

## 快速开始

### 1. 准备靶点蛋白

将靶点 PDB 文件放入工作目录：

```bash
mkdir -p workspace/input_pdbs
cp /path/to/your_target.pdb workspace/input_pdbs/
```

### 2. 编写靶点配置文件

创建 JSON 配置文件，指定靶点和设计参数：

```json
{
    "design_path": "/workspace/designs/MyTarget/",
    "binder_name": "MyTarget",
    "starting_pdb": "/workspace/input_pdbs/target.pdb",
    "chains": "A",
    "target_hotspot_residues": "56",
    "lengths": [65, 150],
    "number_of_final_designs": 5
}
```

**关键参数说明：**

| 参数 | 说明 |
|------|------|
| `design_path` | 输出目录，必须以 `/workspace/` 开头 |
| `binder_name` | 设计名称前缀 |
| `starting_pdb` | 靶点 PDB 文件路径，必须在 `/workspace/` 下 |
| `chains` | 靶点链标识 |
| `target_hotspot_residues` | 靶点热点残基（binder 结合的靶点位置） |
| `lengths` | binder 长度范围 `[min, max]` |
| `number_of_final_designs` | 需要通过的最终设计数量 |

### 3. 启动服务

#### 3.1 创建 Docker 网络

```bash
docker network create bindcraft-net
```

#### 3.2 启动 Rosetta 评分服务（必须先启动）

```bash
docker run -d --name bindcraft-rosetta \
  --network bindcraft-net \
  -v $(pwd)/workspace:/workspace \
  -e ROSETTA_PORT=8000 \
  -e DALPHABALL_PATH=/app/functions/DAlphaBall.gcc \
  docker.1ms.run/ai4science/bindcraft:rosetta
```

#### 3.3 启动 GPU 设计服务

```bash
docker run -d --name bindcraft-gpu \
  --network bindcraft-net \
  --gpus '"device=0"' \
  -v $(pwd)/workspace:/workspace \
  -e START_MODE=all \
  -e ROSETTA_URL=http://bindcraft-rosetta:8000 \
  -p 42001:42001 -p 32210:32210 \
  docker.1ms.run/ai4science/bindcraft:gpu
```

> **注意**: AF2 模型加载需要约 20-30 秒，通过 `curl http://localhost:42001/health` 确认就绪。

### 4. 运行设计

#### 方式一：REST API（推荐）

```bash
curl -X POST http://localhost:42001/api/run_design \
  -H "Content-Type: application/json" \
  -d '{
    "target_settings": {
      "design_path": "/workspace/designs/MyTarget/",
      "binder_name": "MyTarget",
      "starting_pdb": "/workspace/input_pdbs/target.pdb",
      "chains": "A",
      "target_hotspot_residues": "56",
      "lengths": [65, 150],
      "number_of_final_designs": 5
    }
  }'
```

API 返回示例：
```json
{
  "status": "success",
  "msg": "设计完成：5 个binder通过筛选",
  "accepted_designs": 5,
  "accepted_names": ["MyTarget_l98_s393133_mpnn3", ...],
  "elapsed": "0h 38m 0s"
}
```

#### 方式二：命令行（Docker 内直接运行）

```bash
docker exec bindcraft-gpu python -u /app/bindcraft.py \
  --settings /workspace/designs/MyTarget/target_settings.json \
  --filters /app/settings_filters/default_filters.json \
  --advanced /app/settings_advanced/default_4stage_multimer.json
```

### 5. 监控进度

```bash
# 查看容器日志
docker logs -f bindcraft-gpu

# 查看已通过设计数
grep -c "^," workspace/designs/MyTarget/final_design_stats.csv

# 查看 trajectory 进度
tail workspace/designs/MyTarget/trajectory_stats.csv
```

### 6. 停止服务

```bash
docker rm -f bindcraft-gpu bindcraft-rosetta
docker network rm bindcraft-net
```

## 输出文件结构

```
workspace/designs/MyTarget/
├── Accepted/                  ← 通过筛选的设计 PDB 文件
├── Rejected/                  ← 未通过的设计
├── Trajectory/                 ← 原始 hallucination 轨迹
├── MPNN/                       ← MPNN 序列设计
├── final_design_stats.csv     ← 最终通过设计的完整指标
├── mpnn_design_stats.csv      ← 所有 MPNN 设计的验证指标
├── trajectory_stats.csv       ← 所有 trajectory 的评分
└── failure_csv.csv            ← 各指标失败计数
```

## 关键指标说明

| 指标 | 含义 | 判断标准 |
|------|------|----------|
| **pLDDT** | 预测置信度 | 越高越好 (>0.85) |
| **pTM** | 整体结构预测置信度 | 越高越好 (>0.80) |
| **i_pTM** | 界面预测置信度 | 越高越好 (>0.70)，反映binder-靶点相互作用可靠性 |
| **pAE** | 预测对齐误差 | 越低越好 (<0.30) |
| **i_pAE** | 界面预测对齐误差 | 越低越好 |
| **ΔG** | 结合自由能 (kcal/mol) | 越负越好 (< -30) |
| **SC** | 形状互补性 | 越高越好 (>0.60) |
| **dSASA** | 界面埋藏面积 | 越大越好 |
| **RMSD** | 靶点结构偏差 | 越低越好 (<1.0) |

## 设计流程

每个 trajectory 经历以下阶段：

1. **Hallucination** (4-stage): Logits → Softmax → One-hot → PSSM Semigreedy，通过 AF2 反向传播生成 binder 骨架
2. **Relax + Score**: PyRosetta FastRelax 优化 + 界面打分
3. **MPNN 重设计**: 生成多条序列变体
4. **AF2 验证**: 对每条 MPNN 序列预测复合物结构，多个模型取平均
5. **过滤**: 通过 `default_filters.json` 中的阈值条件

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| API health 无响应 | 等待 AF2 模型加载完成（约 30 秒） |
| `predict_complex` 500 错误 | 检查 Rosetta 容器是否就绪：`curl http://localhost:8000/health` |
| GPU 显存不足 | 同一 GPU 只运行一个 worker |
| 设计通过率低 | 放宽过滤条件，或调整 `lengths` 范围 |
| 端口被占用 | 映射到其他端口：`-p 42003:42001` |
