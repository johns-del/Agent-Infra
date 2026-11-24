# STRATA v14

请求先经中转层进入节点守护进程。守护进程冻结多轮系统前缀、同步工具生命周期，并激活引擎侧的 IntervalCache 和 PrefillGovernor。引擎派生脚本安装 CausalPrefixStabilizer 与 PCIe IPC 预填充配置。

## 节点基线

| 角色 | GPU | TP/EP | 显存利用率 | 最大运行请求 | Decode Graph | 特定调优 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dot12 | 4× RTX PRO 6000 96 GB | 4/4 | 0.92 | 16 | 16 | 该分片不使用 dot13 FP8 配置 |
| dot13 | 8× RTX PRO 5000 72 GB | 8/8 | 0.74，可用 `MEM_FRAC` 覆盖 | 128 | 64 | 10 份 4096-row FP8 GEMM 调优文件 |
| dot14 | 8× RTX PRO 5000 72 GB | 8/8 | 0.74 | 128 | 64 | PCIe IPC、DSpark、HiCache |

三个角色使用 FP8 KV、DSpark block size 5、HiCache write-back 和 CPU Engram。上下文默认由模型配置推导，实测模型配置为 1,048,576 tokens。PrefillGovernor 使用 balanced 配置，空闲分块 4096，解码压力下分块 2048。 [历史节点结果](../results/node-baselines.md)

## 部署节点

在已经有目标 GPU、模型权重、CUDA 13、Torch 及相应 SGLang/FlashInfer 运行依赖的 Linux 环境，从此目录运行。中转机与推理机分开时，在节点上为 `DAEMON_ALLOWED_CLIENTS` 填写中转机 IP。

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017-NVFP4
export PYTHON_BIN=/path/to/inference-venv/bin/python
export CUDA_HOME=/usr/local/cuda
export DAEMON_ALLOWED_CLIENTS=127.0.0.1,::1,192.0.2.13
bash launch_node_service.sh dot14 --stage-only
bash launch_node_service.sh dot14
curl -fsS http://127.0.0.1:32683/orvix_daemon_status
```

按节点替换角色。运行时默认使用 [随仓库的 T1 SGLang 源码](../t1-base/overlay/) 派生，也可用 `BASE_SOURCE` 指定通过原生接口合同校验的源码根目录。`STRATA_STATE_DIR` 可覆盖默认的 `.state/<角色>`。准备阶段只写文件。正式启动使用该状态目录下的私有 tmux socket，需在目标宿主机确认端口及 GPU 已分配。引擎监听 `127.0.0.1:32681`，节点代理监听 `:32683`。dot13 的 FP8 包默认安装，配对复测默认参数时可用 `ORVIX_DISABLE_DOT13_FP8_CONFIGS=1` 跳过。

## 跨机负载均衡

下列地址为文档示例，运行前替换为节点代理的实际地址。路由运行环境需要 `sglang-router==0.3.2`，中转的 Python 环境使用仓库根目录的 HTTP 依赖。两个启动脚本分别在独立终端或服务会话运行。

```bash
export DOT14_URL=http://192.0.2.14:32683
export DOT12_URL=http://192.0.2.12:32683
export ROUTER_PYTHON_BIN=/path/to/router-venv/bin/python
export PYTHON_BIN=/path/to/relay-venv/bin/python
bash start_router.sh
# 在另一个具有相同环境变量的会话
bash start_priority_relay.sh
# 在第三个终端检查
curl -fsS http://127.0.0.1:32859/orvix_relay_status
```

| 节点 | 权重 | 软并发上限 | 粘性并发上限 | KV 阈值 | 队列上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dot14 | 3 | 18 | 26 | 85% | 16 |
| dot12 | 2 | 12 | 20 | 80% | 12 |

中转监听本机 `:32859`，根据显式 `X-Orvix-Session-Id` 或消息指纹找到续轮的原节点。`/get_load` 不可用时从 Prometheus `/metrics` 读取 KV 与排队状态。超出中转的节点限制时，请求可溢出到本机 `:32860` 的 `power_of_two` 路由。默认池含 dot14 和 dot12。启用 dot13 前需压测该节点，再将其 URL 和容量同时写入中转与兜底路由配置。外部鉴权由独立网关提供，可将其 DeepSeek 上游设为中转器的 `/v1`。
