# Agent-Infra

DeepSeek V4.1 Flash NVFP4 的 T1 推理基线、STRATA v14 调度和跨机负载均衡。此仓库记录三个节点的部署配置、可运行的评测代码、公开合成测试集和历史实机指标。

| 内容 | 入口 |
| --- | --- |
| .12、.13、.14 的 Linux 部署与运行时派生 | [STRATA v14 节点部署](strata-v14/README.md) |
| 前缀稳定、IntervalCache、PrefillGovernor | [节点守护进程](strata-v14/ours_node_daemon.py) · [调度模块](strata-v14/package/deploy/deepseek_v41_nvfp4/) |
| .14:.12 = 3:2、会话亲和、KV 压力探测 | [跨机负载均衡](strata-v14/README.md#跨机负载均衡) |
| T1 SGLang 源码、Dockerfile、八卡 Compose | [T1 基线](t1-base/README.md) |
| 48 会话、690 轮与七组主评测队列 | [评测脚本与合成测试集](evaluation/README.md) |
| V1/V2 全表、持续混流、单节点和 FP8 配对测试 | [历史结果](results/README.md) |

## 本地检查

```bash
python3.12 -m pip install -r requirements.txt
(cd strata-v14 && python3.12 -B -m unittest test_dot14_priority_relay test_system_prefix_patcher test_derive_node_runtime)
python3.12 -B evaluation/run_20261120_lb24_eval.py --bundle-dir evaluation/synthetic-lb24 --output-dir /tmp/strata-lb24-check --dry-run
```

真实请求轨迹与生产配置保存在原工作区。GitHub 提供独立生成的公开样本，历史实机数据以汇总表发布。
