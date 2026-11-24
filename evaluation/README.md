# STRATA 与负载均衡评测

`synthetic-lb24/` 包含 48 个虚构库存会话和 690 个多轮请求。会话覆盖四档相对输入长度、系统上下文变更、工具调用历史和间隔。七组主评测队列包含两个互斥的 24 会话并发集、独享集、深轮集及三类混合集。清单中的 token 字段沿用回放接口，记录的是估计值。实测 token 以推理服务回执为准。

```bash
python3.12 -B evaluation/run_20261120_lb24_eval.py \
  --bundle-dir evaluation/synthetic-lb24 --output-dir /tmp/lb24-v1-check --dry-run
python3.12 -B evaluation/run_v2_equal_lb_eval.py \
  --bundle-dir evaluation/synthetic-lb24 --output-dir /tmp/lb24-v2-check --dry-run
```

以上两条命令在仓库根目录执行，只解析样本与评测参数。合成内容可以重新生成。调高 `--context-scale` 会增加模拟输入长度，并增加输出文件大小和评测内存开销。

```bash
python3.12 -B evaluation/build_synthetic_bundle.py \
  --output-dir /tmp/lb24-generated --context-scale 1
```

- `run_20261120_lb24_eval.py` 支持并发梯度、缓存冷热对照、乱序到达、流量波形、长期混流和深轮测试。
- `run_v2_equal_lb_eval.py` 支持关键点复测与读取 V1 回执做配对比较。路由依赖服务端设置。复现历史 V2 时把两节点权重设为 `1:1`。
- `bench_trinode_pd.py` 测量单节点冷/热 Prefill 与独享/并发 Decode，需额外安装 `tokenizers` 并通过 `--tokenizer-path` 给出模型 tokenizer，通过 `--dot12-url`、`--dot13-url` 或 `--dot14-url` 选择测试节点。

实机运行时通过 `--base-url` 和节点 URL 参数配置目标，用 `OPENAI_API_KEY` 或 `--api-key-file` 传递鉴权，并为完整跑设置适当的 `--deadline-cst`。先用 V1 的 `--smoke` 验证请求链路，再选择完整测试。历史真实流量的指标见 [结果目录](../results/)。
