# DeepSeek V4.1 Flash NVFP4 T1 基线

`overlay/sglang/` 是在 SGLang 上调整过的源码快照，包含 SM120 相关的 T1 优化。STRATA 运行时派生的原生接口合同基于 SGLang 的固定提交。此快照面向 DeepSeek 部署，移除了可选的 Mistral 3 多模态模型入口。此目录附上该版本的 Apache-2.0 许可证，并保留随源码分发的第三方许可证与声明。 [SGLang 固定源码提交](https://github.com/sgl-project/sglang/commit/858073e4f6d246bd6e3baae1d4e07919a578a04f)

`Dockerfile` 将源码覆盖到已经备好 CUDA 与模型依赖的 `deepseek-v41-flash-nvfp4:1017-sm120-dspark5` 镜像。拥有该基础镜像和可用的 8 张 SM120 GPU 时，可在本目录构建并启动 T1。

```bash
docker build -t deepseek-v41-flash-nvfp4:1017-sm120-t1 .
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017-NVFP4
docker compose config --quiet
docker compose up -d
python3 smoke.py --url http://127.0.0.1:30000
```

权重通过 `MODEL_PATH` 只读挂载，目录应包含 `config.json` 及其权重分片。容器使用 `TP8/EP8`，服务本机端口为 `30000`。STRATA 的 4 卡和 8 卡配置使用 [STRATA 节点脚本](../strata-v14/README.md)。
