FROM deepseek-v41-flash-nvfp4:1017-sm120-dspark5

ENV PYTHONPATH=/opt/deepseek/runtime/t1-python \
    PYTHONDONTWRITEBYTECODE=1
COPY overlay/sglang /opt/deepseek/runtime/t1-python/sglang
RUN mkdir -p /cache/home /cache/xdg /cache/hf
ENTRYPOINT ["/opt/deepseek/.venv/bin/python", "-m", "sglang.launch_server"]
