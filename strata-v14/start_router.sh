#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROUTER_LOG_DIR:-$ROOT/.state/router/logs}"
: "${DOT14_URL:?Set DOT14_URL to the node daemon URL}" "${DOT12_URL:?Set DOT12_URL to the node daemon URL}"
mkdir -p -m 0700 "$LOG_DIR"
exec "${ROUTER_PYTHON_BIN:-python3}" -m sglang_router.launch_router \
  --host 127.0.0.1 \
  --port 32860 \
  --worker-urls "$DOT14_URL" "$DOT12_URL" \
  --policy power_of_two \
  --history-backend none \
  --prometheus-host 127.0.0.1 \
  --prometheus-port 32861 \
  --request-timeout-secs 3600 \
  --worker-startup-timeout-secs 120 \
  --health-check-interval-secs 15 \
  --log-dir "$LOG_DIR"
