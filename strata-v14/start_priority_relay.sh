#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" -B "$ROOT/dot14_priority_relay.py" \
  --listen-host 127.0.0.1 \
  --listen-port 32859 \
  --router-url http://127.0.0.1:32860 \
  --dot14-url "${DOT14_URL:?Set DOT14_URL to the node daemon URL}" \
  --dot14-soft-cap 18 \
  --dot14-max-token-usage 0.85 \
  --dot14-max-waiting 16 \
  --dot14-weight 3 \
  --dot12-url "${DOT12_URL:?Set DOT12_URL to the node daemon URL}" \
  --dot12-soft-cap 12 \
  --dot12-max-token-usage 0.80 \
  --dot12-max-waiting 12 \
  --dot12-weight 2 \
  --dot13-url "" \
  --dot13-soft-cap 0 \
  --dispatch-mode equal \
  --sticky-burst 8
