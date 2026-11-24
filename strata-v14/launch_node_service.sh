#!/usr/bin/env bash
set -euo pipefail
umask 077

ROLE="${1:-}"
STAGE_ONLY=0
if [[ "${2:-}" == "--stage-only" ]]; then
    STAGE_ONLY=1
elif [[ -n "${2:-}" ]]; then
    echo "Usage: $0 <dot12|dot13|dot14> [--stage-only]" >&2
    exit 1
fi
if [[ -z "$ROLE" ]] || [[ ! "$ROLE" =~ ^(dot12|dot13|dot14)$ ]]; then
    echo "Usage: $0 <dot12|dot13|dot14> [--stage-only]" >&2
    exit 1
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STRATA_STATE_DIR:-$ROOT/.state/$ROLE}"
SOCK="$STATE_DIR/tmux.sock"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local model weights directory}"
BASE_SOURCE="${BASE_SOURCE:-$ROOT/../t1-base/overlay}"
PY="${PYTHON_BIN:-$(command -v python3)}"
CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
SEED_HOME="${SEED_HOME:-}"
SEED_CACHE="${SEED_CACHE:-}"
DAEMON_ALLOWED_CLIENTS="${DAEMON_ALLOWED_CLIENTS:-127.0.0.1,::1}"
SERVED_MODEL="deepseek-v41-flash-nvfp4"
ENGINE_PORT="32681"
DAEMON_PORT="32683"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-auto}"

CONTEXT_FLAG_LINE=""
case "$CONTEXT_LENGTH" in
    ""|auto|AUTO|Auto)
        CONTEXT_LENGTH="auto"
        ;;
    *)
        if [[ ! "$CONTEXT_LENGTH" =~ ^[0-9]+(Ki|Mi|Gi|Ti|k|M|G|T)?$ ]]; then
            echo "Invalid CONTEXT_LENGTH='$CONTEXT_LENGTH' (use 'auto' or positive integer / 1Mi)" >&2
            exit 1
        fi
        CONTEXT_FLAG_LINE="      --context-length $CONTEXT_LENGTH \\"$'\n'
        ;;
esac

FP8_CONFIGS_TAR=""
case "$ROLE" in
    dot12)
        TP=4
        EP=4
        VISIBLE_GPUS="0,1,2,3"
        MEM_FRAC="0.92"
        MAX_RUNNING="16"
        DECODE_GRAPH_BS="16"
        TORCH_BS="32"
        SWA_TAILS="64"
        BIND_DAEMON="${DAEMON_LISTEN_HOST:-0.0.0.0}"
        ;;
    dot13)
        TP=8
        EP=8
        VISIBLE_GPUS="0,1,2,3,4,5,6,7"
        MEM_FRAC="${MEM_FRAC:-0.74}"
        MAX_RUNNING="128"
        DECODE_GRAPH_BS="64"
        TORCH_BS="64"
        SWA_TAILS="512"
        BIND_DAEMON="${DAEMON_LISTEN_HOST:-127.0.0.1}"
        if [[ "${ORVIX_DISABLE_DOT13_FP8_CONFIGS:-0}" != "1" ]]; then
            FP8_CONFIGS_TAR="$ROOT/dot13-fp8-configs.tar.gz"
        fi
        ;;
    dot14)
        TP=8
        EP=8
        VISIBLE_GPUS="0,1,2,3,4,5,6,7"
        MEM_FRAC="0.74"
        MAX_RUNNING="128"
        DECODE_GRAPH_BS="64"
        TORCH_BS="128"
        SWA_TAILS="512"
        BIND_DAEMON="${DAEMON_LISTEN_HOST:-0.0.0.0}"
        ;;
esac

test -d "$MODEL_PATH"
test -f "$MODEL_PATH/config.json"
if [[ -n "$FP8_CONFIGS_TAR" && ! -f "$FP8_CONFIGS_TAR" ]]; then
    echo "Missing required FP8 GEMM config archive for $ROLE: $FP8_CONFIGS_TAR" >&2
    exit 1
fi
mkdir -p -m 0700 "$STATE_DIR"

# Stop prior managed sessions and wait for clean port/GPU exit unless staging only
if [[ "$STAGE_ONLY" -eq 0 ]]; then
    for s in ours-daemon ours-engine; do
        if tmux -S "$SOCK" has-session -t ="$s" 2>/dev/null; then
            pane_pid="$(tmux -S "$SOCK" list-panes -t ="$s" -F '#{pane_pid}' 2>/dev/null || true)"
            if [[ -n "$pane_pid" ]] && kill -0 "$pane_pid" 2>/dev/null; then
                pkill -TERM -P "$pane_pid" 2>/dev/null || true
                kill -TERM "$pane_pid" 2>/dev/null || true
            fi
            tmux -S "$SOCK" kill-session -t ="$s" 2>/dev/null || true
        fi
    done

    for ((i = 0; i < 60; i++)); do
        if ! ss -Hltn sport = :$ENGINE_PORT | grep -q . \
            && ! ss -Hltn sport = :$DAEMON_PORT | grep -q .; then
            break
        fi
        sleep 0.5
    done

    if ss -Hltn sport = :$ENGINE_PORT | grep -q .; then
        echo "Port $ENGINE_PORT already in use on $ROLE" >&2
        exit 1
    fi
    if ss -Hltn sport = :$DAEMON_PORT | grep -q .; then
        echo "Port $DAEMON_PORT already in use on $ROLE" >&2
        exit 1
    fi
fi

RUNTIME_DIR="$STATE_DIR/runtime-ours"
if [[ ! -f "$RUNTIME_DIR/orvix-trinode-runtime.json" ]]; then
    DERIVE_ARGS=(
        --base-source "$BASE_SOURCE"
        --target-runtime "$RUNTIME_DIR"
        --package-dir "$ROOT/package"
        --force
    )
    if [[ -n "$FP8_CONFIGS_TAR" ]]; then
        DERIVE_ARGS+=(--fp8-configs-tar "$FP8_CONFIGS_TAR")
    fi
    "$PY" -B "$ROOT/derive_node_runtime.py" "${DERIVE_ARGS[@]}"
elif [[ -n "$FP8_CONFIGS_TAR" ]]; then
    "$PY" -B "$ROOT/derive_node_runtime.py" \
        --target-runtime "$RUNTIME_DIR" \
        --fp8-configs-tar "$FP8_CONFIGS_TAR" \
        --install-fp8-configs-only
fi

mkdir -p -m 0700 "$STATE_DIR/engine"
if [[ ! -d "$STATE_DIR/engine/home/.cache" ]]; then
    rm -rf "$STATE_DIR/engine/home" "$STATE_DIR/engine/cache"
    if [[ -d "$SEED_HOME" ]]; then
        cp -a "$SEED_HOME" "$STATE_DIR/engine/home"
    else
        mkdir -p -m 0700 "$STATE_DIR/engine/home"
    fi
    if [[ -d "$SEED_CACHE" ]]; then
        cp -a "$SEED_CACHE" "$STATE_DIR/engine/cache"
    else
        mkdir -p -m 0700 "$STATE_DIR/engine/cache"
    fi
fi

PYTHONPATH="$ROOT/package/deploy/deepseek_v41_nvfp4" "$PY" -B - "$STATE_DIR/engine/home" <<'PY'
from pathlib import Path
import sys
from launch_private_unified import rebase_triton_cache_groups
n = rebase_triton_cache_groups(Path(sys.argv[1]))
print(f"Rebased {n} Triton cache group JSON files", flush=True)
PY

RUN_ENGINE="$STATE_DIR/engine/start_engine.sh"
cat > "$RUN_ENGINE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
umask 077
exec /usr/bin/env -i \\
    HOME="$STATE_DIR/engine/home" \\
    XDG_CACHE_HOME="$STATE_DIR/engine/cache" \\
    HF_HOME="$STATE_DIR/engine/cache/hf" \\
    PYTHONPATH="$RUNTIME_DIR" \\
    PYTHONNOUSERSITE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    LC_ALL=C.UTF-8 \\
    CUDA_HOME="$CUDA_DIR" \\
    CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \\
    PATH="$(dirname "$PY"):$CUDA_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \\
    HF_HUB_OFFLINE=1 \\
    TRANSFORMERS_OFFLINE=1 \\
    ORVIX_DSV41_UNIFIED=1 \\
    ORVIX_ENGINE_TRACE=0 \\
    ORVIX_DSV41_SKIP_ENGRAM_FADVISE=1 \\
    SGLANG_SM120_FLASHMLA_BACKEND=flashinfer \\
    SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1 \\
    SGLANG_PCIE_IPC_MAX_NUMEL=20971520 \\
    SGLANG_DSPARK_SHARDED_ACCEPT=1 \\
    SGLANG_DSPARK_FUSED_ACCEPT_TAIL=1 \\
    SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK=1 \\
    SGLANG_OPT_DSV41_DSPARK_POST_FOLD=1 \\
    SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 \\
    SGLANG_OPT_DSV41_BLOCKWISE_REJECTION=1 \\
    SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE=1 \\
    SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 \\
    SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 \\
    SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH=1 \\
    SGLANG_OPT_SCHEDULER_SHM_SYNC=1 \\
    "$PY" -m sglang.launch_server \\
      --model-path "$MODEL_PATH" \\
      --served-model-name "$SERVED_MODEL" \\
      --host 127.0.0.1 --port "$ENGINE_PORT" \\
      --trust-remote-code \\
      --tensor-parallel-size "$TP" --ep-size "$EP" \\
${CONTEXT_FLAG_LINE}      --kv-cache-dtype fp8_e4m3 \\
      --engram-config '{"placement":"cpu","replication":"none","lookup":"auto"}' \\
      --dsv4-attn-backend flashmla \\
      --moe-runner-backend flashinfer_cutlass \\
      --disable-shared-experts-fusion \\
      --chunked-prefill-size 2048 \\
      --mem-fraction-static "$MEM_FRAC" \\
      --max-running-requests "$MAX_RUNNING" \\
      --cuda-graph-backend-prefill disabled \\
      --cuda-graph-backend-decode full \\
      --cuda-graph-max-bs-decode "$DECODE_GRAPH_BS" \\
      --torch-compile-max-bs "$TORCH_BS" \\
      --reasoning-parser auto \\
      --tool-call-parser auto \\
      --default-chat-template-kwargs '{"thinking":true}' \\
      --enable-metrics --enable-cache-report \\
      --speculative-algorithm DSPARK \\
      --speculative-dspark-block-size 5 \\
      --speculative-moe-runner-backend flashinfer_mxfp4 \\
      --enable-decoder-swa-bounded-replay \\
      --enable-hierarchical-cache \\
      --hicache-write-policy write_back \\
      --hicache-ratio 0.45 \\
      --swa-prefix-tails "$SWA_TAILS" \\
      --min-free-slots-delay 0 \\
      --random-seed 0 \\
      --disable-custom-all-reduce \\
      --fp4-gemm-backend flashinfer_cudnn \\
      > "$STATE_DIR/engine/engine.log" 2>&1
EOF
chmod 0700 "$RUN_ENGINE"

RUN_DAEMON="$STATE_DIR/engine/start_daemon.sh"
cat > "$RUN_DAEMON" <<EOF
#!/usr/bin/env bash
set -euo pipefail
umask 077
exec "$PY" -B "$ROOT/ours_node_daemon.py" \\
    --node-label "$ROLE" \\
    --listen-host "$BIND_DAEMON" \\
    --listen-port "$DAEMON_PORT" \\
    --engine-url "http://127.0.0.1:$ENGINE_PORT" \\
    --engine-model "$SERVED_MODEL" \\
    --package-dir "$ROOT/package" \\
    --allowed-clients "$DAEMON_ALLOWED_CLIENTS" \\
    --state-file "$STATE_DIR/ours_activated.json" \\
    > "$STATE_DIR/engine/daemon.log" 2>&1
EOF
chmod 0700 "$RUN_DAEMON"

if [[ "$STAGE_ONLY" -eq 1 ]]; then
    echo "[$ROLE] Staged runtime and startup scripts only (--stage-only; CONTEXT_LENGTH=$CONTEXT_LENGTH; FP8_CONFIGS_TAR=${FP8_CONFIGS_TAR:-none}); no processes started."
    exit 0
fi

tmux -S "$SOCK" new-session -d -s ours-engine -c "$STATE_DIR/engine" "bash '$RUN_ENGINE'"
tmux -S "$SOCK" new-session -d -s ours-daemon -c "$ROOT" "bash '$RUN_DAEMON'"
echo "[$ROLE] Started tmux sessions ours-engine (:$ENGINE_PORT) and ours-daemon ($BIND_DAEMON:$DAEMON_PORT) in $STATE_DIR (CONTEXT_LENGTH=$CONTEXT_LENGTH; FP8_CONFIGS_TAR=${FP8_CONFIGS_TAR:-none})"
