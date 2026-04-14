#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

########################################
# 可按需修改的配置
########################################
CONDA_ROOT="/media/jiayueru/Ckpt/miniconda3"
CONDA_BIN="$CONDA_ROOT/bin/conda"
UV_BIN="$CONDA_ROOT/bin/uv"
ENV_NAME="openpi311"

CLASH_DIR="/media/jiayueru/Ckpt/clash_for_linux_backup"
OPENPI_DIR="/media/jiayueru/Ckpt/WRL/openpi_tactile"
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force"

TASK_NAME="pi05_wipe_whiteboard_force"
EXP_NAME="run_20260414"
OVERWRITE_FLAG="--overwrite"

WANDB_BASE_URL_DEFAULT="https://api.bandw.top"
WANDB_API_KEY_DEFAULT="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
WANDB_ENTITY_DEFAULT="liushb9-peking-university"

log() { echo ">>> $*"; }

ensure_conda_env() {
  log "阶段1: ensure_conda_env"
  "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  "$CONDA_BIN" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

  if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "环境 $ENV_NAME 已存在"
  else
    "$CONDA_BIN" create -n "$ENV_NAME" python=3.11 -c conda-forge -y
  fi
}

ensure_system_deps() {
  log "阶段2: ensure_system_deps（缺失才安装，apt 不走代理）"

  local missing=()
  dpkg -s cmake >/dev/null 2>&1 || missing+=("cmake")
  dpkg -s build-essential >/dev/null 2>&1 || missing+=("build-essential")

  if [ "${#missing[@]}" -eq 0 ]; then
    log "系统编译依赖已满足，跳过 apt"
    return
  fi

  local _http_proxy="${http_proxy-}" _https_proxy="${https_proxy-}" _HTTP_PROXY="${HTTP_PROXY-}" _HTTPS_PROXY="${HTTPS_PROXY-}" _ALL_PROXY="${ALL_PROXY-}"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

  apt-get update
  apt-get install -y --fix-missing "${missing[@]}"

  [ -n "${_http_proxy}" ] && export http_proxy="${_http_proxy}"
  [ -n "${_https_proxy}" ] && export https_proxy="${_https_proxy}"
  [ -n "${_HTTP_PROXY}" ] && export HTTP_PROXY="${_HTTP_PROXY}"
  [ -n "${_HTTPS_PROXY}" ] && export HTTPS_PROXY="${_HTTPS_PROXY}"
  [ -n "${_ALL_PROXY}" ] && export ALL_PROXY="${_ALL_PROXY}"

  log "系统编译依赖安装完成"
}

ensure_conda_ffmpeg7() {
  log "阶段3: ensure_conda_ffmpeg7（自动装到 ${ENV_NAME}）"

  "$CONDA_BIN" install -n "$ENV_NAME" -c conda-forge -y "ffmpeg=7" "pkg-config"

  export FFMPEG_ROOT="$CONDA_ROOT/envs/$ENV_NAME"
  export PATH="$FFMPEG_ROOT/bin:$PATH"
  export PKG_CONFIG_PATH="$FFMPEG_ROOT/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$FFMPEG_ROOT/lib:${LIBRARY_PATH:-}"

  command -v pkg-config >/dev/null 2>&1 || { echo "ERROR: pkg-config 不存在"; exit 1; }
  pkg-config --exists libavformat || { echo "ERROR: pkg-config 找不到 libavformat"; exit 1; }

  local avf_ver
  avf_ver="$(pkg-config --modversion libavformat || true)"
  log "FFmpeg root: $FFMPEG_ROOT"
  log "libavformat version: $avf_ver"

  case "$avf_ver" in
    61.*) ;;
    *) echo "ERROR: 需要 ffmpeg7 (libavformat 61.x)，当前: $avf_ver"; exit 1 ;;
  esac
}

start_clash_and_proxy() {
  log "阶段4: start_clash_and_proxy"
  cd "$CLASH_DIR"
  chmod +x start_us.sh

  if curl -s --max-time 2 http://127.0.0.1:9090 >/dev/null 2>&1; then
    log "Clash 已在运行"
  else
    nohup ./start_us.sh us >/tmp/clash.log 2>&1 &
    echo -n ">>> 等待 Clash 就绪"
    for i in $(seq 1 40); do
      if curl -s --max-time 2 http://127.0.0.1:9090 >/dev/null 2>&1; then
        echo " ... 就绪"
        break
      fi
      echo -n "."
      sleep 1
    done
  fi

  export http_proxy="http://127.0.0.1:7890"
  export https_proxy="http://127.0.0.1:7890"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
}

ensure_python_and_uv() {
  log "阶段5: ensure_python_and_uv"
  [ -x "$UV_BIN" ] || { echo "ERROR: 未找到 uv: $UV_BIN"; exit 1; }

  local py="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
  [ -x "$py" ] || { echo "ERROR: 未找到环境 Python: $py"; exit 1; }

  export UV_PYTHON="$py"
  export UV_PYTHON_DOWNLOADS=never
  log "使用 Python: $UV_PYTHON"
}

ensure_project_venv_and_av() {
  log "阶段6: ensure_project_venv_and_av"
  cd "$OPENPI_DIR"

  "$UV_BIN" sync -v

  "$UV_BIN" pip install -U pip setuptools wheel "cython>=3.0"

  if "$UV_BIN" run python -c "import av; print(av.__version__)" >/dev/null 2>&1; then
    log "av 已可用"
  else
    log "av 不可用，自动编译安装 av==14.4.0"
    "$UV_BIN" pip install --reinstall --no-cache-dir "av==14.4.0"
  fi

  "$UV_BIN" run python -c "import av,sys; print('av ok:', av.__version__, '| py:', sys.version.split()[0])"
}

calc_workers() {
  log "阶段7: calc_workers"
  CPU_CORES="$(nproc)"
  NUM_WORKERS=$(( CPU_CORES / 2 ))
  [ "$NUM_WORKERS" -lt 1 ] && NUM_WORKERS=1
  [ "$NUM_WORKERS" -gt 16 ] && NUM_WORKERS=16
  [ "$NUM_WORKERS" -gt "$CPU_CORES" ] && NUM_WORKERS="$CPU_CORES"

  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  log "CPU_CORES=$CPU_CORES, NUM_WORKERS=$NUM_WORKERS"
}

run_train() {
  log "阶段8: run_train"
  cd "$OPENPI_DIR"
  export HF_LEROBOT_HOME="$HF_LEROBOT_HOME"

  WANDB_BASE_URL="${WANDB_BASE_URL:-$WANDB_BASE_URL_DEFAULT}" \
  WANDB_API_KEY="${WANDB_API_KEY:-$WANDB_API_KEY_DEFAULT}" \
  WANDB_ENTITY="${WANDB_ENTITY:-$WANDB_ENTITY_DEFAULT}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$UV_BIN" run scripts/train.py \
    "$TASK_NAME" \
    --exp-name "$EXP_NAME" \
    $OVERWRITE_FLAG \
    --num-workers "$NUM_WORKERS"
}

ensure_conda_env
ensure_system_deps
ensure_conda_ffmpeg7
start_clash_and_proxy
ensure_python_and_uv
ensure_project_venv_and_av
calc_workers
run_train
