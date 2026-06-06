#!/usr/bin/env bash
set -euo pipefail

cd /mnt/public/gaojiahuan/liushengbang/openpi_tactile

WAIT_LOG="wait_then_train_twist_bottle_cap_ki_advantage.log"
TRAIN_LOG="train_twist_bottle_cap_origin_synth0523_ki_advantage.log"
BEAT_PATTERN="scripts/train.py.*pi05_beat_xylophone_origin_synth0523_force_stride5_posecont_dupfront_192x256_ki_advantage"
POLL_SECONDS=1800

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${WAIT_LOG}"
}

log "waiting for beat_xylophone training to finish"
while pgrep -f "${BEAT_PATTERN}" >/dev/null; do
  pids="$(pgrep -f "${BEAT_PATTERN}" | tr '\n' ' ')"
  log "beat_xylophone still running: ${pids}; sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "beat_xylophone process is gone; checking GPU memory"
while true; do
  busy_gpus="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ if ($1 > 1000) busy += 1 } END { print busy + 0 }')"
  if [[ "${busy_gpus}" -eq 0 ]]; then
    break
  fi
  log "GPUs still busy: ${busy_gpus} GPU(s) above 1000 MiB; sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "starting twist_bottle_cap ki+advantage 8-GPU training"
./train_twist_bottle_cap_origin_synth0523_force_stride5_posecont_dupfront_ki_advantage.sh 2>&1 | tee -a "${TRAIN_LOG}"
