#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

cd /mnt/public/gaojiahuan/liushengbang/openpi_tactile

./prepare_beat_xylophone_origin_synth0523_force_stride5_posecont_dupfront_ki_advantage.sh
./train_beat_xylophone_origin_synth0523_force_stride5_posecont_dupfront_ki_advantage.sh
