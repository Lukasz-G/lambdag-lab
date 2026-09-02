#!/usr/bin/env bash
# TM hyperparameter sweep on the SYMMETRIC-length protocol (same data config as tm_sym):
# stage 1 capacity (clauses,T), stage 2 specificity (S). Runs sequentially; one log+scores per config.
#   bash phase4/hp_sweep.sh
set -e
cd "$(dirname "$0")/.."

export P4_ATOMS=pretrained
export P4_KLENS=150,300,600,1200,3000
export P4_QLENS=150,300,600,1200
export P4_KSTRIDE=2.0 P4_QSTRIDE=2.0
export P4_NORM=1
export P4_KBITS=200,400,800,2000
export P4_EVALLENS=0,1200,600,300,150
export P4_EVALSYM=1
export P4_SIZES=18000 P4_EPOCHS=60 P4_ENSEMBLE=3

run () {  # run <tag> <clauses> <T> <S>
  echo "=== $1 (CLAUSES=$2 T=$3 S=$4) ==="
  P4_CLAUSES=$2 P4_T=$3 P4_S=$4 P4_OUT=tm_hp_$1.jsonl \
    julia -t auto phase4/tm_scale.jl > phase4/tm_hp_$1.log 2>&1
  tail -6 phase4/tm_hp_$1.log
}

run c1024 1024 256 4096
run c2048 2048 512 4096
run s2048  512 128 2048
run s8192  512 128 8192
echo "sweep done"
