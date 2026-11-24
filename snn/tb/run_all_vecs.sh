#!/bin/bash
# run_all_vecs.sh [prefix=m3k3] [ZERO_SKIP=1] — run every golden vector dir of snn/golden/vectors/<prefix>/ (count mode)
cd "$(dirname "$0")"
P=${1:-m3k3}; ZS=${2:-1}
for d in ../golden/vectors/$P/*/; do ./run_vec.sh $d count $ZS 2>&1 | grep RESULT; done
