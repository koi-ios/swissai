#!/bin/bash
# Usage: run_matrix.sh <base|fixed> <scenario: pub|relay> <history> <timeout> <delays...>
# Appends one CSV row per trial to ../results/e2e_<scenario>_<binary>.csv.
set -u
cd "$(dirname "$0")"
bin=$1; scenario=$2; history=$3; timeout=$4; shift 4
extra=""; [ "$scenario" = pub ] && extra="--worker-public-addr"
for d in "$@"; do
  python3 -I restart_repro.py --binary bin/otela-$bin --label $bin $extra --history $history \
    --delay $d --timeout $timeout --runs 3 --workdir runs/$scenario --csv ../results/e2e_${scenario}_${bin}.csv
done
