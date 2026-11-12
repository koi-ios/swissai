#!/usr/bin/env bash
# TP=2 checkpoint, keeping the inter-rank TCPStore connection ALIVE across the snapshot
# instead of --tcp-close force-closing it. Needs a criu binary carrying CAP_NET_ADMIN
# (default /usr/local/sbin/criu-netadmin -- a copy of the built criu with
# `setcap cap_net_admin,cap_sys_ptrace,cap_checkpoint_restore=ep`), so criu can snapshot
# the established connection via TCP_REPAIR. See results/tp2-exp/results.md section 3b for
# why --tcp-close breaks the TP=2 restore.
#
# Usage: scripts/tp2-exp/checkpoint-keeptcp.sh [--thin] [run-dir]
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CRIU="${CRIU:-/usr/local/sbin/criu-netadmin}"
export CRIU_TCP_ARGS="${CRIU_TCP_ARGS:---tcp-established --network-lock skip}"

exec "$DIR/checkpoint.sh" "$@"
