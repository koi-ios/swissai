#!/usr/bin/env bash
# Restore a TP=2 snapshot taken by checkpoint-keeptcp.sh: the TCPStore connection was
# snapshotted via TCP_REPAIR (not --tcp-close'd), so restore it intact with
# --tcp-established and the same CAP_NET_ADMIN criu binary.
#
# Usage: scripts/tp2-exp/restore-keeptcp.sh [--cold] <run-dir>
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CRIU="${CRIU:-/usr/local/sbin/criu-netadmin}"
export RESTORE_TCP_ARGS="${RESTORE_TCP_ARGS:---tcp-established}"

exec "$DIR/restore.sh" "$@"
