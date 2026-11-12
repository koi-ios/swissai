# Source this to use the folder-local Go toolchain and caches.
OT73_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export GOROOT="$OT73_ROOT/toolchain/go"
export GOPATH="$OT73_ROOT/toolchain/gopath"
export GOCACHE="$OT73_ROOT/toolchain/gocache"
export GOMODCACHE="$OT73_ROOT/toolchain/gopath/pkg/mod"
export GOTOOLCHAIN=local
export PATH="$GOROOT/bin:$GOPATH/bin:$PATH"
