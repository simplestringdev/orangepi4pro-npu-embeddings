#!/usr/bin/env bash
# Build the NPU server on the Orange Pi 4 Pro. Needs gcc and the Allwinner NPU runtime that ships with the
# board's Debian image: /usr/include/vip_lite.h, libVIPhal.so, libNBGlinker.so (check: ls /dev/vipcore).
# usage: server/build.sh [output]     (default: ~/.local/bin/npu_embed_server)
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=${1:-$HOME/.local/bin/npu_embed_server}
[[ -c /dev/vipcore ]] || echo "warning: /dev/vipcore not found: the NPU driver is not loaded on this machine" >&2
[[ -f /usr/include/vip_lite.h ]] || { echo "missing /usr/include/vip_lite.h (NPU runtime headers)" >&2; exit 1; }
install -d "$(dirname "$out")"
gcc -O2 -Wall -Wextra -o "$out" "$here/npu_embed_server.c" -lVIPhal -lNBGlinker
echo "built $out"
