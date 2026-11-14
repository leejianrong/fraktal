#!/usr/bin/env bash
# Stand-in for "a Python with bpy" on WSL: runs the script under the Windows Blender (GPU: OptiX), translating WSL paths.
# Use: FRAKTAL_BPY_PYTHON=$PWD/scripts/wsl/winblender.sh fraktal render3d ... --device gpu
BL="${FRAKTAL_WIN_BLENDER:-/mnt/c/Program Files/Blender Foundation/Blender 5.0/blender.exe}"
script=$1; shift
conv=()
for a in "$@"; do
  if [[ $a == /* && ( -e $a || -e $(dirname "$a") ) ]]; then conv+=("$(wslpath -w "$a")"); else conv+=("$a"); fi
done
exec "$BL" -b --factory-startup --python-exit-code 1 --python "$(wslpath -w "$script")" -- "${conv[@]}"
