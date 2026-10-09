#!/usr/bin/env bash
# Configure + build into ./build on Perlmutter.
set -euo pipefail
cd "$(dirname "$0")/.."
source perlmutter/env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" "$@"
cmake --build build -j 16
