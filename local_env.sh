#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_luffy_had_nounset=0
if [[ $- == *u* ]]; then
    _luffy_had_nounset=1
    set +u
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ROOT_DIR/.conda"

if [[ $_luffy_had_nounset -eq 1 ]]; then
    set -u
fi

export LUFFY_ROOT="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/luffy${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [[ -z "${CC:-}" ]] && command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
    export CC="$(command -v x86_64-conda-linux-gnu-gcc)"
fi

if [[ -z "${CXX:-}" ]] && command -v x86_64-conda-linux-gnu-c++ >/dev/null 2>&1; then
    export CXX="$(command -v x86_64-conda-linux-gnu-c++)"
fi

_nvidia_site_root="$ROOT_DIR/.conda/lib/python3.10/site-packages/nvidia"
if [[ -d "$_nvidia_site_root" ]]; then
    _nvidia_include_paths=()
    _nvidia_lib_paths=()
    while IFS= read -r _p; do
        _nvidia_include_paths+=("$_p")
    done < <(find "$_nvidia_site_root" -maxdepth 2 -type d -name include | sort)
    while IFS= read -r _p; do
        _nvidia_lib_paths+=("$_p")
    done < <(find "$_nvidia_site_root" -maxdepth 2 -type d -name lib | sort)

    if [[ ${#_nvidia_include_paths[@]} -gt 0 ]]; then
        _joined_includes="$(IFS=:; echo "${_nvidia_include_paths[*]}")"
        export CPATH="${_joined_includes}${CPATH:+:$CPATH}"
        export C_INCLUDE_PATH="${_joined_includes}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
        export CPLUS_INCLUDE_PATH="${_joined_includes}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        _nvcc_include_flags=()
        for _p in "${_nvidia_include_paths[@]}"; do
            _nvcc_include_flags+=("-I$_p")
        done
        _joined_nvcc_includes="${_nvcc_include_flags[*]}"
    fi

    if [[ ${#_nvidia_lib_paths[@]} -gt 0 ]]; then
        _joined_libs="$(IFS=:; echo "${_nvidia_lib_paths[*]}")"
        export LIBRARY_PATH="${_joined_libs}${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${_joined_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        _ldflags=()
        for _p in "${_nvidia_lib_paths[@]}"; do
            _ldflags+=("-L$_p")
        done
        _joined_ldflags="${_ldflags[*]}"
    fi
fi

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-allow-unsupported-compiler${_joined_nvcc_includes:+ $_joined_nvcc_includes}"
export LDFLAGS="${_joined_ldflags:+$_joined_ldflags }${LDFLAGS:+ $LDFLAGS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

unset _luffy_had_nounset _nvidia_site_root _nvidia_include_paths _nvidia_lib_paths _nvcc_include_flags _ldflags _joined_includes _joined_libs _joined_nvcc_includes _joined_ldflags _p
