#!/usr/bin/env bash
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ROCm smoke: build, link, and generate on a real AMD device (issue #1811).
#
# This is the body of the `ROCm build and generate` job in
# `.github/workflows/ci.yml`, factored out so the job and `make verify-rocm`
# run the same thing. The job is parked behind `vars.ROCM_CI_ENABLED` until a
# gfx1151 runner exists; until then this script is the only thing that runs it,
# which is exactly why it must not drift from the job.
#
# Why this exists next to `verify-test-rocm`, which already runs the suite:
# three failure classes get past a test run.
#
#   - The binary links but cannot start, because the ROCm rpath is missing.
#     Issue #1802 shipped that way on two crates and their test binaries failed
#     with `libamdhip64.so.7: cannot open shared object file`. LD_LIBRARY_PATH
#     is unset below so the rpath has to carry it.
#   - A kernel compiles and links but launches nothing. The missing primitives
#     in issue #1825 had that shape.
#   - The run silently falls back to the CPU, which still "succeeds" at
#     generating text.
#
# Environment:
#   MLXCEL_ROCM_SMOKE_MODEL  checkpoint to generate with; default is the small
#                            Qwen3 used by CI.
#   ROCM_JOBS                cargo parallelism flag, e.g. `-j 6`; see Makefile.
set -euo pipefail

MODEL="${MLXCEL_ROCM_SMOKE_MODEL:-models/qwen3-0.6b-4bit}"
JOBS="${ROCM_JOBS:--j 6}"
PROMPT="Write one sentence about the ocean."
OUT="$(mktemp -t rocm-smoke-out.XXXXXX)"
ERR="$(mktemp -t rocm-smoke-err.XXXXXX)"
trap 'rm -f "$OUT" "$ERR"' EXIT

if [ ! -d "$MODEL" ]; then
    echo "checkpoint not found: $MODEL" >&2
    echo "set MLXCEL_ROCM_SMOKE_MODEL to a local checkpoint, or fetch the CI fixture" >&2
    exit 1
fi

# Memory preflight. An OOM kill partway through `cargo build` surfaces as an
# opaque crash, and on this host the usual cause is not the build at all: the
# validation machine shares 30 GiB with the GPU services it runs, which have
# been measured holding 25.5 GiB between them. Saying so before the build
# starts turns an hour of confusion into one line. Advisory, not a gate: the
# caller may know better, and a smaller `ROCM_JOBS` may well fit.
if [ -r /proc/meminfo ]; then
    avail_mb="$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)"
    echo "[rocm-smoke] available memory: ${avail_mb} MiB (cargo parallelism: ${JOBS:-default})"
    if [ "${avail_mb:-0}" -lt 4096 ]; then
        echo "[rocm-smoke] warning: under 4 GiB available; a build at this parallelism may be OOM-killed." >&2
        echo "[rocm-smoke] largest resident processes:" >&2
        ps -eo rss,comm --no-headers \
            | awk '{a[$2]+=$1} END {for (c in a) if (a[c] > 512000) printf "  %-24s %6.1f GiB\n", c, a[c]/1024/1024}' \
            | sort -k2 -rn >&2 || true
        echo "[rocm-smoke] free memory or lower ROCM_JOBS (for example ROCM_JOBS='-j 2')." >&2
    fi
fi

echo "[rocm-smoke] building and linking with --features rocm"
# shellcheck disable=SC2086
cargo build --release $JOBS --features rocm

echo "[rocm-smoke] generating on the GPU"
# stdout and stderr are kept apart on purpose. Merging them interleaves the
# `[mlx-rocm]` device lines into the generated text, and a word count over the
# combined log then passes on a run that generated nothing, because the
# diagnostics alone carry over a hundred words.
#
# `--show-reasoning` because a reasoning-capable checkpoint sends every token to
# the hidden channel on a prompt this short, leaving the content channel
# legitimately empty on a healthy run.
(
    unset LD_LIBRARY_PATH
    MLXCEL_DEBUG_KERNEL_BACKEND=1 ./target/release/mlxcel generate \
        -m "$MODEL" -p "$PROMPT" -n 32 --temp 0 --show-reasoning \
        > "$OUT" 2> "$ERR"
)
cat "$OUT"
cat "$ERR"

fail() { echo "[rocm-smoke] $1" >&2; echo "--- stdout ---" >&2; cat "$OUT" >&2; echo "--- stderr ---" >&2; cat "$ERR" >&2; exit 1; }

# The active device, which is the thing a silent CPU fall back would change.
# Note that the kernel-backend line below is NOT this: `gpu_kernel_backend()`
# reports which backend MLX resolved, not which device the run used, and the
# two deliberately do not track each other (issue #1805, and the streams test
# `gpu_backend_available_does_not_track_the_default_device` pins it). A run
# with MLXCEL_DEVICE=cpu still prints `custom kernel backend: rocm`, so
# asserting only that would accept a CPU run.
grep -q '^Runtime device: GPU' "$OUT" \
    || fail "expected the run to be on the GPU; a CPU fall back still generates text"
grep -q 'custom kernel backend: rocm' "$ERR" \
    || fail "expected the resolved kernel backend to be rocm"
grep -q '^HIP architecture gfx' "$OUT" \
    || fail "expected the startup diagnostics to name a HIP architecture"
# An AMD device reporting a CUDA compute capability is the issue #1805 defect.
# Assert it stays gone here rather than trusting the unit test alone.
if grep -q 'CUDA compute capability' "$OUT"; then
    fail "an AMD device must not report a CUDA compute capability"
fi

summary="$(grep -oE '^\[Generated [0-9]+ tokens' "$OUT" | head -1 || true)"
[ -n "$summary" ] || fail "no generation summary line; the decode loop did not finish"
tokens="$(printf '%s' "$summary" | grep -oE '[0-9]+')"
[ "$tokens" -ge 16 ] || fail "generated only $tokens tokens, expected at least 16"

# Generated text only: between the banner and the summary, minus the echoed
# prompt and any bracketed notice.
text="$(sed -n '/^Generating\.\.\./,/^\[Generated /p' "$OUT" | sed '1,2d;$d' | grep -v '^\[' || true)"
words="$(printf '%s' "$text" | grep -oE '[A-Za-z]{2,}' | wc -l)"
[ "$words" -ge 8 ] || fail "generated text held $words word-like tokens, expected at least 8"

echo "[rocm-smoke] OK: $tokens tokens, $words word-like tokens of generated text"
