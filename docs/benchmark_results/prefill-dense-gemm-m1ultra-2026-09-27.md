# Dense f16 GEMM for large-M prefill projections (issue #1994)

At prefill an affine 4-bit projection runs MLX's `quantized_matmul`. With f16
scales and an f16 input, dequantizing the weight to f16 and running a dense
matmul is faster once the input has enough rows. `UnifiedLinear` now takes that
path at 1024 rows or more, by default on M1-generation Apple Silicon only
(`MLXCEL_PREFILL_DEQUANT_MIN_M` overrides the threshold on any backend; `0`
disables it).

- **Date:** 2026-09-25 to 2026-09-27
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Condition:** `mlxcel-bench-decode` (warmup 20 or 8, `--ignore-eos`,
  `--prompt-tokens` as listed), prefill time, ABBA order, indexers suspended
- **GPU guard:** an fp16 4096³ GEMM check between models (13.2 to 14.1
  TFLOPS throughout the runs reported here)

## Crossover (dense path on vs off, one binary, threshold forced low)

Prefill time change, mean of two runs per arm. The "on" arm printed a one-time
diagnostic line in every run, so the arms differ.

| Model | pp384 | pp512 | pp768 | pp1024 | pp2048 |
|---|---|---|---|---|---|
| Llama 3.1 8B | -1.6% | +4.3% | +7.6% | +10.0% | +13.5% |
| command-r7b | -0.8% | +5.5% | +7.6% | +7.7% | +10.2% |
| Phi-3 mini | -0.3% | +4.9% | +10.0% | +10.2% | +12.5% |
| Qwen2.5 7B | -11.0% | -3.1% | -1.5% | +1.5% | +3.9% |
| Gemma 2 2B | -0.1% | -1.3% | -1.6% | +0.6% | +1.0% |
| Mixtral 8x7B | noisy | +13 to +15% | 0.0% | +0.6% | +1.2% |

1024 rows is the lowest count with no measured regression, so it is the
default. The Mixtral pp512 gain reproduced on a second run and is not
explained: the path only touches its attention projections, and the gain
vanishes at 768 rows. Prefill peak memory grows 0.1 to 0.6 GB (Qwen2.5 7B at
pp512: 5.03 to 5.63 GB).

## Production default against main (`4dcf5187`)

No environment variable set; main binary against the branch binary, prefill
tok/s, two runs per arm.

| Model | Scales | pp1024 | pp2048 |
|---|---|---|---|
| Llama 3.1 8B | f16 | 815.2 → 898.1 (+10.2%) | 810.5 → 903.8 (+11.5%) |
| Qwen2.5 7B | f16 | 879.0 → 888.7 (+1.1%) | 870.2 → 889.9 (+2.3%) |
| Qwen3 1.7B | bf16 | 2656.7 → 2654.9 (-0.1%) | 2640.6 → 2638.4 (-0.1%) |

Decode is unchanged in every cell. Qwen3 is the bf16-scale control: it stays on
`quantized_matmul`.

## bf16 checkpoints and the exactness guard (issue #2001, 2026-09-28)

bf16-scale checkpoints also run bf16 activations, and with matching dtypes the
dense path reconstructs the same weight `quantized_matmul` does. An in-tree
sweep (f16 and bf16; M 1024 and 2048; K 2048, 2560, 4096; N 256 to 4096) found
identical bytes whenever the output has more than 512 tiles of 32 x 32, and
differences only at or below 512 tiles (M 1024 with N 512 or narrower, M 2048
with N 256). The path now requires matching f16 or bf16 dtypes and more than
512 output tiles, so wherever it runs it returns the same bytes as
`quantized_matmul`. The Gemma 4 E4B logit differences reported below came from
its narrow projections in that region, where bf16's larger ulp made one
rounding step visible.

Main-branch binary with the path forced off (`MLXCEL_PREFILL_DEQUANT_MIN_M=0`)
against on (`=256`), prefill time, mean of two runs per arm, GEMM check 12.9 to
14.0 TFLOPS:

| Model | Scales | pp512 | pp1024 | pp2048 |
|---|---|---|---|---|
| Qwen3 1.7B | bf16 | +12.8% | +15.1% | +18.0% |
| Qwen3-30B-A3B | bf16 | +2.3% | +5.5% | +4.4% |
| Gemma 4 12B | bf16 | -1.0% | +3.0% | +3.2% |
| Gemma 4 E4B | bf16 | -2.7% | +0.9% | +1.9% |
| Gemma 3 4B | bf16 | -1.1% | +0.5% | +0.8% |
| Llama 3.1 8B | f16 | +5.3% | +9.7% | +11.7% |
| Qwen2.5 7B | f16 | -4.2% | +0.6% | +3.0% |

No model regresses at 1024 rows or more, so the M1 gate and the 1024-row
threshold stand for both dtypes. Greedy output with the path off and on is
identical on all seven with a prompt above 1024 tokens.

## Correctness

- f16 scales: teacher-forced logit traces (1024 tokens × 2, top-8, six
  decimals) are byte-identical files with the path on and off for Llama 3.1 8B
  and Qwen2.5 7B, and perplexity matches (9.9874, 8.2434). Greedy output is
  identical to main on Llama 3.1 8B, Qwen2.5 7B, command-r7b, Phi-3 mini,
  Gemma 2 2B and Mixtral 8x7B with a prompt above 1024 tokens.
- The kernels differ only in how MLX tiles the output. A unit test checks
  identical bytes for f16 and bf16 above 512 output tiles and that narrower
  outputs and mixed dtypes are not eligible (see the #2001 section above).
- The first version ran bf16-scale projections regardless of width, and on
  Gemma 4 E4B that moved 2.7% of top-1 choices (max logit delta 4.87). The
  cause was the narrow projections, not bf16 itself.

## Excluded data

During this work the M1 Ultra repeatedly dropped to about 1.3 TFLOPS after
about 100 s of sustained GPU load, reproduced with a Python MLX GEMM alone and
cleared only by a restart. Every run taken in a degraded window was discarded;
the tables above come from windows where the GEMM check held above 13 TFLOPS.

## M5 Max (why the default excludes M5)

Mac17,7 Apple M5 Max 128 GB, branch `3eb871ea`, one binary with the path off
and on (`MLXCEL_PREFILL_DEQUANT_MIN_M=256`), order off on on off with a 20 s
cooldown before every run, indexers suspended, load average 2.2 to 1.4.
Prefill tok/s per run, median change, and MLX peak memory off / on.

| Model | pt | off, on, on, off | Change | Peak GB |
|---|---|---|---|---|
| Llama 3.1 8B | 512 | 3651, 3202, 3212, 3650 | -12.1% | 5.25 / 5.81 |
| Llama 3.1 8B | 1024 | 3757, 3616, 3618, 3758 | -3.7% | 5.62 / 6.22 |
| Llama 3.1 8B | 2048 | 3782, 3752, 3748, 3785 | -0.9% | 6.13 / 6.59 |
| Qwen2.5 7B | 512 | 3898, 3338, 3332, 3890 | -14.4% | 4.91 / 5.70 |
| Qwen2.5 7B | 1024 | 4027, 3918, 3921, 4041 | -2.8% | 5.29 / 5.91 |
| Qwen2.5 7B | 2048 | 4054, 4057, 4075, 4052 | +0.3% | 5.54 / 6.20 |
| Phi-3 mini | 512 | 7003, 6036, 6042, 7006 | -13.8% | 3.07 / 3.63 |
| Phi-3 mini | 1024 | 6913, 6745, 6740, 6912 | -2.5% | 3.76 / 3.74 |
| Phi-3 mini | 2048 | 7084, 7069, 7067, 7086 | -0.2% | 4.45 / 4.58 |

M5's `quantized_matmul` (with the Neural Accelerator) is strong enough that
the dense path only reaches break-even around 2048 rows. Greedy output on
Llama 3.1 8B with a 1609-token prompt is identical with the path on and off.

## Not measured

M2, M3 and M4 (the default is off there until measured), CUDA and ROCm, and
the server's chunked prefill, whose default 512-token chunks stay below the
threshold.
