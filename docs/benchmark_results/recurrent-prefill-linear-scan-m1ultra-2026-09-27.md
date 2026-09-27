# Linear-time prefill scan for Jamba and RecurrentGemma (issue #1999)

Two recurrent families built their prefill output by folding per-step results
with one `concatenate` per step, which re-copies the growing tensor on every
step and makes prefill quadratic in the prompt length:

- **Jamba** (`JambaMambaMixer::ssm_step`) materialized every per-step SSM
  state `[B, 1, intermediate, d_state]`, folded them one step at a time, and
  only then computed `y = state @ C` over the whole `[B, L, 5120, 16]` tensor.
  It now computes `y_t = state_t @ C_t` inside the loop (the same product the
  single-token branch computes) and stacks the `y` rows once.
- **RecurrentGemma** (`rnn_scan`, sequence mode) folded its `[B, 1, width]`
  outputs the same way. It now stacks them once.

The sibling Mamba1, RWKV-7 and gated-delta scans already stacked once and are
unchanged.

- **Date:** 2026-09-27
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428), display attached
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** main `61f7851d` against the change
- **Condition:** `mlxcel-bench-decode`, `--ignore-eos`, `--prompt-tokens` as
  listed, fp16 GEMM check between prompt lengths (13.4 to 14.2 TFLOPS)

## Jamba reasoning 3B 4-bit (prefill tok/s)

Warmup 20, tg128. main at 1536 and 2048 took minutes per run, so those rows use
one earlier main run each.

| Prompt | main | change | Speedup |
|---|---|---|---|
| 256 | 351.2, 352.7 | 833.6, 803.7 | 2.3x |
| 512 | 225.3, 224.3 | 859.8, 851.7 | 3.8x |
| 1024 | 131.6, 131.2 | 838.8, 839.9 | 6.4x |
| 1536 | 8.2 | not run | |
| 2048 | 6.3 | 848.9, 852.3 | about 135x |

Main slows about 3.2 to 3.6x per doubling up to 1024 tokens and then drops to
single-digit tok/s; the change holds about 850 tok/s at every length. Decode is
unchanged (129.7 to 132.7 tok/s in both arms), and peak memory falls (4.41 to
3.24 GB at 1024 tokens).

## RecurrentGemma 9B bf16 (`alpindale/recurrentgemma-9b`, prefill tok/s)

Warmup 8, tg32, order main change change main.

| Prompt | main | change | Change |
|---|---|---|---|
| 512 | 533.6, 530.3 | 606.3, 599.7 | +13.3% |
| 1024 | 509.8, 510.6 | 624.9, 627.2 | +22.7% |
| 2048 | 440.8, 447.0 | 641.6, 639.2 | +44.3% |
| 4096 | 408.2, 407.1 | 640.3, 642.0 | +57.3% |

The per-step output is 16x narrower than Jamba's state, so the quadratic term
is smaller, but it still grows with the prompt: main loses 24% of its
throughput from 512 to 4096 tokens and the change loses none. Decode is
unchanged (32.5 to 32.9 tok/s).

## Correctness

- Greedy output identical to main: Jamba 3B on a short, a 512-token and an
  about 1100-token prompt (100 tokens); RecurrentGemma 9B on a short and a
  512-token prompt (60 tokens).
- New test: a one-layer Mamba Jamba fixture checks that every prefill row
  equals the same position decoded one token at a time, fresh and continuing
  from an existing conv and SSM state. Dropping the previous-state term from
  the prefill loop fails it (max |diff| 0.86).

## Not measured

CUDA and ROCm; RecurrentGemma 2B (gated on Hugging Face); Jamba at 1536 tokens
with the change.
