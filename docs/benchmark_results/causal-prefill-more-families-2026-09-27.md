# Maskless causal prefill for Gemma 3/4, Jamba and Nemotron-H (issue #1991)

Cohere2 (#1957) stopped building a prefill mask array when every live key is
inside the attention window and calls MLX's maskless causal SDPA instead, as
mlx-lm passes `"causal"` for `N <= window`. This extends that to:

- Gemma 3: global layers always, sliding layers while the live keys fit the
  window.
- Gemma 4 text prefill: the same rule, when there are no vision bidirectional
  blocks and no per-row valid ends. `attend` passes window 0 when the keys fit.
- Jamba and Nemotron-H: the main forward passes no mask.

`layers::attention` with no mask runs non-causal SDPA, so each attention layer
now calls `causal_attention` explicitly for a multi-token input with no mask.
Pipeline-stage paths and Gemma 2 (its softcap needs a mask array) are
unchanged.

## M1 Ultra (main `4dcf5187` against the branch)

`mlxcel-bench-decode` with `bench_decode.sh` arguments, ABBA, an fp16 GEMM
check between models (14.07 to 14.13 TFLOPS), load 6.0 to 5.2. Prefill change:

| Model | pp512 | pp1000 | pp2048 |
|---|---|---|---|
| Gemma 3 4B | -0.01% | +0.03% | +0.02% |
| Gemma 4 E4B | +0.12% | | -0.03% |
| Gemma 4 12B | -0.12% | 0.00% | 0.00% |
| Jamba reasoning 3B | +0.12% | | not comparable (both arms about 6 tok/s) |
| Nemotron 3 Nano 30B | -0.22% | | +0.29% |

A temporary diagnostic line confirmed the maskless path ran at pp1000 on both
Gemma models, including the sliding layers. On M1-M4, MLX sends head_dim 256
prefill to the unfused SDPA path (`use_fallback` in
`mlx/backend/metal/scaled_dot_product_attention.cpp`), which builds the causal
mask internally, so the mode switch saves nothing there.

## M5 Max (main `4dcf5187` against the branch)

Mac17,7, separate worktrees, a 20 s cooldown before every run, ABBA medians:

| Model | pp512 | pp1024 | pp2048 |
|---|---|---|---|
| Gemma 3 4B | +0.1% | +3.2% | +0.8% |
| Gemma 4 E4B | +0.1% | +0.2% | +0.2% |
| Gemma 4 12B | +0.1% | +1.4% | +0.1% |

With the Neural Accelerator, head_dim 256 prefill of at least 1024 queries
stays on the fused NAX kernel, which skips fully masked blocks in causal mode.
The gain appears only when that holds and every layer is inside its window:
Gemma 3 4B and Gemma 4 12B have 1024-token windows, so it peaks at 1024; at
2048 the sliding layers exceed the window and keep their mask, and Gemma 4
E4B's 512-token window keeps its sliding mask even at 1024. Decode is
unchanged. A first sweep without cooldowns drifted thermally (12B decode 44.7
to 31.4 tok/s) and was discarded.

## Correctness

- Greedy output identical to main on M1 Ultra for Gemma 3 4B, Gemma 4 E4B and
  12B (short, 512-token and image prompts), Jamba 3B and Nemotron 3 Nano
  (short and 512-token prompts), and on M5 Max for the three Gemma models.
- New tests compare every prefill row with the same position decoded one token
  at a time: Gemma 3 for a fresh prefill, a continued prefill with an offset,
  and prefills past the window; Jamba with varied embeddings, fresh and
  continued. Replacing the causal call with the unmasked one fails both
  (Jamba max |diff| 3.06, Gemma 3 logit diff 0.011).

## Not measured

CUDA and ROCm; Nemotron-H has no dedicated unit test (covered by the greedy
comparison); Jamba 3B's roughly 6 tok/s prefill at 2048 tokens on main is a
separate anomaly.
