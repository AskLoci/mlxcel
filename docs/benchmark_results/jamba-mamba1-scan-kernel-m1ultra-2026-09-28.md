# Fused Metal kernel for the Jamba Mamba1 selective scan (issue #2005)

After #2000 the Jamba Mamba scan was linear but still a Rust loop that issued
several small ops per timestep in each of the 26 Mamba layers. The new kernel
(`mamba1_selective_scan`, mlxcel-core) walks all timesteps of a layer in one
dispatch: one simdgroup per (batch, channel), lane `n` holds `state[d, n]` in
float32, and `simd_sum` over the lanes gives `y[t, d]`. It serves prefill and
decode, so the two stay numerically consistent. `MLXCEL_MAMBA1_SCAN_KERNEL=0`
restores the graph scan; non-Metal backends always use it.

- **Date:** 2026-09-28
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428), display attached
- **Build:** `cargo build --release --features metal,accelerate`
- **Model:** AI21 Jamba reasoning 3B 4-bit (28 layers: 2 attention, 26 Mamba;
  intermediate 5120, d_state 16; bf16 scales and activations)
- **Arms:** one binary, kernel on and `MLXCEL_MAMBA1_SCAN_KERNEL=0`, order off
  on on off, GEMM check between prompt lengths 13.7 to 13.9 TFLOPS

## Ceiling (before building the kernel)

A measurement build replaced only the scan with a shape-correct trivial
computation that still consumed `x_proj`, `dt_proj`, B and C. Prefill tok/s:

| Prompt | Graph scan | Scan removed |
|---|---|---|
| 512 | 821 | 1317 |
| 1024 | 829 | 1442 |
| 2048 | 835 | 1477 |

## Results (tok/s, two runs per arm)

| Prompt | Prefill off | Prefill on | Change | Decode off | Decode on | Peak GB off / on |
|---|---|---|---|---|---|---|
| 512 | 831.4, 831.5 | 1208.0, 1201.7 | +45% | 129.3, 126.5 | 142.2, 141.5 | 3.02 / 3.10 |
| 1024 | 860.7, 857.9 | 1316.0, 1332.2 | +54% | 127.5, 127.0 | 141.8, 140.1 | 3.31 / 2.86 |
| 2048 | 870.4, 866.5 | 1328.8, 1347.8 | +54% | 127.4, 130.5 | 139.2, 139.0 | 4.11 / 3.15 |

The kernel reaches about 90% of the ceiling at 1024 and 2048 tokens. Decode
gains about 10% because the single-token step also runs as one dispatch per
layer. The two `[B, L, D, N]` graph intermediates are gone, which lowers peak
memory at longer prompts.

## Quality

The kernel carries the state in float32; the graph scan rounds the bf16 state
at every step. They are compared by quality, not byte identity.

- Teacher-forced perplexity (`examples/perplexity`, WikiText-2 excerpt, 1024
  tokens x 4): graph 10.1075, kernel 10.0988.
- Logit trace (`examples/logit_trace`, 1024 tokens x 2, top-8,
  `scripts/compare_logit_traces.py`): 64 top-1 disagreements, all at positions
  where the graph path's top two were within one logit (none at a gap of 1 or
  more), 92% of them the graph's second choice; logit delta on the graph's
  choice p50 0.06, p99 0.56; trace perplexity 15.357 to 15.271 (-0.56%).
- Greedy against main (100 tokens, reasoning shown): identical on a short
  prompt; a 512-token prompt diverges at character 3267 of 3635 and an
  about 1100-token prompt at character 6129 of 6570, both into fluent text.

## Tests

- `mamba1_scan_parity_tests`: the kernel matches a scalar float32 reference
  (relative error under 1e-5) for batch 2, 24 channels (not a multiple of the
  eight rows per threadgroup), d_state 8 and 16, one and seven steps, fresh and
  carried state; with bf16 inputs its RMS error against the reference is no
  larger than a bf16-rounding emulation of the graph scan. Dropping the decay
  term fails both (f32 relative error 0.25; bf16 RMS error 1.43 against 0.0022).
- The existing Jamba prefill-vs-decode row tests pass with the kernel on both
  paths.

## Not measured

CUDA and ROCm (graph scan there); Mamba1 / Falcon-Mamba (`mamba.rs`), which has
the same recurrence and can adopt the kernel in a follow-up.
