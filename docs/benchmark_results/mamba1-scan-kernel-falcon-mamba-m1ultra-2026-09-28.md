# Fused Mamba1 scan kernel for Mamba and Falcon-Mamba (issue #2007)

`MambaBlock::forward` (model types `mamba` and `falcon_mamba`) ran prefill as a
Rust loop that called `ssm_step` once per timestep, and `ssm_step` ran `x_proj`
and `dt_proj` on a single row each time. With the Metal kernel from #2005
available, the block now projects the whole sequence through `x_proj`, the
mixer norms and `dt_proj` once and runs `mamba1_selective_scan` for prefill and
decode. `MLXCEL_MAMBA1_SCAN_KERNEL=0` and non-Metal backends keep the per-step
path.

- **Date:** 2026-09-28
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428), display attached
- **Model:** `falcon-mamba-7b-4bit-instruct` (64 layers, hidden 4096,
  intermediate 8192, d_state 16, mixer RMS norm on B, C and dt)
- **Arms:** one binary, kernel on and `MLXCEL_MAMBA1_SCAN_KERNEL=0`, order off
  on on off, `mlxcel-bench-decode` warmup 20, tg128, GEMM check between prompt
  lengths 13.5 to 14.1 TFLOPS

## Results (tok/s, two runs per arm)

| Prompt | Prefill off | Prefill on | Change | Decode off | Decode on |
|---|---|---|---|---|---|
| 512 | 198.9, 200.9 | 665.6, 665.1 | 3.3x | 73.2, 74.2 | 81.2, 80.7 |
| 1024 | 204.9, 199.3 | 742.2, 736.6 | 3.7x | 73.9, 73.7 | 81.0, 80.0 |
| 2048 | crashes | 756.9, 755.4 | | | 81.1, 81.3 |

At 2048 prompt tokens the per-step path aborts with `[metal::malloc] Resource
limit (499000) exceeded`: 64 layers of per-timestep ops allocate more Metal
buffers than the device allows. main (`61f7851d`) aborts the same way; 1536
tokens still runs. The kernel path runs at every length tried. Decode gains
about 9%. Prefill peak memory is 0.2 to 0.4 GB higher (5.17 to 5.55 GB at 512,
6.39 to 6.58 GB at 1024) because the projections now cover the whole sequence
at once.

## Quality

The kernel carries the state in float32; the per-step path rounds it to the
activation dtype every step.

- Teacher-forced perplexity (`examples/perplexity`, WikiText-2 excerpt, 1024 x
  4): per-step 13.5135, kernel 13.5435 (+0.2%).
- Logit trace (`examples/logit_trace`, 1024 x 2, top-8): 13 top-1
  disagreements, every one the per-step path's second choice and every one at a
  position where its top two were within one logit; trace perplexity 17.4027
  to 17.4009 (-0.01%). The two perplexity readings disagree in sign and are both
  within 0.2%, so the kernel is at parity rather than better or worse.
- Greedy against main (100 tokens): identical on a short and a 512-token
  prompt; an about 1100-token prompt diverges at character 6175 of 6560.

## Tests

- `mamba_block_prefill_rows_match_token_by_token_forward`: a seeded
  `MambaBlock` gives the same rows for a multi-token forward as for the same
  inputs fed one token at a time, fresh and continuing from an existing state,
  with and without the Falcon-Mamba mixer norm. Starting every call from a zero
  state fails it.
- The kernel itself is covered by `mamba1_scan_parity_tests` (#2005).

## Not measured

A plain `mamba` checkpoint (the code path is shared with `falcon_mamba` except
for the mixer norm, which the test covers both ways); CUDA and ROCm.
