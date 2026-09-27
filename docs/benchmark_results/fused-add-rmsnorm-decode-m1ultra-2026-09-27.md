# Fused residual-add + RMSNorm through the production decode path

`MLXCEL_FUSED_ADD_RMSNORM` (#905) replaces an elementwise add and a
`fast::rms_norm` at the Llama3-family mid-block residual join with one kernel.
It shipped default off because the op-level microbench could not resolve the
difference from dispatch cost. This record measures it through the production
decode loop.

- **Date:** 2026-09-27
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** main `d1c3df02` plus a temporary one-time diagnostic line in the
  fused branch of `fused_add_rms_norm` (not committed), `--features
  metal,accelerate`
- **Arms:** one binary, `MLXCEL_FUSED_ADD_RMSNORM` unset (off) and `1` (on),
  order off on on off, twice
- **Condition:** `mlxcel-bench-decode` with `bench_decode.sh`'s arguments
  (warmup 20, tg128, `--ignore-eos`, `--prompt-tokens 512`), indexers
  suspended, fp16 GEMM check between models 13.4 to 14.0 TFLOPS
- **Arm check:** the diagnostic line appeared in every on run and in no off run

## Results (mean of four runs per arm, tok/s)

| Model | Decode off | Decode on | Change | Prefill off | Prefill on |
|---|---|---|---|---|---|
| Llama 3.1 8B 4-bit | 113.18 | 113.01 | -0.15% | 821.05 | 821.09 |
| Qwen2.5 7B 4-bit | 115.38 | 115.34 | -0.03% | 873.36 | 873.62 |

Per-run decode ranges overlap (Llama 112.36 to 114.03 off and 112.77 to
113.28 on; Qwen2.5 115.11 to 115.61 off and 115.08 to 115.46 on). Greedy
output (512-token prompt, 100 tokens) is identical with the kernel on and off
for both models.

## Reading

A serial pre-norm block's join is one add and one RMSNorm; fusing them removes
one small dispatch per join, which decode on this machine does not register.
The Cohere2 fusion that did gain (+1.1% to +2.7%, #1948) merged the two adds of
a parallel block and the following LayerNorm, removing two dispatches and a
barrier stage per layer. The default stays off.

## Not measured

Other Apple generations, CUDA and ROCm; the block-boundary join (`h + ff_out`
into the next layer's input norm), which is the same size of fusion and was not
pursued.
