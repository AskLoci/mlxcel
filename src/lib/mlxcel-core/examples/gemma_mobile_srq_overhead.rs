//! Per-token cost of the Gemma mobile rounding chain around decode-sized
//! quantized matmuls, isolated from the model: 340 chained projections per
//! token, one row of 1536, 4-bit weights. Prints the median of ten timed
//! tokens per variant, three alternating rounds.
use cxx::UniquePtr;
use mlxcel_core as mc;
use mlxcel_core::{MlxArray, dtype};
use std::time::Instant;

const PROJECTIONS: usize = 340;
const K: i32 = 1536;
const N: i32 = 1536;
const TOKENS: usize = 10;

struct Weight {
    packed: UniquePtr<MlxArray>,
    scales_f32: UniquePtr<MlxArray>,
    biases_f32: UniquePtr<MlxArray>,
    scales_bf16: UniquePtr<MlxArray>,
    biases_bf16: UniquePtr<MlxArray>,
}

fn weight() -> Weight {
    let packed_columns = K * 4 / 32;
    let bytes: Vec<u8> = (0..(N * packed_columns * 4) as usize)
        .map(|index| (index * 31 % 251) as u8)
        .collect();
    let scales_f32 = mc::full_f32(&[N, K / 128], 0.01, dtype::FLOAT32);
    let biases_f32 = mc::full_f32(&[N, K / 128], -0.08, dtype::FLOAT32);
    Weight {
        packed: mc::from_bytes(&bytes, &[N, packed_columns], dtype::UINT32),
        scales_bf16: mc::astype(&scales_f32, dtype::BFLOAT16),
        biases_bf16: mc::astype(&biases_f32, dtype::BFLOAT16),
        scales_f32,
        biases_f32,
    }
}

fn qmm(x: &MlxArray, w: &Weight, f32: bool) -> UniquePtr<MlxArray> {
    let (scales, biases) = if f32 {
        (&w.scales_f32, &w.biases_f32)
    } else {
        (&w.scales_bf16, &w.biases_bf16)
    };
    unsafe {
        mc::quantized_linear_forward(
            x,
            &w.packed,
            scales,
            &**biases as *const MlxArray,
            std::ptr::null(),
            128,
            4,
            "affine",
        )
    }
}

fn step(variant: &str, x: &MlxArray, scale: &MlxArray, w: &Weight) -> UniquePtr<MlxArray> {
    match variant {
        "regular" => qmm(x, w, false),
        "before" => {
            let rounded = mc::compiled_gemma_mobile_srq(x, scale, dtype::BFLOAT16, dtype::BFLOAT16);
            let input = mc::astype(&rounded, dtype::FLOAT32);
            let output = mc::astype(&qmm(&input, w, true), dtype::BFLOAT16);
            mc::compiled_gemma_mobile_srq(&output, scale, dtype::BFLOAT16, dtype::BFLOAT16)
        }
        "after" => {
            let input = mc::compiled_gemma_mobile_srq(x, scale, dtype::BFLOAT16, dtype::FLOAT32);
            let output = qmm(&input, w, true);
            mc::compiled_gemma_mobile_srq(&output, scale, dtype::BFLOAT16, dtype::BFLOAT16)
        }
        "roundings_only" => {
            let rounded = mc::compiled_gemma_mobile_srq(x, scale, dtype::BFLOAT16, dtype::BFLOAT16);
            mc::compiled_gemma_mobile_srq(&rounded, scale, dtype::BFLOAT16, dtype::BFLOAT16)
        }
        "casts_only" => mc::astype(&mc::astype(x, dtype::FLOAT32), dtype::BFLOAT16),
        "custom" => {
            let input = mc::custom_gemma_mobile_srq(x, scale, dtype::BFLOAT16, dtype::FLOAT32);
            let output = qmm(&input, w, true);
            mc::custom_gemma_mobile_srq(&output, scale, dtype::BFLOAT16, dtype::BFLOAT16)
        }
        "custom_roundings_only" => {
            let rounded = mc::custom_gemma_mobile_srq(x, scale, dtype::BFLOAT16, dtype::BFLOAT16);
            mc::custom_gemma_mobile_srq(&rounded, scale, dtype::BFLOAT16, dtype::BFLOAT16)
        }
        "qmm_f32" => {
            let input = mc::astype(x, dtype::FLOAT32);
            mc::astype(&qmm(&input, w, true), dtype::BFLOAT16)
        }
        _ => unreachable!(),
    }
}

fn token_ms(variant: &str, x: &MlxArray, scale: &MlxArray, w: &Weight) -> f64 {
    let start = Instant::now();
    let mut current = mc::copy(x);
    for _ in 0..PROJECTIONS {
        current = step(variant, &current, scale, w);
    }
    mc::eval(&current);
    mc::synchronize_default();
    start.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let data: Vec<f32> = (0..K as usize)
        .map(|index| ((index * 13 % 73) as i32 - 36) as f32 / 37.0)
        .collect();
    let x = mc::astype(&mc::from_slice_f32(&data, &[1, K]), dtype::BFLOAT16);
    let scale = mc::full_f32(&[], 0.3, dtype::BFLOAT16);
    let w = weight();
    mc::eval(&x);
    mc::eval(&w.packed);
    let variants = [
        "regular",
        "before",
        "after",
        "custom",
        "roundings_only",
        "custom_roundings_only",
        "casts_only",
        "qmm_f32",
    ];
    println!("round,variant,median_ms_per_token,min_ms_per_token");
    for round in 1..=3 {
        for variant in variants {
            for _ in 0..3 {
                token_ms(variant, &x, &scale, &w);
            }
            let mut times: Vec<f64> = (0..TOKENS)
                .map(|_| token_ms(variant, &x, &scale, &w))
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min = times[0];
            let median = times[TOKENS / 2];
            println!("{round},{variant},{median:.2},{min:.2}");
        }
    }
}
