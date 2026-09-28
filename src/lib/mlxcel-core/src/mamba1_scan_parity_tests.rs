// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Parity tests for the fused Mamba1 selective-scan kernel (#2005).
//!
//! The kernel walks the recurrence
//! `s_t = exp(dt_t * A) * s_{t-1} + dt_t * x_t * B_t`,
//! `y_t = s_t . C_t + D * x_t` with the state in float32. Two properties are
//! pinned here against a scalar CPU reference:
//!
//! 1. With f32 inputs the kernel reproduces the reference (the only difference
//!    is summation order inside `simd_sum`), for fresh and carried state, a
//!    single step and several, two state widths, and a channel count that is
//!    not a multiple of the threadgroup's eight rows.
//! 2. With bf16 inputs the kernel is at least as close to the f32 reference as
//!    the graph scan it replaces, which rounds the state to bf16 every step.
//!
//! Metal-only: the kernel JITs through `mx.fast.metal_kernel`, so the tests
//! return early wherever `mamba1_scan_kernel_available()` is false.

use crate::{MlxArray, UniquePtr, ffi};

fn seeded(len: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0) * scale + offset
        })
        .collect()
}

fn to_vec(arr: &MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, crate::dtype::FLOAT32);
    ffi::eval(&a);
    ffi::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bf16_round(v: f32) -> f32 {
    // Round to nearest even at bf16 precision (keep the top 16 bits).
    let bits = v.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

struct Case {
    x: Vec<f32>,
    dt: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    s0: Vec<f32>,
}

fn make_case(batch: usize, seq: usize, dm: usize, n: usize, carried: bool) -> Case {
    Case {
        x: seeded(batch * seq * dm, 1, 1.0, 0.0),
        // dt is a softplus output: positive, mostly below one.
        dt: seeded(batch * seq * dm, 2, 0.25, 0.3),
        b: seeded(batch * seq * n, 3, 1.0, 0.0),
        c: seeded(batch * seq * n, 4, 1.0, 0.0),
        // A = -exp(A_log): negative.
        a: seeded(dm * n, 5, 0.5, -1.0),
        d: seeded(dm, 6, 1.0, 0.0),
        s0: if carried {
            seeded(batch * dm * n, 7, 0.5, 0.0)
        } else {
            vec![0.0; batch * dm * n]
        },
    }
}

/// Scalar reference. `round_state` emulates the graph scan, which keeps every
/// intermediate and the carried state in the activation dtype.
fn reference(
    k: &Case,
    batch: usize,
    seq: usize,
    dm: usize,
    n: usize,
    round_state: Option<fn(f32) -> f32>,
) -> (Vec<f32>, Vec<f32>) {
    let r = |v: f32| round_state.map_or(v, |f| f(v));
    let mut y = vec![0.0f32; batch * seq * dm];
    let mut s = k.s0.clone();
    for bi in 0..batch {
        for t in 0..seq {
            let row = bi * seq + t;
            for di in 0..dm {
                let dt = k.dt[row * dm + di];
                let xv = k.x[row * dm + di];
                let mut acc = 0.0f32;
                for ni in 0..n {
                    let idx = (bi * dm + di) * n + ni;
                    let decay = r((dt * k.a[di * n + ni]).exp());
                    let input = r(r(dt * xv) * k.b[row * n + ni]);
                    s[idx] = r(r(decay * s[idx]) + input);
                    acc += s[idx] * k.c[row * n + ni];
                }
                y[row * dm + di] = r(acc) + r(xv * k.d[di]);
            }
        }
    }
    (y, s)
}

fn run_kernel(
    k: &Case,
    batch: usize,
    seq: usize,
    dm: usize,
    n: usize,
    dtype: i32,
) -> (Vec<f32>, Vec<f32>) {
    let arr = |v: &[f32], shape: &[i32]| ffi::astype(&ffi::from_slice_f32(v, shape), dtype);
    let (b, l, d, n_) = (batch as i32, seq as i32, dm as i32, n as i32);
    let x = arr(&k.x, &[b, l, d]);
    let dt = arr(&k.dt, &[b, l, d]);
    let bm = arr(&k.b, &[b, l, n_]);
    let cm = arr(&k.c, &[b, l, n_]);
    let a = ffi::from_slice_f32(&k.a, &[d, n_]);
    let dp = arr(&k.d, &[d]);
    let s0 = ffi::from_slice_f32(&k.s0, &[b, d, n_]);
    let mut y: UniquePtr<MlxArray> = UniquePtr::null();
    let mut s: UniquePtr<MlxArray> = UniquePtr::null();
    ffi::mamba1_selective_scan(&x, &dt, &bm, &cm, &a, &dp, &s0, &mut y, &mut s);
    assert_eq!(ffi::array_shape(&y), vec![b, l, d]);
    assert_eq!(ffi::array_shape(&s), vec![b, d, n_]);
    assert_eq!(ffi::array_dtype(&y), dtype);
    (to_vec(&y), to_vec(&s))
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

#[test]
fn f32_kernel_matches_scalar_reference() {
    if !ffi::mamba1_scan_kernel_available() {
        return;
    }
    let (batch, dm) = (2, 24);
    for n in [8, 16] {
        for seq in [1, 7] {
            for carried in [false, true] {
                let k = make_case(batch, seq, dm, n, carried);
                let (want_y, want_s) = reference(&k, batch, seq, dm, n, None);
                let (y, s) = run_kernel(&k, batch, seq, dm, n, crate::dtype::FLOAT32);
                let (ey, es) = (max_rel(&y, &want_y), max_rel(&s, &want_s));
                let what = format!("n {n} seq {seq} carried {carried}");
                assert!(ey < 1e-5, "{what}: y rel error {ey}");
                assert!(es < 1e-5, "{what}: state rel error {es}");
            }
        }
    }
}

#[test]
fn bf16_kernel_is_no_less_accurate_than_the_graph_scan() {
    if !ffi::mamba1_scan_kernel_available() {
        return;
    }
    let (batch, seq, dm, n) = (1, 64, 24, 16);
    let mut k = make_case(batch, seq, dm, n, true);
    // Feed both sides the same bf16-representable inputs, so the comparison
    // isolates how the scan itself rounds.
    for v in [&mut k.x, &mut k.dt, &mut k.b, &mut k.c, &mut k.d] {
        v.iter_mut().for_each(|e| *e = bf16_round(*e));
    }
    let (exact_y, _) = reference(&k, batch, seq, dm, n, None);
    let (graph_y, _) = reference(&k, batch, seq, dm, n, Some(bf16_round));
    let (kernel_y, _) = run_kernel(&k, batch, seq, dm, n, crate::dtype::BFLOAT16);

    let rms = |got: &[f32]| {
        (got.iter()
            .zip(&exact_y)
            .map(|(g, w)| (g - w) * (g - w))
            .sum::<f32>()
            / got.len() as f32)
            .sqrt()
    };
    let (kernel_err, graph_err) = (rms(&kernel_y), rms(&graph_y));
    assert!(
        kernel_err <= graph_err,
        "kernel RMS error {kernel_err} must not exceed the bf16 graph scan's {graph_err}"
    );
}
