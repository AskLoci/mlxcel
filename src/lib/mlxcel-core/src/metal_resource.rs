// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! iOS shader payload and the upstream startup-only path configuration.

#[cxx::bridge(namespace = "mlx::core::metal")]
mod ffi {
    unsafe extern "C++" {
        include!("mlx/backend/metal/metal.h");

        /// # Safety
        /// Call once before any other MLX operation or worker can run.
        unsafe fn set_metallib_path(path: &CxxString) -> Result<()>;
    }
}

/// The Metal library compiled for this iOS target and deployment minimum.
///
/// Publish these bytes to an app-private, persistent file before configuring
/// the path with [set_metallib_path]. Device and simulator libraries differ;
/// the build script selects the artifact produced by this compilation.
pub fn embedded_metallib() -> &'static [u8] {
    include_bytes!(env!("MLXCEL_EMBEDDED_METALLIB"))
}

/// Set the path of the embedded Metal library before starting MLX.
///
/// The path must name the complete library returned by [embedded_metallib].
/// Keep that file available for the lifetime of the MLX runtime. A C++
/// allocation failure is returned as an exception; this call does not load
/// the library or initialize a device.
///
/// # Safety
///
/// Call at most once, during process startup, before any MLX operation or
/// worker can run. Upstream stores this path in an unsynchronized global
/// string; concurrent setters or MLX library loading would be a data race.
pub unsafe fn set_metallib_path(path: &str) -> Result<(), cxx::Exception> {
    cxx::let_cxx_string!(native_path = path);
    // SAFETY: The caller owns the startup ordering required by upstream.
    unsafe { ffi::set_metallib_path(&native_path) }
}
