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

//! Qwen2-VL Image Processor
//!
//! Handles dynamic resolution image preprocessing:
//! 1. Resize preserving aspect ratio within constraints
//! 2. Pad to multiples of patch_size * spatial_merge_size (= 28)
//! 3. Duplicate frame for temporal_patch_size (single image -> 2 frames)
//! 4. Flatten to patch format for vision encoder
//!
//! Used by: Qwen2-VL

use super::ImageProcessor;
use image::DynamicImage;
use image::imageops::FilterType;
use mlxcel_core::{MlxArray, UniquePtr};

mod fast_bicubic;
mod image_config;

pub use image_config::{QwenImageError, QwenImageProcessorConfig};

pub struct Qwen2VLProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub video_default_fps: f64,
    pub video_min_frames: usize,
    pub video_max_frames: usize,
    image_config: Option<QwenImageProcessorConfig>,
}

/// Default lower pixel bound the Qwen2-VL processors are constructed with.
pub const DEFAULT_MIN_PIXELS: usize = 4 * 28 * 28;

/// Default upper pixel bound the Qwen2-VL processors are constructed with.
///
/// `smart_resize` caps the resized area at this value, so it is also what
/// bounds the visual token count for any input image. Shared with
/// [`max_image_tokens`] so a capacity floor derived from it cannot drift away
/// from what the processor actually admits.
pub const DEFAULT_MAX_PIXELS: usize = 16384 * 28 * 28;

/// Qwen-VL video sampling defaults from the published processor sidecars.
pub const DEFAULT_VIDEO_FPS: f64 = crate::multimodal::video::DEFAULT_FPS;
pub const DEFAULT_VIDEO_MIN_FRAMES: usize = crate::multimodal::video::FPS_MIN_FRAMES;
pub const DEFAULT_VIDEO_MAX_FRAMES: usize = crate::multimodal::video::FPS_MAX_FRAMES;

/// Qwen video sampling policy parsed from `processor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenVideoProcessorConfig {
    pub fps: f64,
    pub min_frames: usize,
    pub max_frames: usize,
}

impl Default for QwenVideoProcessorConfig {
    fn default() -> Self {
        Self {
            fps: DEFAULT_VIDEO_FPS,
            min_frames: DEFAULT_VIDEO_MIN_FRAMES,
            max_frames: DEFAULT_VIDEO_MAX_FRAMES,
        }
    }
}

impl QwenVideoProcessorConfig {
    /// Resolve Qwen's video sampling sidecar. Missing keys keep the upstream
    /// defaults; present malformed values are rejected by the loader before a
    /// model can serve video with a silently widened or inverted bound.
    pub fn from_processor_config(config: Option<&serde_json::Value>) -> Result<Self, String> {
        let mut resolved = Self::default();
        if let Some(config) = config {
            if let Some(fps) = config.get("fps") {
                resolved.fps = fps
                    .as_f64()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        "Qwen-VL processor_config.json `fps` must be a finite positive number"
                            .to_string()
                    })?;
            }
            if let Some(min_frames) = config.get("min_frames") {
                resolved.min_frames =
                    usize_from_json(min_frames, "Qwen-VL processor_config.json `min_frames`")?;
            }
            if let Some(max_frames) = config.get("max_frames") {
                resolved.max_frames =
                    usize_from_json(max_frames, "Qwen-VL processor_config.json `max_frames`")?;
            }
        }

        if resolved.min_frames == 0 {
            return Err("Qwen-VL processor_config.json `min_frames` must be positive".to_string());
        }
        if resolved.max_frames < resolved.min_frames {
            return Err(format!(
                "Qwen-VL processor_config.json requires min_frames <= max_frames, got {}..={}",
                resolved.min_frames, resolved.max_frames
            ));
        }

        Ok(resolved)
    }

    #[must_use]
    pub fn frame_sampling(
        self,
        temporal_patch_size: usize,
    ) -> crate::multimodal::video::FrameSamplingPolicy {
        crate::multimodal::video::FrameSamplingPolicy {
            min_frames: self.min_frames,
            max_frames: self.max_frames,
            frame_factor: temporal_patch_size.max(1),
            reject_over_max: true,
        }
    }
}

fn usize_from_json(value: &serde_json::Value, field: &str) -> Result<usize, String> {
    let raw = value
        .as_u64()
        .ok_or_else(|| format!("{field} must be a non-negative integer"))?;
    usize::try_from(raw).map_err(|_| format!("{field} is too large for this platform"))
}

#[derive(Debug, Clone, Copy)]
pub enum Qwen2VLMediaInput<'a> {
    Image(&'a DynamicImage),
    Video(&'a [DynamicImage]),
}

struct MediaPlan<'a> {
    frames: Vec<&'a DynamicImage>,
    width: u32,
    height: u32,
    grid: (i32, i32, i32),
    image: bool,
}

fn tensor_shape(
    grids: &[(i32, i32, i32)],
    patch_size: usize,
    temporal: usize,
) -> Result<([i32; 2], usize), QwenImageError> {
    let features = patch_size
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(3))
        .filter(|value| *value > 0)
        .ok_or(QwenImageError::Capacity)?;
    let rows = grids
        .iter()
        .try_fold(0_usize, |rows, &(time, height, width)| {
            let count =
                [time, height, width]
                    .into_iter()
                    .try_fold(temporal, |count, dimension| {
                        usize::try_from(dimension)
                            .ok()
                            .filter(|value| *value > 0)
                            .and_then(|dimension| count.checked_mul(dimension))
                    })?;
            rows.checked_add(count)
        })
        .ok_or(QwenImageError::Capacity)?;
    let values = rows.checked_mul(features).ok_or(QwenImageError::Capacity)?;
    let shape = [
        i32::try_from(rows).map_err(|_| QwenImageError::Capacity)?,
        i32::try_from(features).map_err(|_| QwenImageError::Capacity)?,
    ];
    Ok((shape, values))
}

/// Largest visual token count one image can expand into under these bounds.
///
/// `smart_resize` rounds both edges to `patch_size * spatial_merge_size` and
/// caps the area at `max_pixels`; `insert_qwen_vl_image_tokens` then emits
/// `(h / merge) * (w / merge)` tokens for a single-frame image. Both reduce to
/// `pixels / (patch_size * spatial_merge_size)^2`, so the maximum is that
/// quotient evaluated at the pixel cap.
///
/// Returns `None` for a degenerate geometry rather than dividing by zero.
#[must_use]
pub fn max_image_tokens(
    patch_size: usize,
    spatial_merge_size: usize,
    max_pixels: usize,
) -> Option<usize> {
    let factor = patch_size.checked_mul(spatial_merge_size)?;
    let divisor = factor.checked_mul(factor)?;
    if divisor == 0 {
        return None;
    }
    Some(max_pixels / divisor)
}

impl Qwen2VLProcessor {
    fn media_plan<'a>(&self, item: Qwen2VLMediaInput<'a>) -> Result<MediaPlan<'a>, String> {
        let (frames, height, width, temporal, image) = match item {
            Qwen2VLMediaInput::Image(image) => {
                let (height, width) = if let Some(config) = &self.image_config {
                    image_config::validate_image(image).map_err(|error| error.to_string())?;
                    let (width, height) = config
                        .image_dimensions(image.width(), image.height())
                        .map_err(|error| error.to_string())?;
                    (height, width)
                } else {
                    self.smart_resize(image.height(), image.width())
                };
                (
                    std::iter::repeat_n(image, self.temporal_patch_size).collect(),
                    height,
                    width,
                    1,
                    true,
                )
            }
            Qwen2VLMediaInput::Video(frames) => {
                let frames = self.pad_video_frames(frames)?;
                let first = frames
                    .first()
                    .ok_or_else(|| "Qwen-VL video contains no decoded frames".to_owned())?;
                let (height, width) = self.smart_resize(first.height(), first.width());
                let temporal = frames.len() / self.temporal_patch_size;
                (frames, height, width, temporal, false)
            }
        };
        let convert =
            |value| i32::try_from(value).map_err(|_| QwenImageError::Capacity.to_string());
        let grid = (
            convert(temporal)?,
            convert(height as usize / self.patch_size)?,
            convert(width as usize / self.patch_size)?,
        );
        Ok(MediaPlan {
            frames,
            width,
            height,
            grid,
            image,
        })
    }

    fn media_plans<'a>(
        &self,
        media: &[Qwen2VLMediaInput<'a>],
    ) -> Result<Vec<MediaPlan<'a>>, String> {
        if let Some(config) = &self.image_config {
            self.validate_image_config(config)
                .map_err(|error| error.to_string())?;
            let mut budget = image_config::ImageBatchBudget::default();
            for item in media {
                if let Qwen2VLMediaInput::Image(image) = item {
                    budget
                        .push(config, image.width(), image.height())
                        .map_err(|error| error.to_string())?;
                    image_config::validate_image(image).map_err(|error| error.to_string())?;
                }
            }
        }
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(media.len())
            .map_err(|_| QwenImageError::Capacity.to_string())?;
        for item in media {
            plans.push(self.media_plan(*item)?);
        }
        Ok(plans)
    }
    /// Used by: Qwen2-VL, Qwen2.5-VL (CLIP normalization)
    pub fn new(patch_size: usize, temporal_patch_size: usize, spatial_merge_size: usize) -> Self {
        crate::vision::image_token_overrides::note_dynamic_resolution_processor();
        Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
            video_default_fps: DEFAULT_VIDEO_FPS,
            video_min_frames: DEFAULT_VIDEO_MIN_FRAMES,
            video_max_frames: DEFAULT_VIDEO_MAX_FRAMES,
            image_config: None,
        }
    }

    /// Used by: Qwen3-VL (simple 0.5/0.5 normalization)
    pub fn new_with_norm(
        patch_size: usize,
        temporal_patch_size: usize,
        spatial_merge_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        crate::vision::image_token_overrides::note_dynamic_resolution_processor();
        Self {
            patch_size,
            temporal_patch_size,
            spatial_merge_size,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            mean,
            std,
            video_default_fps: DEFAULT_VIDEO_FPS,
            video_min_frames: DEFAULT_VIDEO_MIN_FRAMES,
            video_max_frames: DEFAULT_VIDEO_MAX_FRAMES,
            image_config: None,
        }
    }

    #[must_use]
    pub fn with_video_config(mut self, config: QwenVideoProcessorConfig) -> Self {
        self.video_default_fps = config.fps;
        self.video_min_frames = config.min_frames;
        self.video_max_frames = config.max_frames;
        self
    }

    pub fn with_image_config(
        self,
        config: QwenImageProcessorConfig,
    ) -> Result<Self, QwenImageError> {
        self.with_image_config_policy(config, crate::vision::image_token_overrides::installed())
    }

    fn with_image_config_policy(
        mut self,
        config: QwenImageProcessorConfig,
        token_override: Option<&crate::vision::image_token_overrides::ImageTokenOverride>,
    ) -> Result<Self, QwenImageError> {
        self.validate_image_config_policy(&config, token_override)?;
        self.image_config = Some(config);
        Ok(self)
    }

    pub fn image_config(&self) -> Option<&QwenImageProcessorConfig> {
        self.image_config.as_ref()
    }

    fn validate_image_config(
        &self,
        config: &QwenImageProcessorConfig,
    ) -> Result<(), QwenImageError> {
        self.validate_image_config_policy(config, crate::vision::image_token_overrides::installed())
    }

    fn validate_image_config_policy(
        &self,
        config: &QwenImageProcessorConfig,
        token_override: Option<&crate::vision::image_token_overrides::ImageTokenOverride>,
    ) -> Result<(), QwenImageError> {
        config.validate_token_override(token_override)?;
        if self.patch_size != config.patch_size as usize
            || self.temporal_patch_size != config.temporal_patch_size as usize
            || self.spatial_merge_size != config.merge_size as usize
        {
            return Err(QwenImageError::Configuration);
        }
        Ok(())
    }

    pub fn image_grid(&self, width: u32, height: u32) -> Result<(i32, i32, i32), QwenImageError> {
        let config = self
            .image_config
            .as_ref()
            .ok_or(QwenImageError::Configuration)?;
        self.validate_image_config(config)?;
        config.image_grid(width, height)
    }

    #[must_use]
    pub fn video_sampling_policy(&self) -> crate::multimodal::video::FrameSamplingPolicy {
        QwenVideoProcessorConfig {
            fps: self.video_default_fps,
            min_frames: self.video_min_frames,
            max_frames: self.video_max_frames,
        }
        .frame_sampling(self.temporal_patch_size)
    }

    /// The pixel bounds this processor resizes against.
    ///
    /// Normally the checkpoint's own `min_pixels` / `max_pixels`. When an
    /// operator passed b10621's `--image-min-tokens` / `--image-max-tokens`,
    /// [`crate::vision::image_token_overrides`] converts the token budget into
    /// the same two bounds with upstream's own `tokens * patch_size^2 *
    /// merge^2` arithmetic and returns those instead, counting the application
    /// so startup can tell an ignored budget from an applied one.
    ///
    /// Used by: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Qwen3-VL-MoE,
    /// Qwen3-Omni, GLM-4V, GLM-4V-MoE, GLM-OCR, ColQwen2.5.
    #[must_use]
    pub(crate) fn effective_pixel_bounds(&self) -> (usize, usize) {
        crate::vision::image_token_overrides::resolve_pixel_bounds(
            self.patch_size,
            self.spatial_merge_size,
            self.min_pixels,
            self.max_pixels,
        )
    }

    /// Compute target size that satisfies constraints
    /// Returns (height, width) padded to multiples of factor
    pub(crate) fn smart_resize(&self, orig_h: u32, orig_w: u32) -> (u32, u32) {
        let factor = (self.patch_size * self.spatial_merge_size) as u32; // 28
        let (min_pixels, max_pixels) = self.effective_pixel_bounds();

        // Start with original size, round to factor
        let mut h = ((orig_h as f64 / factor as f64).round() as u32).max(1) * factor;
        let mut w = ((orig_w as f64 / factor as f64).round() as u32).max(1) * factor;

        // Ensure within pixel limits
        let pixels = (h * w) as usize;
        if pixels > max_pixels {
            let scale = (max_pixels as f64 / pixels as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).round() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).round() as u32).max(1) * factor;
        }
        if (h * w) as usize > max_pixels {
            // Further reduce if needed
            let scale = (max_pixels as f64 / (h * w) as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).floor() as u32).max(1) * factor;
        }
        let pixels = (h * w) as usize;
        if pixels < min_pixels {
            let scale = (min_pixels as f64 / pixels as f64).sqrt();
            h = ((h as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
            w = ((w as f64 * scale / factor as f64).ceil() as u32).max(1) * factor;
        }

        (h, w)
    }

    /// Compute grid_thw for a set of images
    /// Returns Vec of (temporal, h_patches, w_patches)
    pub fn compute_grid_thw(&self, images: &[image::DynamicImage]) -> Vec<(i32, i32, i32)> {
        images
            .iter()
            .map(|img| {
                let (h, w) = self.smart_resize(img.height(), img.width());
                let h_patches = h as i32 / self.patch_size as i32;
                let w_patches = w as i32 / self.patch_size as i32;
                (1i32, h_patches, w_patches) // temporal=1 for single images
            })
            .collect()
    }

    /// Preprocess images and return (pixel_values, grid_thw)
    pub fn preprocess_with_grid(
        &self,
        images: &[image::DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        let (all_patches, grid_thw) = self.preprocess_values_with_grid(images);
        let in_channels = 3usize;
        let patch_area = self.patch_size * self.patch_size;
        let features_per_pixel = in_channels * patch_area;
        let total_rows: usize = grid_thw
            .iter()
            .map(|&(t, h, w)| (t as usize) * (h as usize) * (w as usize) * self.temporal_patch_size)
            .sum();
        let pixel_values = mlxcel_core::from_slice_f32(
            &all_patches,
            &[total_rows as i32, features_per_pixel as i32],
        );
        (pixel_values, grid_thw)
    }

    /// Pure host form of [`Self::preprocess_with_grid`].
    ///
    /// The OpenXLA path consumes these owned F32 values directly, while the MLX
    /// path wraps the exact same values in an `MlxArray`. Keeping one processor
    /// implementation prevents resize/normalize/patch-order drift.
    pub fn preprocess_values_with_grid(
        &self,
        images: &[DynamicImage],
    ) -> (Vec<f32>, Vec<(i32, i32, i32)>) {
        self.preprocess_media_values_with_grid(
            &images
                .iter()
                .map(Qwen2VLMediaInput::Image)
                .collect::<Vec<_>>(),
        )
        .expect("image-only Qwen-VL preprocessing cannot fail")
    }

    /// Preprocess a mixed image/video sequence in media order and return the
    /// flattened Conv3d patch rows plus one `(t, h, w)` grid per media item.
    ///
    /// For videos, `t` is the number of temporal-patch groups, not raw frames:
    /// a 4-frame clip with `temporal_patch_size=2` has `t=2`. The spatial
    /// axes are shared across all frames in one clip, matching upstream's
    /// `_smart_resize_video` contract.
    pub fn preprocess_media_values_with_grid(
        &self,
        media: &[Qwen2VLMediaInput<'_>],
    ) -> Result<(Vec<f32>, Vec<(i32, i32, i32)>), String> {
        let plans = self.media_plans(media)?;
        let grid_thw: Vec<_> = plans.iter().map(|plan| plan.grid).collect();
        let (_, capacity) = tensor_shape(&grid_thw, self.patch_size, self.temporal_patch_size)
            .map_err(|error| error.to_string())?;
        let mut all_patches: Vec<f32> = Vec::new();
        all_patches
            .try_reserve_exact(capacity)
            .map_err(|_| QwenImageError::Capacity.to_string())?;
        let in_channels = 3usize;

        for plan in plans {
            let (target_h, target_w) = (plan.height, plan.width);
            let (grid_t, h_patches, w_patches) = (
                plan.grid.0 as usize,
                plan.grid.1 as usize,
                plan.grid.2 as usize,
            );
            let normalized_frames = plan
                .frames
                .iter()
                .map(|frame| match (&self.image_config, plan.image) {
                    (Some(config), true) => config
                        .normalize(frame, target_w, target_h)
                        .map_err(|error| error.to_string()),
                    _ => Ok(self.normalize_resized_frame(frame, target_h, target_w)),
                })
                .collect::<Result<Vec<_>, String>>()?;

            for temporal_group in 0..grid_t {
                for block_y in 0..h_patches / self.spatial_merge_size {
                    for block_x in 0..w_patches / self.spatial_merge_size {
                        for inner_y in 0..self.spatial_merge_size {
                            for inner_x in 0..self.spatial_merge_size {
                                let py = block_y * self.spatial_merge_size + inner_y;
                                let px = block_x * self.spatial_merge_size + inner_x;
                                let y_start = py * self.patch_size;
                                let x_start = px * self.patch_size;
                                for tp in 0..self.temporal_patch_size {
                                    let frame_idx = temporal_group * self.temporal_patch_size + tp;
                                    let normalized = &normalized_frames[frame_idx];
                                    for c in 0..in_channels {
                                        for dy in 0..self.patch_size {
                                            for dx in 0..self.patch_size {
                                                let y = y_start + dy;
                                                let x = x_start + dx;
                                                all_patches.push(
                                                    normalized[c
                                                        * target_h as usize
                                                        * target_w as usize
                                                        + y * target_w as usize
                                                        + x],
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if all_patches.len() != capacity {
            return Err(QwenImageError::Capacity.to_string());
        }
        Ok((all_patches, grid_thw))
    }

    pub fn preprocess_media_with_grid(
        &self,
        media: &[Qwen2VLMediaInput<'_>],
    ) -> Result<(UniquePtr<MlxArray>, Vec<(i32, i32, i32)>), String> {
        let (all_patches, grid_thw) = self.preprocess_media_values_with_grid(media)?;
        let (shape, capacity) = tensor_shape(&grid_thw, self.patch_size, self.temporal_patch_size)
            .map_err(|error| error.to_string())?;
        if all_patches.len() != capacity {
            return Err(QwenImageError::Capacity.to_string());
        }
        let pixel_values = mlxcel_core::from_slice_f32(&all_patches, &shape);
        Ok((pixel_values, grid_thw))
    }

    fn normalize_resized_frame(
        &self,
        image: &DynamicImage,
        target_h: u32,
        target_w: u32,
    ) -> Vec<f32> {
        let resized = image.resize_exact(target_w, target_h, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        let h = target_h as usize;
        let w = target_w as usize;
        let mut normalized = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    normalized[c * h * w + y * w + x] = (val - self.mean[c]) / self.std[c];
                }
            }
        }
        normalized
    }

    fn pad_video_frames<'a>(
        &self,
        frames: &'a [DynamicImage],
    ) -> Result<Vec<&'a DynamicImage>, String> {
        if frames.len() < self.video_min_frames {
            return Err(format!(
                "Qwen-VL video has {} decoded frame(s), below min_frames={}",
                frames.len(),
                self.video_min_frames
            ));
        }
        if frames.len() > self.video_max_frames {
            return Err(format!(
                "Qwen-VL video has {} decoded frame(s), exceeding max_frames={}",
                frames.len(),
                self.video_max_frames
            ));
        }
        if self.temporal_patch_size == 0 {
            return Err("Qwen-VL temporal_patch_size must be positive".to_string());
        }
        let padded_len = frames.len().div_ceil(self.temporal_patch_size) * self.temporal_patch_size;
        if padded_len > self.video_max_frames {
            return Err(format!(
                "Qwen-VL video needs {} frame slot(s) after temporal padding, exceeding \
                 max_frames={}",
                padded_len, self.video_max_frames
            ));
        }
        let mut padded = frames.iter().collect::<Vec<_>>();
        if let Some(last) = frames.last() {
            while padded.len() < padded_len {
                padded.push(last);
            }
        }
        Ok(padded)
    }
}

impl ImageProcessor for Qwen2VLProcessor {
    fn preprocess(&self, images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        let (pixel_values, _) = self.preprocess_with_grid(images);
        pixel_values
    }
}

#[cfg(test)]
mod configured_image_tests {
    use super::{
        Qwen2VLMediaInput, Qwen2VLProcessor, QwenImageError, QwenImageProcessorConfig, tensor_shape,
    };
    use crate::vision::image_token_overrides::ImageTokenOverride;
    use image::{DynamicImage, Rgb, RgbImage};
    use serde_json::json;

    fn profile() -> Result<QwenImageProcessorConfig, QwenImageError> {
        QwenImageProcessorConfig::from_processor_config(&json!({
            "image_processor_type":"Qwen2VLImageProcessorFast", "patch_size":1,
            "temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],
            "image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4,"longest_edge":64}
        }))
    }

    fn base() -> Qwen2VLProcessor {
        let mut processor = Qwen2VLProcessor::new_with_norm(1, 2, 2, [0.5; 3], [0.5; 3]);
        processor.min_pixels = 4;
        processor.max_pixels = 64;
        processor
    }

    #[test]
    fn configured_images_keep_media_patch_temporal_and_channel_order() -> Result<(), String> {
        let processor = base()
            .with_image_config(profile().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let first = DynamicImage::ImageRgb8(RgbImage::from_fn(4, 4, |x, y| {
            Rgb([(y * 4 + x) as u8, 64, 255])
        }));
        let second = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([255, 0, 128])));
        let (values, grids) = processor.preprocess_media_values_with_grid(&[
            Qwen2VLMediaInput::Image(&first),
            Qwen2VLMediaInput::Image(&second),
        ])?;
        assert_eq!(grids, [(1, 4, 4), (1, 2, 2)]);
        assert_eq!(values.len(), (16 + 4) * 2 * 3);
        for (patch, red) in [0_u8, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
            .iter()
            .enumerate()
        {
            let expected = [
                (f32::from(*red) - 127.5) / 127.5,
                (64.0 - 127.5) / 127.5,
                1.0,
            ];
            assert_eq!(&values[patch * 6..patch * 6 + 3], &expected);
            assert_eq!(&values[patch * 6 + 3..patch * 6 + 6], &expected);
        }
        assert_eq!(&values[96..99], &[1.0, -1.0, (128.0 - 127.5) / 127.5]);
        assert_eq!(
            processor
                .image_grid(first.width(), first.height())
                .map_err(|error| error.to_string())?,
            grids[0]
        );
        Ok(())
    }

    #[test]
    fn image_profile_does_not_change_video_values_or_legacy_defaults() -> Result<(), String> {
        let legacy = base();
        let configured = base()
            .with_image_config(profile().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let frames = vec![
            DynamicImage::ImageRgb8(RgbImage::from_fn(3, 4, |x, y| {
                Rgb([(x * 70) as u8, (y * 40) as u8, 128])
            }));
            4
        ];
        let media = [Qwen2VLMediaInput::Video(&frames)];
        assert_eq!(
            legacy.preprocess_media_values_with_grid(&media)?,
            configured.preprocess_media_values_with_grid(&media)?
        );
        assert_eq!(legacy.smart_resize(4, 3), (4, 4));
        assert!(legacy.image_config.is_none());
        assert_eq!(
            legacy.video_sampling_policy(),
            configured.video_sampling_policy()
        );
        Ok(())
    }

    #[test]
    fn configured_inputs_and_mutated_geometry_fail_before_pixel_work() -> Result<(), String> {
        let profile = profile().map_err(|error| error.to_string())?;
        assert!(matches!(
            Qwen2VLProcessor::new(14, 2, 2).with_image_config(profile.clone()),
            Err(QwenImageError::Configuration)
        ));
        let mut processor = base()
            .with_image_config(profile)
            .map_err(|error| error.to_string())?;
        for image in [
            DynamicImage::new_rgb8(0, 1),
            DynamicImage::new_rgb8(201, 1),
            DynamicImage::new_rgb16(2, 2),
        ] {
            assert!(
                processor
                    .preprocess_media_values_with_grid(&[Qwen2VLMediaInput::Image(&image)])
                    .is_err()
            );
        }
        assert_eq!(
            processor.image_grid(u32::MAX, u32::MAX),
            Err(QwenImageError::Geometry)
        );
        assert!(
            processor
                .preprocess_media_values_with_grid(&[Qwen2VLMediaInput::Video(&[])])
                .is_err()
        );
        processor.patch_size = usize::MAX;
        assert_eq!(
            processor.image_grid(2, 2),
            Err(QwenImageError::Configuration)
        );
        assert!(
            processor
                .preprocess_media_values_with_grid(&[Qwen2VLMediaInput::Image(
                    &DynamicImage::new_rgb8(2, 2)
                )])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn profile_accessor_preserves_exact_installed_configuration() -> Result<(), QwenImageError> {
        let expected = profile()?;
        assert_eq!(base().image_config(), None);
        let installed = base().with_image_config(expected.clone())?;
        assert_eq!(installed.image_config(), Some(&expected));
        Ok(())
    }

    #[test]
    fn configured_image_boundaries_reject_token_overrides() -> Result<(), QwenImageError> {
        let profile = QwenImageProcessorConfig::from_processor_config(&json!({
            "image_processor_type":"Qwen2VLImageProcessorFast", "patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],
            "image_std":[0.5,0.5,0.5],
            "size":{"shortest_edge":65536,"longest_edge":16777216}
        }))?;
        assert_eq!(profile.image_grid(1024, 1024)?, (1, 64, 64));
        let processor = Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
            .with_image_config_policy(profile.clone(), None)?;
        assert_eq!(processor.image_config(), Some(&profile));
        for (minimum, maximum) in [
            (None, Some(256)),
            (Some(256), None),
            (Some(64), Some(256)),
            (None, Some(0)),
        ] {
            let token_override = ImageTokenOverride::from_bounds(minimum, maximum);
            assert_eq!(
                profile.validate_token_override(token_override.as_ref()),
                Err(QwenImageError::TokenOverride)
            );
            assert!(matches!(
                Qwen2VLProcessor::new_with_norm(16, 2, 2, [0.5; 3], [0.5; 3])
                    .with_image_config_policy(profile.clone(), token_override.as_ref()),
                Err(QwenImageError::TokenOverride)
            ));
            assert_eq!(
                processor.validate_image_config_policy(&profile, token_override.as_ref()),
                Err(QwenImageError::TokenOverride)
            );
        }
        Ok(())
    }

    #[test]
    fn image_batch_rejects_excess_count_before_preprocessing() -> Result<(), String> {
        let processor = base()
            .with_image_config(profile().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let image = DynamicImage::new_rgb8(2, 2);
        let images = [Qwen2VLMediaInput::Image(&image); 8];
        assert_eq!(processor.media_plans(&images)?.len(), 8);
        assert!(
            processor
                .media_plans(&[Qwen2VLMediaInput::Image(&image); 9])
                .is_err()
        );
        assert!(
            processor
                .preprocess_media_values_with_grid(&[Qwen2VLMediaInput::Image(&image); 9])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn tensor_boundary_checks_cumulative_rows_and_feature_products() -> Result<(), QwenImageError> {
        assert_eq!(
            tensor_shape(&[(1, 2, 2), (2, 2, 2)], 16, 2)?,
            ([24, 768], 18432)
        );
        assert_eq!(
            tensor_shape(&[(1, i32::MAX / 2, 1); 2], 1, 2),
            Err(QwenImageError::Capacity)
        );
        assert_eq!(
            tensor_shape(&[(i32::MAX, i32::MAX, i32::MAX)], 1, 16),
            Err(QwenImageError::Capacity)
        );
        assert_eq!(
            tensor_shape(&[(1, 1, 1)], usize::MAX, 2),
            Err(QwenImageError::Capacity)
        );
        assert_eq!(
            tensor_shape(&[(1, 0, 1)], 16, 2),
            Err(QwenImageError::Capacity)
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, Rgb, RgbImage};

    use super::*;

    #[test]
    fn owned_and_mlx_paths_share_spatial_merge_grouped_patch_order() {
        let processor = Qwen2VLProcessor::new(14, 2, 2);
        let mut image = RgbImage::new(56, 56);
        for patch_y in 0..4u32 {
            for patch_x in 0..4u32 {
                let value = (patch_y * 4 + patch_x) as u8 * 10;
                for y in patch_y * 14..(patch_y + 1) * 14 {
                    for x in patch_x * 14..(patch_x + 1) * 14 {
                        image.put_pixel(x, y, Rgb([value, 0, 0]));
                    }
                }
            }
        }
        let image = DynamicImage::ImageRgb8(image);
        let (values, grids) = processor.preprocess_values_with_grid(&[image]);
        assert_eq!(grids, vec![(1, 4, 4)]);
        let row_width = 3 * 14 * 14;
        let observed = (0..16)
            .map(|patch| values[patch * 2 * row_width])
            .collect::<Vec<_>>();
        let grouped_patch_ids = [0u8, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
        for (actual, patch_id) in observed.into_iter().zip(grouped_patch_ids) {
            let expected = (patch_id as f32 * 10.0 / 255.0 - processor.mean[0]) / processor.std[0];
            assert!((actual - expected).abs() < 1e-6);
        }
        let (mlx, mlx_grids) = processor.preprocess_with_grid(&[DynamicImage::new_rgb8(56, 56)]);
        assert_eq!(mlx_grids, grids);
        assert_eq!(mlxcel_core::array_shape(&mlx), vec![32, row_width as i32]);
    }

    #[test]
    fn video_processor_config_reads_qwen_sidecar_bounds() {
        let config = serde_json::json!({
            "fps": 1.5,
            "min_frames": 6,
            "max_frames": 128
        });
        let parsed = QwenVideoProcessorConfig::from_processor_config(Some(&config)).unwrap();
        assert_eq!(parsed.fps, 1.5);
        assert_eq!(parsed.min_frames, 6);
        assert_eq!(parsed.max_frames, 128);
    }

    #[test]
    fn video_processor_config_rejects_inverted_frame_bounds() {
        let config = serde_json::json!({
            "min_frames": 8,
            "max_frames": 4
        });
        let err = QwenVideoProcessorConfig::from_processor_config(Some(&config)).unwrap_err();
        assert!(err.contains("min_frames <= max_frames"), "{err}");
    }

    #[test]
    fn preprocess_video_pads_frames_to_temporal_patch_grid() {
        let processor =
            Qwen2VLProcessor::new(14, 2, 2).with_video_config(QwenVideoProcessorConfig {
                fps: 2.0,
                min_frames: 1,
                max_frames: 4,
            });
        let frames = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([10, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([20, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(56, 56, Rgb([30, 0, 0]))),
        ];
        let (pixel_values, grids) = processor
            .preprocess_media_with_grid(&[Qwen2VLMediaInput::Video(&frames)])
            .unwrap();

        assert_eq!(grids, vec![(2, 4, 4)]);
        assert_eq!(
            mlxcel_core::array_shape(&pixel_values),
            vec![64, 3 * 14 * 14]
        );
    }
}
