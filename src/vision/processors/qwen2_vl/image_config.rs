use image::{DynamicImage, RgbImage};
use serde_json::Value;
use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::io::Read;
use std::path::Path;

use super::fast_bicubic;
use crate::vision::image_token_overrides::ImageTokenOverride;

const MAX_PIXELS: u32 = 16_777_216;
const MAX_INPUT_PIXELS: u64 = 268_435_456;

#[derive(Default)]
pub(super) struct ImageBatchBudget {
    images: usize,
    input_pixels: u64,
    output_pixels: u64,
    values: u64,
}

impl ImageBatchBudget {
    pub(super) fn push(
        &mut self,
        config: &QwenImageProcessorConfig,
        width: u32,
        height: u32,
    ) -> Result<(), QwenImageError> {
        let (target_width, target_height) = config.image_dimensions(width, height)?;
        let images = self.images.checked_add(1).ok_or(QwenImageError::Capacity)?;
        let input_pixels = self
            .input_pixels
            .checked_add(u64::from(width) * u64::from(height))
            .ok_or(QwenImageError::Capacity)?;
        let pixels = u64::from(target_width) * u64::from(target_height);
        let output_pixels = self
            .output_pixels
            .checked_add(pixels)
            .ok_or(QwenImageError::Capacity)?;
        let values = pixels
            .checked_mul(3)
            .and_then(|value| value.checked_mul(u64::from(config.temporal_patch_size)))
            .and_then(|value| self.values.checked_add(value))
            .ok_or(QwenImageError::Capacity)?;
        if images > 8
            || input_pixels > MAX_INPUT_PIXELS
            || output_pixels > u64::from(MAX_PIXELS)
            || values > u64::from(MAX_PIXELS) * 6
        {
            return Err(QwenImageError::Capacity);
        }
        self.images = images;
        self.input_pixels = input_pixels;
        self.output_pixels = output_pixels;
        self.values = values;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenImageError {
    Configuration,
    Geometry,
    Capacity,
    PixelFormat,
    Sidecar,
    TokenOverride,
}

impl Display for QwenImageError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Configuration => "unsupported Qwen image processor configuration",
            Self::Geometry => "unsupported Qwen image dimensions",
            Self::Capacity => "Qwen image preprocessing capacity exceeded",
            Self::PixelFormat => "unsupported Qwen image pixel format",
            Self::Sidecar => "invalid Qwen image processor sidecar",
            Self::TokenOverride => {
                "image-token overrides are unsupported for configured Qwen images"
            }
        })
    }
}

impl std::error::Error for QwenImageError {}

#[derive(Clone, Debug, PartialEq)]
pub struct QwenImageProcessorConfig {
    pub(super) patch_size: u32,
    pub(super) temporal_patch_size: u32,
    pub(super) merge_size: u32,
    min_pixels: u32,
    max_pixels: u32,
    mean: [f32; 3],
    std: [f32; 3],
}

impl QwenImageProcessorConfig {
    pub(crate) fn validate_token_override(
        &self,
        token_override: Option<&ImageTokenOverride>,
    ) -> Result<(), QwenImageError> {
        if token_override.is_some() {
            return Err(QwenImageError::TokenOverride);
        }
        Ok(())
    }

    pub fn from_model_path(path: &Path) -> Result<Option<Self>, QwenImageError> {
        let processor = read_sidecar(path, "processor_config.json")?;
        let preprocessor = read_sidecar(path, "preprocessor_config.json")?;
        Self::from_sidecars(processor.as_ref(), preprocessor.as_ref())
    }

    pub fn from_sidecars(
        processor: Option<&Value>,
        preprocessor: Option<&Value>,
    ) -> Result<Option<Self>, QwenImageError> {
        if processor.is_some_and(|value| !value.is_object())
            || preprocessor.is_some_and(|value| !value.is_object())
        {
            return Err(QwenImageError::Configuration);
        }
        let processor = processor.and_then(|value| value.get("image_processor"));
        let first = processor.map(Self::from_processor_config).transpose()?;
        let second = preprocessor.map(Self::from_processor_config).transpose()?;
        match (first, second) {
            (Some(first), Some(second)) if first != second => Err(QwenImageError::Configuration),
            (Some(config), _) | (_, Some(config)) => Ok(Some(config)),
            (None, None) => Ok(None),
        }
    }

    pub fn from_processor_config(config: &Value) -> Result<Self, QwenImageError> {
        let config = config.get("image_processor").unwrap_or(config);
        validate_operations(config)?;
        let (min_pixels, max_pixels) = match config.get("size") {
            Some(size) => (
                positive_integer(size, "shortest_edge")?,
                positive_integer(size, "longest_edge")?,
            ),
            None => (
                positive_integer(config, "min_pixels")?,
                positive_integer(config, "max_pixels")?,
            ),
        };
        let resolved = Self {
            patch_size: positive_integer(config, "patch_size")?,
            temporal_patch_size: positive_integer(config, "temporal_patch_size")?,
            merge_size: positive_integer(config, "merge_size")?,
            min_pixels,
            max_pixels,
            mean: channel_values(config, "image_mean")?,
            std: channel_values(config, "image_std")?,
        };
        let factor = resolved.factor()?;
        let minimum = factor
            .checked_mul(factor)
            .ok_or(QwenImageError::Configuration)?;
        if resolved.min_pixels < minimum
            || resolved.max_pixels < resolved.min_pixels
            || resolved.max_pixels > MAX_PIXELS
            || resolved.temporal_patch_size > 16
            || resolved.std.iter().any(|value| *value <= 0.0)
        {
            return Err(QwenImageError::Configuration);
        }
        for (key, expected) in [
            ("min_pixels", resolved.min_pixels),
            ("max_pixels", resolved.max_pixels),
        ] {
            if config
                .get(key)
                .is_some_and(|value| value.as_u64() != Some(u64::from(expected)))
            {
                return Err(QwenImageError::Configuration);
            }
        }
        Ok(resolved)
    }

    pub fn image_grid(&self, width: u32, height: u32) -> Result<(i32, i32, i32), QwenImageError> {
        let (width, height) = self.image_dimensions(width, height)?;
        Ok((
            1,
            i32::try_from(height / self.patch_size).map_err(|_| QwenImageError::Geometry)?,
            i32::try_from(width / self.patch_size).map_err(|_| QwenImageError::Geometry)?,
        ))
    }

    pub fn image_dimensions(&self, width: u32, height: u32) -> Result<(u32, u32), QwenImageError> {
        if width == 0
            || height == 0
            || u64::from(width.max(height)) > u64::from(width.min(height)) * 200
            || u64::from(width) * u64::from(height) > MAX_INPUT_PIXELS
        {
            return Err(QwenImageError::Geometry);
        }
        let factor = f64::from(self.factor()?);
        let original_width = f64::from(width);
        let original_height = f64::from(height);
        let mut resized_width = (original_width / factor).round_ties_even() * factor;
        let mut resized_height = (original_height / factor).round_ties_even() * factor;
        let area = resized_width * resized_height;
        if area > f64::from(self.max_pixels) {
            let beta = (original_width * original_height / f64::from(self.max_pixels)).sqrt();
            resized_width = (original_width / beta / factor).floor().max(1.0) * factor;
            resized_height = (original_height / beta / factor).floor().max(1.0) * factor;
        } else if area < f64::from(self.min_pixels) {
            let beta = (f64::from(self.min_pixels) / (original_width * original_height)).sqrt();
            resized_width = (original_width * beta / factor).ceil() * factor;
            resized_height = (original_height * beta / factor).ceil() * factor;
        }
        if resized_width < 1.0
            || resized_height < 1.0
            || resized_width * resized_height > f64::from(MAX_PIXELS)
        {
            return Err(QwenImageError::Geometry);
        }
        Ok((resized_width as u32, resized_height as u32))
    }

    fn factor(&self) -> Result<u32, QwenImageError> {
        self.patch_size
            .checked_mul(self.merge_size)
            .filter(|value| *value > 0)
            .ok_or(QwenImageError::Configuration)
    }

    pub(super) fn normalize(
        &self,
        image: &DynamicImage,
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, QwenImageError> {
        let input = rgb_input(image)?;
        let rgb = fast_bicubic::resize(&input, width, height)?;
        let plane = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| QwenImageError::Capacity)?;
        let length = plane.checked_mul(3).ok_or(QwenImageError::Capacity)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| QwenImageError::Capacity)?;
        output.resize(length, 0.0);
        for (index, pixel) in rgb.pixels().enumerate() {
            for channel in 0..3 {
                output[channel * plane + index] = (f32::from(pixel[channel])
                    - self.mean[channel] * 255.0)
                    / (self.std[channel] * 255.0);
            }
        }
        Ok(output)
    }
}

pub(super) fn validate_image(image: &DynamicImage) -> Result<(), QwenImageError> {
    if !matches!(
        image,
        DynamicImage::ImageRgb8(_)
            | DynamicImage::ImageRgba8(_)
            | DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLumaA8(_)
    ) {
        return Err(QwenImageError::PixelFormat);
    }
    rgb_length(image.width(), image.height())?;
    Ok(())
}

fn rgb_length(width: u32, height: u32) -> Result<usize, QwenImageError> {
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > MAX_INPUT_PIXELS {
        return Err(QwenImageError::Capacity);
    }
    usize::try_from(pixels.checked_mul(3).ok_or(QwenImageError::Capacity)?)
        .map_err(|_| QwenImageError::Capacity)
}

fn rgb_input(image: &DynamicImage) -> Result<Cow<'_, RgbImage>, QwenImageError> {
    validate_image(image)?;
    if let Some(rgb) = image.as_rgb8() {
        return Ok(Cow::Borrowed(rgb));
    }
    let length = rgb_length(image.width(), image.height())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| QwenImageError::Capacity)?;
    match image {
        DynamicImage::ImageRgba8(image) => {
            for pixel in image.pixels() {
                bytes.extend_from_slice(&pixel.0[..3]);
            }
        }
        DynamicImage::ImageLuma8(image) => {
            for pixel in image.pixels() {
                bytes.extend_from_slice(&[pixel[0]; 3]);
            }
        }
        DynamicImage::ImageLumaA8(image) => {
            for pixel in image.pixels() {
                bytes.extend_from_slice(&[pixel[0]; 3]);
            }
        }
        _ => return Err(QwenImageError::PixelFormat),
    }
    RgbImage::from_raw(image.width(), image.height(), bytes)
        .map(Cow::Owned)
        .ok_or(QwenImageError::Capacity)
}

fn read_sidecar(path: &Path, name: &str) -> Result<Option<Value>, QwenImageError> {
    let file = match std::fs::File::open(path.join(name)) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(QwenImageError::Sidecar),
    };
    decode_sidecar(file).map(Some)
}

fn decode_sidecar(reader: impl Read) -> Result<Value, QwenImageError> {
    let mut bytes = Vec::new();
    reader
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(|_| QwenImageError::Sidecar)?;
    if bytes.len() > 65536 {
        return Err(QwenImageError::Sidecar);
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|_| QwenImageError::Sidecar)?;
    if !value.is_object() {
        return Err(QwenImageError::Sidecar);
    }
    Ok(value)
}

fn positive_integer(config: &Value, key: &str) -> Result<u32, QwenImageError> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or(QwenImageError::Configuration)
}

fn channel_values(config: &Value, key: &str) -> Result<[f32; 3], QwenImageError> {
    let values = config
        .get(key)
        .and_then(Value::as_array)
        .filter(|values| values.len() == 3)
        .ok_or(QwenImageError::Configuration)?;
    let mut result = [0.0; 3];
    for (slot, value) in result.iter_mut().zip(values) {
        let value = value
            .as_f64()
            .filter(|value| value.is_finite() && value.abs() <= 1.0)
            .ok_or(QwenImageError::Configuration)?;
        *slot = value as f32;
    }
    Ok(result)
}

fn validate_operations(config: &Value) -> Result<(), QwenImageError> {
    if !matches!(
        config.get("image_processor_type").and_then(Value::as_str),
        Some("Qwen2VLImageProcessorFast" | "Qwen3VLImageProcessor")
    ) {
        return Err(QwenImageError::Configuration);
    }
    for name in ["do_resize", "do_rescale", "do_normalize", "do_convert_rgb"] {
        if config
            .get(name)
            .is_some_and(|value| value.as_bool() != Some(true))
        {
            return Err(QwenImageError::Configuration);
        }
    }
    if config
        .get("resample")
        .is_some_and(|value| value.as_u64() != Some(3))
        || config
            .get("rescale_factor")
            .is_some_and(|value| value.as_f64() != Some(1.0 / 255.0))
        || config
            .get("data_format")
            .is_some_and(|value| value.as_str() != Some("channels_first"))
    {
        return Err(QwenImageError::Configuration);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ImageBatchBudget, QwenImageError, QwenImageProcessorConfig, decode_sidecar, rgb_input,
        rgb_length,
    };
    use image::{DynamicImage, GrayAlphaImage, LumaA, Rgba, RgbaImage};
    use serde_json::{Value, json};
    use std::borrow::Cow;
    use std::io::{Error, ErrorKind, Read};

    fn pinned_config() -> Value {
        json!({"image_processor_type":"Qwen2VLImageProcessorFast", "patch_size":16,
            "temporal_patch_size":2, "merge_size":2, "image_mean":[0.5,0.5,0.5],
            "image_std":[0.5,0.5,0.5], "size":{"shortest_edge":65536,"longest_edge":16777216}})
    }

    #[test]
    fn fast_geometry_uses_ties_even_original_area_and_axis_order() -> Result<(), QwenImageError> {
        let profile = QwenImageProcessorConfig::from_processor_config(&pinned_config())?;
        assert_eq!(profile.image_dimensions(272, 512)?, (256, 512));
        assert_eq!(profile.image_grid(1024, 512)?, (1, 32, 64));
        assert_eq!(profile.image_grid(512, 1024)?, (1, 64, 32));
        assert_eq!(profile.image_dimensions(1, 1)?, (256, 256));
        assert_eq!(profile.image_dimensions(8192, 4096)?, (5792, 2880));
        assert_eq!(
            profile.image_dimensions(0, 1),
            Err(QwenImageError::Geometry)
        );
        assert_eq!(
            profile.image_dimensions(201, 1),
            Err(QwenImageError::Geometry)
        );
        assert_eq!(
            profile.image_dimensions(u32::MAX, u32::MAX),
            Err(QwenImageError::Geometry)
        );
        Ok(())
    }

    #[test]
    fn unsupported_declared_operations_fail_closed() {
        for (key, value) in [
            ("do_resize", json!(false)),
            ("resample", json!(1)),
            ("rescale_factor", json!(1.0)),
            ("do_convert_rgb", json!(null)),
            ("image_processor_type", json!("Qwen2VLImageProcessor")),
            ("data_format", json!("channels_last")),
            ("patch_size", json!(0)),
            ("image_std", json!([0.0, 0.5, 0.5])),
        ] {
            let mut config = pinned_config();
            config[key] = value;
            assert_eq!(
                QwenImageProcessorConfig::from_processor_config(&config),
                Err(QwenImageError::Configuration)
            );
        }
    }

    #[test]
    fn qwen3_vl_processor_bounds_resolve_to_the_same_profile() -> Result<(), QwenImageError> {
        let processor = json!({"image_processor":{"do_convert_rgb":true, "do_normalize":true,
            "do_rescale":true, "image_mean":[0.5,0.5,0.5],
            "image_processor_type":"Qwen3VLImageProcessor", "image_std":[0.5,0.5,0.5],
            "max_pixels":16777216, "merge_size":2, "min_pixels":65536, "patch_size":16,
            "rescale_factor":0.00392156862745098, "temporal_patch_size":2},
            "processor_class":"Qwen3VLProcessor"});
        let preprocessor = pinned_config();
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&processor), Some(&preprocessor))?,
            Some(QwenImageProcessorConfig::from_processor_config(&preprocessor)?)
        );
        let mut unbounded = processor.clone();
        if let Some(config) = unbounded["image_processor"].as_object_mut() {
            config.remove("max_pixels");
        }
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&unbounded), None),
            Err(QwenImageError::Configuration)
        );
        let mut wider = processor.clone();
        wider["image_processor"]["max_pixels"] = json!(8388608);
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&wider), Some(&preprocessor)),
            Err(QwenImageError::Configuration)
        );
        let mut renamed = processor;
        renamed["image_processor"]["image_processor_type"] = json!("Qwen3VLImageProcessorSlow");
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&renamed), None),
            Err(QwenImageError::Configuration)
        );
        Ok(())
    }

    #[test]
    fn sidecars_must_agree_and_malformed_data_is_not_absence() -> Result<(), QwenImageError> {
        let direct = pinned_config();
        let nested = json!({"image_processor":direct});
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&nested), Some(&direct))?,
            Some(QwenImageProcessorConfig::from_processor_config(&direct)?)
        );
        assert_eq!(QwenImageProcessorConfig::from_sidecars(None, None)?, None);
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&Value::Null), None),
            Err(QwenImageError::Configuration)
        );
        let mut changed = direct.clone();
        changed["size"]["shortest_edge"] = json!(131072);
        assert_eq!(
            QwenImageProcessorConfig::from_sidecars(Some(&nested), Some(&changed)),
            Err(QwenImageError::Configuration)
        );
        for invalid in [b"{".as_slice(), b"null", b"[]"] {
            assert_eq!(decode_sidecar(invalid), Err(QwenImageError::Sidecar));
        }
        assert_eq!(
            decode_sidecar(vec![b' '; 65537].as_slice()),
            Err(QwenImageError::Sidecar)
        );
        struct FailedReader;
        impl Read for FailedReader {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(Error::new(ErrorKind::PermissionDenied, "fixture"))
            }
        }
        assert_eq!(decode_sidecar(FailedReader), Err(QwenImageError::Sidecar));
        Ok(())
    }

    #[test]
    fn rgb_conversion_precedes_resize_and_fast_normalization() -> Result<(), QwenImageError> {
        let profile = QwenImageProcessorConfig::from_processor_config(&pinned_config())?;
        let rgba = DynamicImage::ImageRgba8(RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([0, 64, 255, 0])
            } else {
                Rgba([255, 128, 0, 255])
            }
        }));
        let values = profile.normalize(&rgba, 3, 1)?;
        assert_eq!(values[1].to_bits(), ((128.0_f32 - 127.5) / 127.5).to_bits());
        assert_eq!(values[4].to_bits(), ((96.0_f32 - 127.5) / 127.5).to_bits());
        assert_eq!(values[7].to_bits(), values[1].to_bits());
        assert_eq!(values[0], -1.0);
        assert_eq!(values[6], 1.0);
        let unsupported = DynamicImage::new_rgb16(1, 1);
        assert_eq!(
            profile.normalize(&unsupported, 1, 1),
            Err(QwenImageError::PixelFormat)
        );
        Ok(())
    }

    #[test]
    fn rgb_input_borrows_rgb8_and_checked_conversion_discards_alpha() -> Result<(), QwenImageError>
    {
        let rgb = DynamicImage::new_rgb8(2, 1);
        assert!(matches!(rgb_input(&rgb)?, Cow::Borrowed(_)));
        let gray_alpha =
            DynamicImage::ImageLumaA8(GrayAlphaImage::from_pixel(1, 1, LumaA([19, 0])));
        let converted = rgb_input(&gray_alpha)?;
        assert!(matches!(&converted, Cow::Owned(_)));
        assert_eq!(converted.as_raw(), &[19, 19, 19]);
        assert_eq!(
            rgb_length(u32::MAX, u32::MAX),
            Err(QwenImageError::Capacity)
        );
        assert_eq!(rgb_length(0, 1), Err(QwenImageError::Capacity));
        assert_eq!(rgb_length(16384, 16384)?, 805306368);
        Ok(())
    }

    #[test]
    fn image_budget_limits_cumulative_input_output_and_expanded_values()
    -> Result<(), QwenImageError> {
        let profile = QwenImageProcessorConfig::from_processor_config(&pinned_config())?;
        let mut budget = ImageBatchBudget::default();
        budget.push(&profile, 4096, 4096)?;
        assert_eq!(budget.push(&profile, 2, 2), Err(QwenImageError::Capacity));
        let mut small = pinned_config();
        small["patch_size"] = json!(1);
        small["size"] = json!({"shortest_edge":4,"longest_edge":64});
        let small = QwenImageProcessorConfig::from_processor_config(&small)?;
        let mut inputs = ImageBatchBudget::default();
        inputs.push(&small, 16384, 8192)?;
        inputs.push(&small, 16384, 8192)?;
        assert_eq!(inputs.push(&small, 2, 2), Err(QwenImageError::Capacity));
        let mut more_temporal = profile.clone();
        more_temporal.temporal_patch_size = 16;
        assert_eq!(
            ImageBatchBudget::default().push(&more_temporal, 4096, 4096),
            Err(QwenImageError::Capacity)
        );
        let mut overflow = ImageBatchBudget {
            images: 1,
            input_pixels: u64::MAX,
            ..ImageBatchBudget::default()
        };
        assert_eq!(overflow.push(&small, 2, 2), Err(QwenImageError::Capacity));
        Ok(())
    }
}
