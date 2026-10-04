use image::{Rgb, RgbImage};

use super::image_config::QwenImageError;

struct Weights {
    start: u32,
    values: Vec<i16>,
}

struct Axis {
    precision: u32,
    weights: Vec<Weights>,
}

fn cubic(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn floating_weights(
    input: u32,
    output: u32,
    index: u32,
) -> Result<(u32, Vec<f64>), QwenImageError> {
    let scale = f64::from(input) / f64::from(output);
    let support = 2.0 * scale.max(1.0);
    let center = scale * (f64::from(index) + 0.5);
    let start = (center - support + 0.5).max(0.0) as u32;
    let end = (center + support + 0.5).min(f64::from(input)) as u32;
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut weights = Vec::new();
    weights
        .try_reserve_exact((end - start) as usize)
        .map_err(|_| QwenImageError::Capacity)?;
    let mut total = 0.0;
    for source in start..end {
        let weight = cubic((f64::from(source) - center + 0.5) * inverse_scale);
        weights.push(weight);
        total += weight;
    }
    if total == 0.0 || !total.is_finite() {
        return Err(QwenImageError::Geometry);
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Ok((start, weights))
}

impl Axis {
    fn new(input: u32, output: u32) -> Result<Self, QwenImageError> {
        if input == 0 || output == 0 || input > 268_435_456 || output > 16_777_216 {
            return Err(QwenImageError::Geometry);
        }
        let mut raw = Vec::new();
        raw.try_reserve_exact(output as usize)
            .map_err(|_| QwenImageError::Capacity)?;
        let mut maximum = 0.0_f64;
        for index in 0..output {
            let (start, weights) = floating_weights(input, output, index)?;
            for weight in &weights {
                maximum = maximum.max(*weight);
            }
            raw.push((start, weights));
        }
        let mut precision = 0;
        while precision < 22 {
            if (0.5 + maximum * f64::from(1_u32 << (precision + 1))).trunc() >= 32768.0 {
                break;
            }
            precision += 1;
        }
        if precision == 0 {
            return Err(QwenImageError::Geometry);
        }
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(raw.len())
            .map_err(|_| QwenImageError::Capacity)?;
        for (start, raw_values) in raw {
            let mut values = Vec::new();
            values
                .try_reserve_exact(raw_values.len())
                .map_err(|_| QwenImageError::Capacity)?;
            for value in raw_values {
                let value = value * f64::from(1_u32 << precision);
                let rounded = if value < 0.0 {
                    value - 0.5
                } else {
                    value + 0.5
                };
                if !rounded.is_finite()
                    || rounded.trunc() < f64::from(i16::MIN)
                    || rounded.trunc() > f64::from(i16::MAX)
                {
                    return Err(QwenImageError::Geometry);
                }
                values.push(rounded as i16);
            }
            weights.push(Weights { start, values });
        }
        Ok(Self { precision, weights })
    }
}

fn allocate(width: u32, height: u32) -> Result<RgbImage, QwenImageError> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || pixels > 268_435_456 {
        return Err(QwenImageError::Capacity);
    }
    let length = usize::try_from(pixels.checked_mul(3).ok_or(QwenImageError::Capacity)?)
        .map_err(|_| QwenImageError::Capacity)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| QwenImageError::Capacity)?;
    bytes.resize(length, 0);
    RgbImage::from_raw(width, height, bytes).ok_or(QwenImageError::Capacity)
}

fn pass(input: &RgbImage, output_size: u32, horizontal: bool) -> Result<RgbImage, QwenImageError> {
    let input_size = if horizontal {
        input.width()
    } else {
        input.height()
    };
    let axis = Axis::new(input_size, output_size)?;
    let (width, height) = if horizontal {
        (output_size, input.height())
    } else {
        (input.width(), output_size)
    };
    let mut output = allocate(width, height)?;
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let weights = &axis.weights[if horizontal { x } else { y } as usize];
        let mut accumulators = [1_i64 << (axis.precision - 1); 3];
        for (offset, weight) in weights.values.iter().enumerate() {
            let position =
                weights.start + u32::try_from(offset).map_err(|_| QwenImageError::Geometry)?;
            let source = if horizontal {
                input.get_pixel(position, y)
            } else {
                input.get_pixel(x, position)
            };
            for channel in 0..3 {
                accumulators[channel] += i64::from(source[channel]) * i64::from(*weight);
            }
        }
        let mut values = [0; 3];
        for (value, accumulator) in values.iter_mut().zip(accumulators) {
            *value = u8::try_from((accumulator >> axis.precision).clamp(0, 255))
                .map_err(|_| QwenImageError::Geometry)?;
        }
        *pixel = Rgb(values);
    }
    Ok(output)
}

pub(super) fn resize(
    input: &RgbImage,
    width: u32,
    height: u32,
) -> Result<RgbImage, QwenImageError> {
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 16_777_216 {
        return Err(QwenImageError::Geometry);
    }
    if input.width() == 0 || input.height() == 0 {
        return Err(QwenImageError::Geometry);
    }
    let horizontal;
    let intermediate = if width == input.width() {
        input
    } else {
        horizontal = pass(input, width, true)?;
        &horizontal
    };
    if height != input.height() {
        pass(intermediate, height, false)
    } else {
        let mut output = allocate(width, height)?;
        output.as_mut().copy_from_slice(intermediate.as_raw());
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{Axis, resize};
    use crate::vision::processors::qwen2_vl::QwenImageError;
    use image::{Rgb, RgbImage};

    fn gray(width: u32, height: u32, values: &[u8]) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            Rgb([values[(y * width + x) as usize]; 3])
        })
    }

    fn values(image: &RgbImage) -> Vec<u8> {
        image.pixels().map(|pixel| pixel[0]).collect()
    }

    #[test]
    fn fast_uint8_coefficients_use_axis_specific_signed_precision() -> Result<(), QwenImageError> {
        let up = Axis::new(2, 3)?;
        assert_eq!(up.precision, 14);
        assert_eq!(up.weights[1].values, [8192, 8192]);
        let down = Axis::new(4, 2)?;
        assert_eq!(down.precision, 16);
        assert_eq!(down.weights[0].values, [30060, 30060, 7853, -2437]);
        Ok(())
    }

    #[test]
    fn controlled_ramps_cover_antialias_edges_and_rounding() -> Result<(), QwenImageError> {
        let input = gray(2, 1, &[0, 255]);
        assert_eq!(values(&resize(&input, 3, 1)?), [0, 128, 255]);
        assert_eq!(values(&resize(&input, 4, 1)?), [0, 53, 202, 255]);
        assert_eq!(
            values(&resize(&gray(4, 1, &[0, 64, 128, 255]), 2, 1)?),
            [35, 183]
        );
        assert_eq!(values(&resize(&input, 2, 1)?), [0, 255]);
        Ok(())
    }

    #[test]
    fn two_pass_uint8_rounding_preserves_channel_order() -> Result<(), QwenImageError> {
        let input = gray(2, 2, &[0, 64, 128, 255]);
        assert_eq!(
            values(&resize(&input, 3, 3)?),
            [0, 21, 56, 60, 112, 162, 128, 203, 255]
        );
        let rgb = RgbImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([0, 64, 255])
            } else {
                Rgb([255, 128, 0])
            }
        });
        let output = resize(&rgb, 3, 1)?;
        assert_eq!(output.get_pixel(1, 0), &Rgb([128, 96, 128]));
        assert_eq!(resize(&rgb, 0, 1), Err(QwenImageError::Geometry));
        assert_eq!(resize(&rgb, u32::MAX, 2), Err(QwenImageError::Geometry));
        Ok(())
    }
}
