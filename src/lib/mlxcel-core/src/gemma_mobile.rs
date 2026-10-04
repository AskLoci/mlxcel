use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidLayout,
    InvalidScale,
    InvalidActivation,
    InvalidMarker,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidLayout => "invalid Gemma mobile packed tensor layout",
            Self::InvalidScale => "invalid Gemma mobile weight scale",
            Self::InvalidActivation => "invalid Gemma mobile activation scale",
            Self::InvalidMarker => "invalid Gemma mobile runtime metadata",
        })
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub rows: i32,
    pub columns: i32,
    pub bits: i32,
    pub group_size: i32,
    repeats: i32,
}

impl Layout {
    pub fn new(
        packed: &[i32],
        scales: &[i32],
        rows: i32,
        columns: i32,
        dtype: i32,
    ) -> Result<Self, Error> {
        let [packed_rows, packed_columns] = packed else {
            return Err(Error::InvalidLayout);
        };
        let [scale_rows, scale_columns] = scales else {
            return Err(Error::InvalidLayout);
        };
        if rows <= 0
            || columns <= 0
            || *packed_rows != rows
            || *scale_rows != rows
            || *packed_columns <= 0
            || packed_columns % 4 != 0
            || *scale_columns <= 0
            || columns % scale_columns != 0
        {
            return Err(Error::InvalidLayout);
        }
        let packed_bits = i64::from(*packed_columns) * 8;
        let bits =
            i32::try_from(packed_bits / i64::from(columns)).map_err(|_| Error::InvalidLayout)?;
        if ![2, 4, 8].contains(&bits)
            || packed_bits != i64::from(columns) * i64::from(bits)
            || dtype != if bits == 8 { dtype::INT8 } else { dtype::UINT8 }
        {
            return Err(Error::InvalidLayout);
        }
        let group_size = [128, 64, 32]
            .into_iter()
            .find(|group| columns % group == 0 && (columns / scale_columns) % group == 0)
            .ok_or(Error::InvalidLayout)?;
        Ok(Self {
            rows,
            columns,
            bits,
            group_size,
            repeats: columns / group_size / scale_columns,
        })
    }
}

pub struct Prepared {
    pub weight: UniquePtr<MlxArray>,
    pub scales: UniquePtr<MlxArray>,
    pub biases: UniquePtr<MlxArray>,
    pub layout: Layout,
}

fn floating(dtype: i32) -> bool {
    matches!(dtype, dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16)
}

fn nonnegative_finite(value: &MlxArray) -> bool {
    let finite = ffi::all_all(&ffi::isfinite(value));
    let nonnegative = ffi::all_all(&ffi::greater_equal(value, &ffi::zeros_like(value)));
    ffi::item_bool(&finite) && ffi::item_bool(&nonnegative)
}

pub fn prepare(
    packed: &MlxArray,
    scale: &MlxArray,
    rows: i32,
    columns: i32,
    linear: bool,
) -> Result<Prepared, Error> {
    let layout = Layout::new(
        &ffi::array_shape(packed),
        &ffi::array_shape(scale),
        rows,
        columns,
        ffi::array_dtype(packed),
    )?;
    if !floating(ffi::array_dtype(scale)) || !nonnegative_finite(scale) {
        return Err(Error::InvalidScale);
    }
    let unsigned = if layout.bits == 8 {
        ffi::bitwise_xor(
            &ffi::view(packed, dtype::UINT8),
            &ffi::full_f32(&[], 128.0, dtype::UINT8),
        )
    } else {
        ffi::copy(packed)
    };
    let weight = ffi::view(&unsigned, dtype::UINT32);
    let repeated = ffi::repeat(scale, layout.repeats, 1);
    let zero_point = ffi::full_f32(
        &[],
        -((1_i32 << (layout.bits - 1)) as f32),
        ffi::array_dtype(scale),
    );
    let offsets = ffi::multiply(&repeated, &zero_point);
    let scales = if linear {
        ffi::astype(&repeated, dtype::FLOAT32)
    } else {
        repeated
    };
    let biases = if linear {
        ffi::astype(&offsets, dtype::FLOAT32)
    } else {
        offsets
    };
    Ok(Prepared {
        weight,
        scales,
        biases,
        layout,
    })
}

pub fn is_mobile(weights: &WeightMap, prefix: &str) -> bool {
    weights.contains_key(&format!("{prefix}.gemma_mobile_layout"))
}

pub fn insert(weights: &mut WeightMap, prefix: &str, prepared: Prepared) {
    let layout = prepared.layout;
    weights.insert(format!("{prefix}.weight"), prepared.weight);
    weights.insert(format!("{prefix}.scales"), prepared.scales);
    weights.insert(format!("{prefix}.biases"), prepared.biases);
    weights.insert(
        format!("{prefix}.gemma_mobile_layout"),
        ffi::from_slice_i32(
            &[layout.rows, layout.columns, layout.group_size, layout.bits],
            &[4],
        ),
    );
}

pub fn parameters(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> Result<(i32, i32), Error> {
    let Some(marker) = weights.get(&format!("{prefix}.gemma_mobile_layout")) else {
        return Ok((group_size, bits));
    };
    if mode != "affine"
        || ffi::array_shape(marker) != [4]
        || ffi::array_dtype(marker) != dtype::INT32
    {
        return Err(Error::InvalidMarker);
    }
    let bytes = ffi::array_to_raw_bytes(marker);
    let mut values = bytes
        .chunks_exact(4)
        .map(|chunk| <[u8; 4]>::try_from(chunk).map(i32::from_ne_bytes));
    let mut next = || {
        values
            .next()
            .and_then(Result::ok)
            .ok_or(Error::InvalidMarker)
    };
    let (rows, columns, group, width) = (next()?, next()?, next()?, next()?);
    if rows <= 0
        || columns <= 0
        || ![32, 64, 128].contains(&group)
        || ![2, 4, 8].contains(&width)
        || columns % group != 0
    {
        return Err(Error::InvalidMarker);
    }
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .ok_or(Error::InvalidMarker)?;
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .ok_or(Error::InvalidMarker)?;
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .ok_or(Error::InvalidMarker)?;
    let packed_columns = i64::from(columns) * i64::from(width) / 32;
    if ffi::array_shape(weight)
        != [i64::from(rows), packed_columns].map(|v| i32::try_from(v).unwrap_or(-1))
        || ffi::array_dtype(weight) != dtype::UINT32
        || ffi::array_shape(scales) != [rows, columns / group]
        || ffi::array_shape(biases) != ffi::array_shape(scales)
        || !floating(ffi::array_dtype(scales))
        || ffi::array_dtype(scales) != ffi::array_dtype(biases)
    {
        return Err(Error::InvalidMarker);
    }
    Ok((group, width))
}

pub struct Activation {
    input: UniquePtr<MlxArray>,
    output: UniquePtr<MlxArray>,
}

pub fn validate_activation(scale: &MlxArray) -> Result<(), Error> {
    if !ffi::array_shape(scale).is_empty()
        || !floating(ffi::array_dtype(scale))
        || !nonnegative_finite(scale)
    {
        return Err(Error::InvalidActivation);
    }
    Ok(())
}

impl Activation {
    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Option<Self>, Error> {
        if !is_mobile(weights, prefix) {
            return Ok(None);
        }
        if let Some(bias) = weights.get(&format!("{prefix}.bias")) {
            let weight = weights
                .get(&format!("{prefix}.weight"))
                .ok_or(Error::InvalidMarker)?;
            let rows = ffi::array_shape(weight)
                .first()
                .copied()
                .ok_or(Error::InvalidMarker)?;
            if ffi::array_shape(bias) != [rows] || !floating(ffi::array_dtype(bias)) {
                return Err(Error::InvalidLayout);
            }
        }
        let scale = |suffix: &str| -> Result<UniquePtr<MlxArray>, Error> {
            let value = weights
                .get(&format!("{prefix}.{suffix}"))
                .map(|value| ffi::copy(value))
                .unwrap_or_else(|| ffi::full_f32(&[], 0.0, dtype::FLOAT32));
            validate_activation(&value)?;
            Ok(value)
        };
        Ok(Some(Self {
            input: scale("input_activation_scale")?,
            output: scale("output_activation_scale")?,
        }))
    }

    pub fn clone_shared(&self) -> Self {
        Self {
            input: ffi::copy(&self.input),
            output: ffi::copy(&self.output),
        }
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        weight: &super::QuantizedWeight,
        bias: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let original_dtype = ffi::array_dtype(x);
        let accumulation = accumulation_dtype(&ffi::array_shape(x), original_dtype);
        let input = ffi::astype(&srq(x, &self.input), accumulation);
        let scales = ffi::astype(&weight.scales, accumulation);
        let offsets = weight
            .biases
            .as_ref()
            .map(|value| ffi::astype(value, accumulation));
        let offset_pointer = offsets
            .as_ref()
            .and_then(|value| value.as_ref())
            .map_or(std::ptr::null(), |value| value as *const MlxArray);
        let output = unsafe {
            ffi::quantized_linear_forward(
                &input,
                &weight.weight,
                &scales,
                offset_pointer,
                std::ptr::null(),
                weight.group_size,
                weight.bits,
                "affine",
            )
        };
        let output = match bias {
            Some(bias) => ffi::add(&output, &ffi::astype(bias, accumulation)),
            None => output,
        };
        srq(&ffi::astype(&output, original_dtype), &self.output)
    }
}

fn accumulation_dtype(shape: &[i32], original: i32) -> i32 {
    let rows = shape.split_last().and_then(|(_, leading)| {
        leading.iter().try_fold(1_u64, |rows, dimension| {
            u64::try_from(*dimension)
                .ok()
                .and_then(|dimension| rows.checked_mul(dimension))
        })
    });
    if rows.is_some_and(|rows| rows < 32) {
        dtype::FLOAT32
    } else {
        original
    }
}

fn srq(input: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let scale = ffi::astype(scale, ffi::array_dtype(input));
    let calibrated = ffi::not_equal(&scale, &ffi::zeros_like(&scale));
    let safe = ffi::where_cond(&calibrated, &scale, &ffi::ones_like(&scale));
    let rounded = ffi::round(&ffi::divide(input, &safe));
    let lower = ffi::full_like(&scale, -128.0);
    let upper = ffi::full_like(&scale, 127.0);
    let quantized = ffi::multiply(&ffi::clip(&rounded, &lower, &upper), &safe);
    ffi::where_cond(&calibrated, &quantized, input)
}

#[cfg(test)]
mod tests {
    use super::{
        Error, Layout, accumulation_dtype, insert, parameters, prepare, srq, validate_activation,
    };
    use crate::layers::{FusedQKVLinear, UnifiedEmbedding, UnifiedLinear};
    use crate::weights::WeightMap;
    use crate::{dtype, ffi};

    fn packed_values(
        bits: i32,
        rows: usize,
        columns: usize,
    ) -> (cxx::UniquePtr<ffi::MlxArray>, Vec<f32>) {
        let offset = 1_i32 << (bits - 1);
        let values: Vec<i32> = (0..rows * columns)
            .map(|index| ((index * 7 + index / columns) % (1 << bits)) as i32 - offset)
            .collect();
        let per_byte = (8 / bits) as usize;
        let bytes: Vec<u8> = values
            .chunks(per_byte)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0_u8, |byte, (index, value)| {
                    let encoded = if bits == 8 {
                        *value as i8 as u8
                    } else {
                        (value + offset) as u8
                    };
                    byte | (encoded << (index * bits as usize))
                })
            })
            .collect();
        let storage = ffi::from_bytes(
            &bytes,
            &[rows as i32, (columns / per_byte) as i32],
            if bits == 8 { dtype::INT8 } else { dtype::UINT8 },
        );
        (
            storage,
            values.into_iter().map(|value| value as f32).collect(),
        )
    }

    #[test]
    fn invalid_scales_and_activation_metadata_fail_before_forward() {
        let packed = ffi::zeros(&[2, 32], dtype::UINT8);
        for value in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(
                prepare(
                    &packed,
                    &ffi::full_f32(&[2, 1], value, dtype::FLOAT32),
                    2,
                    128,
                    true
                )
                .is_err()
            );
            assert!(validate_activation(&ffi::full_f32(&[], value, dtype::FLOAT32)).is_err());
        }
        assert!(validate_activation(&ffi::zeros(&[1], dtype::FLOAT32)).is_err());
        assert!(validate_activation(&ffi::zeros(&[], dtype::INT32)).is_err());
    }

    #[test]
    fn native_layout_overrides_root_defaults_and_disables_fusion()
    -> Result<(), Box<dyn std::error::Error>> {
        let (packed, _) = packed_values(2, 32, 256);
        let scales = ffi::ones(&[32, 1], dtype::BFLOAT16);
        let mut weights = WeightMap::new();
        insert(
            &mut weights,
            "attention.q_proj",
            prepare(&packed, &scales, 32, 256, true)?,
        );
        assert_eq!(
            parameters(&weights, "attention.q_proj", 64, 4, "affine")?,
            (128, 2)
        );
        let layer = UnifiedLinear::from_weights(&weights, "attention.q_proj", 64, 4)?;
        assert!(layer.quantized_weight().is_none());
        assert!(layer.as_quantized_weight().is_none());
        assert!(layer.clone_shared().quantized_weight().is_none());
        weights.insert(
            "attention.q_proj.bias".to_owned(),
            ffi::ones(&[1], dtype::BFLOAT16),
        );
        assert!(UnifiedLinear::from_weights(&weights, "attention.q_proj", 64, 4).is_err());
        weights.remove("attention.q_proj.bias");
        assert!(
            FusedQKVLinear::from_weights_separate(&weights, "attention", 64, 4, 2, 1, 128).is_err()
        );
        weights.insert(
            "attention.q_proj.gemma_mobile_layout".to_owned(),
            ffi::from_slice_i32(&[32, 256, 64, 2], &[4]),
        );
        assert!(parameters(&weights, "attention.q_proj", 64, 4, "affine").is_err());
        Ok(())
    }

    #[test]
    fn embedding_gathers_duplicate_rows_and_keeps_plain_tied_projection()
    -> Result<(), Box<dyn std::error::Error>> {
        for bits in [2, 4, 8] {
            let (packed, values) = packed_values(bits, 3, 512);
            let scales = ffi::astype(
                &ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]),
                dtype::BFLOAT16,
            );
            let mut weights = WeightMap::new();
            insert(
                &mut weights,
                "table",
                prepare(&packed, &scales, 3, 512, false)?,
            );
            let table = UnifiedEmbedding::from_weights(&weights, "table", 64, 4)?;
            let output = table.forward(&ffi::from_slice_i32(&[2, 0, 2], &[1, 3]));
            let dense: Vec<f32> = values
                .iter()
                .enumerate()
                .map(|(index, value)| value * (index / 512 * 2 + index % 512 / 256 + 1) as f32)
                .collect();
            let dense = ffi::astype(&ffi::from_slice_f32(&dense, &[3, 512]), dtype::BFLOAT16);
            let expected = ffi::reshape(
                &ffi::take(&dense, &ffi::from_slice_i32(&[2, 0, 2], &[3]), 0),
                &[1, 3, 512],
            );
            assert_eq!(ffi::array_shape(&output), [1, 3, 512]);
            assert_eq!(ffi::array_dtype(&output), dtype::BFLOAT16);
            assert!(ffi::item_bool(&ffi::allclose(&output, &expected, 0.0, 0.0)));
            let input = ffi::full_f32(&[1, 512], 0.25, dtype::BFLOAT16);
            let logits = table.as_linear(&input);
            let expected = ffi::matmul(&input, &ffi::transpose(&dense));
            assert!(ffi::item_bool(&ffi::allclose(&logits, &expected, 0.0, 0.0)));
        }
        Ok(())
    }

    fn assert_linear_batch(bits: i32, batch: usize) -> Result<(), Box<dyn std::error::Error>> {
        let (packed, values) = packed_values(bits, 32, 512);
        let scales = ffi::astype(
            &ffi::from_slice_f32(
                &(0..32)
                    .map(|row| (row + 1) as f32 / 1024.0)
                    .collect::<Vec<_>>(),
                &[32, 1],
            ),
            dtype::BFLOAT16,
        );
        let bias = ffi::astype(
            &ffi::from_slice_f32(
                &(0..32)
                    .map(|row| (row * 3 - 13) as f32 / 1024.0)
                    .collect::<Vec<_>>(),
                &[32],
            ),
            dtype::BFLOAT16,
        );
        let mut weights = WeightMap::new();
        insert(
            &mut weights,
            "projection",
            prepare(&packed, &scales, 32, 512, true)?,
        );
        weights.insert("projection.bias".to_owned(), ffi::copy(&bias));
        let layer = UnifiedLinear::from_weights(&weights, "projection", 64, 4)?;
        let data: Vec<f32> = (0..batch * 512)
            .map(|index| ((index * 13 % 73) as i32 - 36) as f32 / 37.0)
            .collect();
        let input = ffi::astype(
            &ffi::from_slice_f32(&data, &[batch as i32, 512]),
            dtype::BFLOAT16,
        );
        let accumulation = if batch < 32 {
            dtype::FLOAT32
        } else {
            dtype::BFLOAT16
        };
        let dense = ffi::multiply(
            &ffi::astype(&ffi::from_slice_f32(&values, &[32, 512]), accumulation),
            &ffi::astype(&scales, accumulation),
        );
        let expected = ffi::astype(
            &ffi::add(
                &ffi::matmul(&ffi::astype(&input, accumulation), &ffi::transpose(&dense)),
                &ffi::astype(&bias, accumulation),
            ),
            dtype::BFLOAT16,
        );
        let actual = layer.forward(&input);
        assert_eq!(ffi::array_dtype(&actual), dtype::BFLOAT16);
        assert!(ffi::item_bool(&ffi::allclose(&actual, &expected, 0.0, 0.0)));
        let batched = layer.forward(&ffi::reshape(&input, &[1, batch as i32, 512]));
        assert!(ffi::item_bool(&ffi::allclose(
            &ffi::reshape(&batched, &[batch as i32, 32]),
            &actual,
            0.0,
            0.0
        )));
        Ok(())
    }

    #[test]
    fn linear_preserves_native_31_and_32_row_dtype_and_bias_order()
    -> Result<(), Box<dyn std::error::Error>> {
        for bits in [2, 4, 8] {
            assert_linear_batch(bits, 31)?;
            assert_linear_batch(bits, 32)?;
        }
        Ok(())
    }

    #[test]
    fn linear_input_and_output_rounding_enclose_bias() -> Result<(), Box<dyn std::error::Error>> {
        for bits in [2, 4, 8] {
            let (packed, values) = packed_values(bits, 32, 128);
            let scales = ffi::full_f32(&[32, 1], 0.125, dtype::FLOAT32);
            let mut weights = WeightMap::new();
            insert(
                &mut weights,
                "projection",
                prepare(&packed, &scales, 32, 128, true)?,
            );
            let input_scale = ffi::full_f32(&[], 0.5, dtype::FLOAT32);
            let output_scale = ffi::full_f32(&[], 2.0, dtype::FLOAT32);
            let bias = ffi::from_slice_f32(
                &(0..32).map(|row| row as f32 * 0.25).collect::<Vec<_>>(),
                &[32],
            );
            weights.insert("projection.bias".to_owned(), ffi::copy(&bias));
            weights.insert(
                "projection.input_activation_scale".to_owned(),
                ffi::copy(&input_scale),
            );
            weights.insert(
                "projection.output_activation_scale".to_owned(),
                ffi::copy(&output_scale),
            );
            let layer = UnifiedLinear::from_weights(&weights, "projection", 64, 4)?;
            let input = ffi::from_slice_f32(
                &(0..384)
                    .map(|index| (index % 11 - 5) as f32 * 0.3)
                    .collect::<Vec<_>>(),
                &[3, 128],
            );
            let dense = ffi::from_slice_f32(
                &values.iter().map(|value| value * 0.125).collect::<Vec<_>>(),
                &[32, 128],
            );
            let expected = srq(
                &ffi::add(
                    &ffi::matmul(&srq(&input, &input_scale), &ffi::transpose(&dense)),
                    &bias,
                ),
                &output_scale,
            );
            assert!(ffi::item_bool(&ffi::allclose(
                &layer.forward(&input),
                &expected,
                0.0,
                0.0
            )));
        }
        Ok(())
    }

    #[test]
    fn exact_layout_preserves_group_and_mixed_bits() -> Result<(), Error> {
        for bits in [2, 4, 8] {
            let dtype = if bits == 8 { dtype::INT8 } else { dtype::UINT8 };
            let layout = Layout::new(&[2, 256 * bits / 8], &[2, 2], 2, 256, dtype)?;
            assert_eq!(
                (layout.group_size, layout.bits, layout.repeats),
                (128, bits, 1)
            );
        }
        assert_eq!(
            Layout::new(&[2, 64], &[2, 4], 2, 256, dtype::UINT8)?.group_size,
            64
        );
        Ok(())
    }

    #[test]
    fn malformed_layouts_are_rejected() {
        for (packed, scale, columns, dtype) in [
            ([2, 3], [2, 1], 12, dtype::UINT8),
            ([2, 64], [3, 1], 256, dtype::UINT8),
            ([2, 64], [2, 3], 256, dtype::UINT8),
            ([2, 256], [2, 1], 256, dtype::UINT8),
            ([2, 64], [2, 1], 255, dtype::UINT8),
        ] {
            assert!(Layout::new(&packed, &scale, 2, columns, dtype).is_err());
        }
    }

    #[test]
    fn accumulation_changes_at_32_flattened_rows() {
        assert_eq!(
            accumulation_dtype(&[1, 31, 128], dtype::BFLOAT16),
            dtype::FLOAT32
        );
        assert_eq!(
            accumulation_dtype(&[1, 32, 128], dtype::BFLOAT16),
            dtype::BFLOAT16
        );
        assert_eq!(
            accumulation_dtype(&[2, 16, 128], dtype::FLOAT16),
            dtype::FLOAT16
        );
        assert_eq!(accumulation_dtype(&[128], dtype::BFLOAT16), dtype::FLOAT32);
    }

    #[test]
    fn rounding_is_even_clipped_and_zero_scale_is_identity() {
        let input = ffi::from_slice_f32(&[-200.0, -2.5, -1.5, 0.5, 1.5, 200.0], &[6]);
        let actual = srq(&input, &ffi::full_f32(&[], 1.0, dtype::FLOAT32));
        let expected = ffi::from_slice_f32(&[-128.0, -2.0, -2.0, 0.0, 2.0, 127.0], &[6]);
        assert!(ffi::item_bool(&ffi::allclose(&actual, &expected, 0.0, 0.0)));
        let unchanged = srq(&input, &ffi::full_f32(&[], 0.0, dtype::FLOAT32));
        assert!(ffi::item_bool(&ffi::allclose(&unchanged, &input, 0.0, 0.0)));
    }

    #[test]
    fn packed_extrema_and_scale_dtype_are_preserved() -> Result<(), Error> {
        for (bits, byte, expected, dtype) in [
            (2, 0xe4, [-2.0, -1.0, 0.0, 1.0], dtype::UINT8),
            (4, 0xf0, [-8.0, 7.0, -8.0, 7.0], dtype::UINT8),
            (8, 0x80, [-128.0; 4], dtype::INT8),
        ] {
            let packed = ffi::from_bytes(
                &vec![byte; 32 * bits / 8],
                &[1, (32 * bits / 8) as i32],
                dtype,
            );
            let scale = ffi::full_f32(&[1, 1], 1.0, crate::dtype::BFLOAT16);
            let prepared = prepare(&packed, &scale, 1, 32, false)?;
            assert_eq!(ffi::array_dtype(&prepared.scales), crate::dtype::BFLOAT16);
            let decoded = unsafe {
                ffi::dequantize(
                    &prepared.weight,
                    &prepared.scales,
                    &*prepared.biases,
                    prepared.layout.group_size,
                    prepared.layout.bits,
                    "affine",
                )
            };
            let first = ffi::slice(&decoded, &[0, 0], &[1, 4]);
            let expected = ffi::from_slice_f32(&expected, &[1, 4]);
            assert!(ffi::item_bool(&ffi::allclose(&first, &expected, 0.0, 0.0)));
        }
        Ok(())
    }
}
