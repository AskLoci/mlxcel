use super::{ModelArgs, TextConfig};
use mlxcel_core::layers::gemma_mobile::{self, Prepared};
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::fmt;

#[derive(Debug)]
pub(crate) enum Error {
    InvalidConfig,
    Quantization(String),
    UnsupportedModule(String),
    MissingTensor(String),
    InvalidTensor(String),
    Tensor(String, gemma_mobile::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => {
                formatter.write_str("invalid or unsupported Gemma mobile configuration")
            }
            Self::Quantization(reason) => formatter.write_str(reason),
            Self::UnsupportedModule(path) => {
                write!(formatter, "unsupported Gemma mobile module: {path}")
            }
            Self::MissingTensor(path) => write!(formatter, "missing Gemma mobile tensor: {path}"),
            Self::InvalidTensor(path) => write!(formatter, "invalid Gemma mobile tensor: {path}"),
            Self::Tensor(path, error) => write!(formatter, "{error}: {path}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn uses(config: &Value) -> bool {
    config
        .pointer("/quantization_config/quant_method")
        .and_then(Value::as_str)
        == Some("gemma")
}

pub(crate) fn validate(config: &Value) -> Result<(), Error> {
    if !uses(config) {
        return Err(Error::InvalidConfig);
    }
    validate_declaration(
        config
            .get("quantization_config")
            .ok_or(Error::InvalidConfig)?,
    )?;
    if config.get("model_type").and_then(Value::as_str) != Some("gemma4")
        || config.get("quantization").is_some()
        || config.pointer("/text_config/quantization").is_some()
        || config.pointer("/text_config/quantization_config").is_some()
        || config
            .pointer("/quantization_config/quantize_embeddings")
            .and_then(Value::as_bool)
            != Some(true)
        || !matches!(
            config
                .pointer("/quantization_config/num_bits")
                .and_then(Value::as_u64),
            Some(2 | 4 | 8)
        )
    {
        return Err(Error::InvalidConfig);
    }
    let text: TextConfig = serde_json::from_value(
        config
            .get("text_config")
            .cloned()
            .ok_or(Error::InvalidConfig)?,
    )
    .map_err(|_| Error::InvalidConfig)?;
    validate_text(&text)?;
    tied(config)?;
    Ok(())
}

fn validate_declaration(value: &Value) -> Result<(), Error> {
    let object = value.as_object().ok_or(Error::InvalidConfig)?;
    if object.keys().any(|key| {
        ![
            "quant_method",
            "num_bits",
            "quantize_embeddings",
            "module_quant_configs",
            "modules_to_not_convert",
        ]
        .contains(&key.as_str())
    }) {
        return Err(Error::InvalidConfig);
    }
    let rules = object
        .get("module_quant_configs")
        .and_then(Value::as_object)
        .ok_or(Error::InvalidConfig)?;
    if rules.len() > 1024 {
        return Err(Error::InvalidConfig);
    }
    for (pattern, rule) in rules {
        let rule = rule.as_object().ok_or(Error::InvalidConfig)?;
        if pattern.is_empty()
            || pattern.len() > 512
            || rule.len() != 1
            || !matches!(
                rule.get("num_bits").and_then(Value::as_u64),
                Some(2 | 4 | 8)
            )
        {
            return Err(Error::InvalidConfig);
        }
    }
    let excluded = object
        .get("modules_to_not_convert")
        .and_then(Value::as_array)
        .ok_or(Error::InvalidConfig)?;
    if excluded.len() > 1024
        || excluded.iter().any(|value| {
            value
                .as_str()
                .is_none_or(|value| value.is_empty() || value.len() > 512)
        })
    {
        return Err(Error::InvalidConfig);
    }
    Ok(())
}

fn validate_text(text: &TextConfig) -> Result<(), Error> {
    if text.num_hidden_layers == 0
        || text.num_hidden_layers > 4096
        || text.num_kv_shared_layers >= text.num_hidden_layers
        || text.layer_types.len() != text.num_hidden_layers
        || text.enable_moe_block
        || text
            .layer_types
            .iter()
            .any(|kind| kind != "sliding_attention" && kind != "full_attention")
    {
        return Err(Error::InvalidConfig);
    }
    for value in [
        text.hidden_size,
        text.intermediate_size,
        text.num_attention_heads,
        text.num_key_value_heads,
        text.head_dim,
        text.global_head_dim.unwrap_or(text.head_dim),
        text.num_global_key_value_heads
            .unwrap_or(text.num_key_value_heads),
        text.vocab_size,
    ] {
        dimension(value)?;
    }
    product(
        text.num_attention_heads,
        text.global_head_dim.unwrap_or(text.head_dim),
    )?;
    product(text.num_attention_heads, text.head_dim)?;
    product(text.intermediate_size, 2)?;
    if text.hidden_size_per_layer_input > 0 {
        product(text.hidden_size_per_layer_input, text.num_hidden_layers)?;
        dimension(text.vocab_size_per_layer_input)?;
    }
    Ok(())
}

fn dimension(value: usize) -> Result<i32, Error> {
    i32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or(Error::InvalidConfig)
}

fn product(left: usize, right: usize) -> Result<i32, Error> {
    dimension(left.checked_mul(right).ok_or(Error::InvalidConfig)?)
}

pub(crate) fn tied(config: &Value) -> Result<bool, Error> {
    for value in [
        config.get("tie_word_embeddings"),
        config.pointer("/text_config/tie_word_embeddings"),
    ]
    .into_iter()
    .flatten()
    {
        if !value.is_boolean() {
            return Err(Error::InvalidConfig);
        }
    }
    let text = config
        .pointer("/text_config/tie_word_embeddings")
        .and_then(Value::as_bool);
    let root = config.get("tie_word_embeddings").and_then(Value::as_bool);
    if root.zip(text).is_some_and(|(root, text)| root != text) {
        return Err(Error::InvalidConfig);
    }
    root.or(text).ok_or(Error::InvalidConfig)
}

fn layer_path(path: &str) -> Result<(usize, &str), Error> {
    let (index, component) = path
        .split_once('.')
        .ok_or_else(|| Error::UnsupportedModule(path.to_owned()))?;
    let index = index
        .parse()
        .map_err(|_| Error::UnsupportedModule(path.to_owned()))?;
    Ok((index, component))
}

fn text_shape(text: &TextConfig, path: &str, embedding: bool) -> Result<(i32, i32), Error> {
    let hidden = dimension(text.hidden_size)?;
    match (path, embedding) {
        ("language_model.model.embed_tokens", true) => {
            return Ok((dimension(text.vocab_size)?, hidden));
        }
        ("language_model.model.embed_tokens_per_layer", true) => {
            return Ok((
                dimension(text.vocab_size_per_layer_input)?,
                product(text.num_hidden_layers, text.hidden_size_per_layer_input)?,
            ));
        }
        ("language_model.lm_head", false) => return Ok((dimension(text.vocab_size)?, hidden)),
        ("language_model.model.per_layer_model_projection", false) => {
            return Ok((
                product(text.num_hidden_layers, text.hidden_size_per_layer_input)?,
                hidden,
            ));
        }
        _ => {}
    }
    if embedding {
        return Err(Error::UnsupportedModule(path.to_owned()));
    }
    let rest = path
        .strip_prefix("language_model.model.layers.")
        .ok_or_else(|| Error::UnsupportedModule(path.to_owned()))?;
    let (index, component) = layer_path(rest)?;
    if index >= text.num_hidden_layers {
        return Err(Error::UnsupportedModule(path.to_owned()));
    }
    let heads = product(
        text.num_attention_heads,
        usize::try_from(text.head_dim_for_layer(index)).map_err(|_| Error::InvalidConfig)?,
    )?;
    let kv = text
        .num_kv_heads_for_layer(index)
        .checked_mul(text.head_dim_for_layer(index))
        .ok_or(Error::InvalidConfig)?;
    let intermediate = dimension(text.mlp_intermediate_size(index))?;
    match component {
        "self_attn.q_proj" => Ok((heads, hidden)),
        "self_attn.k_proj" | "self_attn.v_proj" if !text.is_kv_shared_layer(index) => {
            Ok((kv, hidden))
        }
        "self_attn.o_proj" => Ok((hidden, heads)),
        "mlp.gate_proj" | "mlp.up_proj" => Ok((intermediate, hidden)),
        "mlp.down_proj" => Ok((hidden, intermediate)),
        "per_layer_input_gate" => Ok((dimension(text.hidden_size_per_layer_input)?, hidden)),
        "per_layer_projection" => Ok((hidden, dimension(text.hidden_size_per_layer_input)?)),
        _ => Err(Error::UnsupportedModule(path.to_owned())),
    }
}

fn vision_shape(config: &Value, path: &str) -> Result<(i32, i32), Error> {
    let vision = config.get("vision_config").ok_or(Error::InvalidConfig)?;
    let value = |key| {
        vision
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(Error::InvalidConfig)
    };
    let hidden = dimension(value("hidden_size")?)?;
    let rest = path
        .strip_prefix("vision_tower.encoder.layers.")
        .ok_or_else(|| Error::UnsupportedModule(path.to_owned()))?;
    let (index, component) = layer_path(rest)?;
    if index >= value("num_hidden_layers")? {
        return Err(Error::UnsupportedModule(path.to_owned()));
    }
    let heads = product(value("num_attention_heads")?, value("head_dim")?)?;
    let kv = product(value("num_key_value_heads")?, value("head_dim")?)?;
    let intermediate = dimension(value("intermediate_size")?)?;
    match component {
        "self_attn.q_proj.linear" => Ok((heads, hidden)),
        "self_attn.k_proj.linear" | "self_attn.v_proj.linear" => Ok((kv, hidden)),
        "self_attn.o_proj.linear" => Ok((hidden, heads)),
        "mlp.gate_proj.linear" | "mlp.up_proj.linear" => Ok((intermediate, hidden)),
        "mlp.down_proj.linear" => Ok((hidden, intermediate)),
        _ => Err(Error::UnsupportedModule(path.to_owned())),
    }
}

fn ignored(path: &str, is_tied: bool) -> bool {
    path.starts_with("audio_tower.")
        || path.starts_with("embed_audio.")
        || (is_tied && path.starts_with("language_model.lm_head."))
}

fn prepare_one(
    weights: &WeightMap,
    config: &Value,
    text: &TextConfig,
    key: &str,
) -> Result<(String, String, Prepared), Error> {
    let embedding = key.ends_with(".embedding_scale");
    let suffix = if embedding {
        ".embedding_scale"
    } else {
        ".weight_scale"
    };
    let path = key
        .strip_suffix(suffix)
        .ok_or_else(|| Error::InvalidTensor(key.to_owned()))?;
    let weight_key = format!(
        "{path}.{}",
        if embedding {
            "embedding_quantized"
        } else {
            "weight"
        }
    );
    let packed = weights
        .get(&weight_key)
        .ok_or_else(|| Error::MissingTensor(weight_key.clone()))?;
    let scale = weights
        .get(key)
        .ok_or_else(|| Error::MissingTensor(key.to_owned()))?;
    let (rows, columns) = if path.starts_with("language_model.") {
        text_shape(text, path, embedding)?
    } else if !embedding {
        vision_shape(config, path)?
    } else {
        return Err(Error::UnsupportedModule(path.to_owned()));
    };
    for suffix in ["scales", "biases", "gemma_mobile_layout", "bias"] {
        if weights.contains_key(&format!("{path}.{suffix}")) {
            return Err(Error::InvalidTensor(format!("{path}.{suffix}")));
        }
    }
    if !embedding {
        for suffix in ["input_activation_scale", "output_activation_scale"] {
            if let Some(value) = weights.get(&format!("{path}.{suffix}")) {
                gemma_mobile::validate_activation(value)
                    .map_err(|error| Error::Tensor(path.to_owned(), error))?;
            }
        }
    }
    let prepared = gemma_mobile::prepare(packed, scale, rows, columns, !embedding)
        .map_err(|error| Error::Tensor(path.to_owned(), error))?;
    Ok((path.to_owned(), weight_key, prepared))
}

pub(crate) fn prepare(weights: &mut WeightMap, config: &Value) -> Result<(), Error> {
    if !uses(config) {
        return Ok(());
    }
    validate(config)?;
    let text: TextConfig =
        serde_json::from_value(config["text_config"].clone()).map_err(|_| Error::InvalidConfig)?;
    let is_tied = tied(config)?;
    let mut keys: Vec<_> = weights
        .keys()
        .filter(|key| {
            !ignored(key, is_tied)
                && (key.ends_with(".weight_scale") || key.ends_with(".embedding_scale"))
        })
        .cloned()
        .collect();
    keys.sort();
    if keys.is_empty() {
        return Err(Error::MissingTensor("weight_scale".to_owned()));
    }
    let plans: Vec<_> = keys
        .iter()
        .map(|key| prepare_one(weights, config, &text, key))
        .collect::<Result<_, _>>()?;
    for (key, value) in weights.iter().filter(|(key, _)| !ignored(key, is_tied)) {
        if [
            ".scales",
            ".biases",
            ".gemma_mobile_layout",
            ".global_scale",
            ".weight_scale_2",
            ".input_scale",
        ]
        .iter()
        .any(|suffix| key.ends_with(suffix))
        {
            return Err(Error::InvalidTensor(key.clone()));
        }
        let packed = matches!(
            mlxcel_core::array_dtype(value),
            mlxcel_core::dtype::UINT8 | mlxcel_core::dtype::INT8
        );
        if packed && !plans.iter().any(|(_, weight_key, _)| weight_key == key) {
            return Err(Error::InvalidTensor(key.clone()));
        }
        if let Some(path) = key
            .strip_suffix(".input_activation_scale")
            .or_else(|| key.strip_suffix(".output_activation_scale"))
            && !plans.iter().any(|(candidate, weight_key, _)| {
                candidate == path && weight_key.ends_with(".weight")
            })
        {
            return Err(Error::InvalidTensor(key.clone()));
        }
    }
    weights.retain(|key, _| !ignored(key, is_tied));
    for ((path, weight_key, prepared), scale_key) in plans.into_iter().zip(keys) {
        weights.remove(&weight_key);
        weights.remove(&scale_key);
        gemma_mobile::insert(weights, &path, prepared);
    }
    Ok(())
}

pub(crate) fn load_head(
    weights: &WeightMap,
    args: &ModelArgs,
) -> Result<Option<mlxcel_core::layers::UnifiedLinear>, String> {
    let Some(quantization) = args.quantization_config.as_ref() else {
        return Ok(None);
    };
    if quantization.get("quant_method").and_then(Value::as_str) != Some("gemma") {
        return Ok(None);
    }
    let config = serde_json::json!({"model_type": args.model_type, "text_config": args.text_config, "quantization_config": quantization});
    validate(&config).map_err(|error| error.to_string())?;
    if !gemma_mobile::is_mobile(weights, "language_model.model.embed_tokens") {
        return Err("Gemma mobile weights have not passed the mobile loader".to_owned());
    }
    if tied(&config).map_err(|error| error.to_string())? {
        return Ok(None);
    }
    if !gemma_mobile::is_mobile(weights, "language_model.lm_head") {
        return Err("Gemma mobile untied output head is missing".to_owned());
    }
    mlxcel_core::layers::UnifiedLinear::from_weights(weights, "language_model.lm_head", 64, 4)
        .map(Some)
}

pub(crate) fn fused_qkv_allowed(weights: &WeightMap, prefix: &str, requested: bool) -> bool {
    requested
        && !["q_proj", "k_proj", "v_proj"]
            .into_iter()
            .any(|name| gemma_mobile::is_mobile(weights, &format!("{prefix}.{name}")))
}

#[cfg(test)]
mod tests {
    use super::super::ModelArgs;
    use super::{Error, fused_qkv_allowed, load_head, prepare, text_shape, tied, validate};
    use mlxcel_core::weights::WeightMap;

    fn config() -> serde_json::Value {
        serde_json::json!({"model_type":"gemma4","quantization_config":{"quant_method":"gemma","num_bits":4,"quantize_embeddings":true,"module_quant_configs":{},"modules_to_not_convert":[]},"text_config":{"model_type":"gemma4_text","hidden_size":128,"num_hidden_layers":2,"intermediate_size":256,"num_attention_heads":2,"head_dim":64,"rms_norm_eps":0.000001,"vocab_size":32,"num_key_value_heads":1,"rope_parameters":{},"sliding_window":64,"max_position_embeddings":256,"layer_types":["sliding_attention","full_attention"],"num_kv_shared_layers":1,"use_double_wide_mlp":true,"tie_word_embeddings":false},"vision_config":{"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64}})
    }

    fn add_packed(
        weights: &mut WeightMap,
        prefix: &str,
        rows: i32,
        columns: i32,
        bits: i32,
        embedding: bool,
    ) {
        let suffix = if embedding {
            "embedding_quantized"
        } else {
            "weight"
        };
        let dtype = if bits == 8 {
            mlxcel_core::dtype::INT8
        } else {
            mlxcel_core::dtype::UINT8
        };
        weights.insert(
            format!("{prefix}.{suffix}"),
            mlxcel_core::zeros(&[rows, columns * bits / 8], dtype),
        );
        weights.insert(
            format!(
                "{prefix}.{}",
                if embedding {
                    "embedding_scale"
                } else {
                    "weight_scale"
                }
            ),
            mlxcel_core::ones(&[rows, 1], mlxcel_core::dtype::BFLOAT16),
        );
    }

    #[test]
    fn public_preflight_validates_mobile_architecture_and_declaration() {
        let valid = config();
        assert!(super::super::validate_quantization_scheme(&valid).is_ok());
        let mut invalid = valid.clone();
        invalid["text_config"]["layer_types"] = serde_json::json!([]);
        assert!(validate(&invalid).is_err());
        invalid = valid.clone();
        invalid["quantization_config"]["mode"] = serde_json::json!("nvfp4");
        assert!(super::super::validate_quantization_scheme(&invalid).is_err());
        invalid = valid.clone();
        invalid["quantization_config"]["module_quant_configs"] =
            serde_json::json!({".*":{"num_bits":3}});
        assert!(validate(&invalid).is_err());
        invalid = valid;
        invalid["text_config"]["hidden_size"] = serde_json::json!(u64::MAX);
        assert!(validate(&invalid).is_err());
    }

    #[test]
    fn loader_prepares_text_vision_and_untied_output_without_changing_dense_siblings()
    -> Result<(), Box<dyn std::error::Error>> {
        let config = config();
        let mut weights = WeightMap::new();
        add_packed(
            &mut weights,
            "language_model.model.embed_tokens",
            32,
            128,
            2,
            true,
        );
        add_packed(&mut weights, "language_model.lm_head", 32, 128, 2, false);
        add_packed(
            &mut weights,
            "language_model.model.layers.0.self_attn.q_proj",
            128,
            128,
            4,
            false,
        );
        add_packed(
            &mut weights,
            "vision_tower.encoder.layers.0.self_attn.q_proj.linear",
            128,
            128,
            8,
            false,
        );
        weights.insert(
            "embed_vision.embedding_projection.weight".to_owned(),
            mlxcel_core::ones(&[128, 128], mlxcel_core::dtype::BFLOAT16),
        );
        weights.insert(
            "audio_tower.unsupported.weight".to_owned(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::UINT8),
        );
        prepare(&mut weights, &config)?;
        assert!(!weights.contains_key("audio_tower.unsupported.weight"));
        assert!(
            !weights
                .keys()
                .any(|key| key.ends_with(".weight_scale") || key.ends_with(".embedding_scale"))
        );
        assert_eq!(
            weights
                .get("embed_vision.embedding_projection.weight")
                .map(|value| mlxcel_core::array_dtype(value)),
            Some(mlxcel_core::dtype::BFLOAT16)
        );
        let args: ModelArgs = serde_json::from_value(config)?;
        let head = load_head(&weights, &args)?.ok_or("untied head missing")?;
        assert!(head.quantized_weight().is_none());
        assert_eq!(
            mlxcel_core::array_shape(
                &head.forward(&mlxcel_core::ones(&[1, 128], mlxcel_core::dtype::BFLOAT16))
            ),
            [1, 32]
        );
        weights.remove("language_model.lm_head.gemma_mobile_layout");
        assert!(load_head(&weights, &args).is_err());
        Ok(())
    }

    #[test]
    fn tied_output_uses_embedding_and_discards_unused_head()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut config = config();
        config["text_config"]["tie_word_embeddings"] = serde_json::json!(true);
        let mut weights = WeightMap::new();
        add_packed(
            &mut weights,
            "language_model.model.embed_tokens",
            32,
            128,
            2,
            true,
        );
        weights.insert(
            "language_model.lm_head.weight_scale".to_owned(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::INT32),
        );
        prepare(&mut weights, &config)?;
        assert!(
            !weights
                .keys()
                .any(|key| key.starts_with("language_model.lm_head."))
        );
        let args = serde_json::from_value(config)?;
        assert!(load_head(&weights, &args)?.is_none());
        Ok(())
    }

    #[test]
    fn preparation_and_head_loading_preserve_root_and_nested_tying()
    -> Result<(), Box<dyn std::error::Error>> {
        for is_tied in [false, true] {
            for (root, nested) in [(true, false), (false, true), (true, true)] {
                let mut config = config();
                if root {
                    config["tie_word_embeddings"] = serde_json::json!(is_tied);
                }
                let text = config
                    .get_mut("text_config")
                    .and_then(serde_json::Value::as_object_mut)
                    .ok_or(Error::InvalidConfig)?;
                if nested {
                    text.insert("tie_word_embeddings".to_owned(), serde_json::json!(is_tied));
                } else {
                    text.remove("tie_word_embeddings");
                }
                let mut weights = WeightMap::new();
                add_packed(
                    &mut weights,
                    "language_model.model.embed_tokens",
                    32,
                    128,
                    2,
                    true,
                );
                add_packed(&mut weights, "language_model.lm_head", 32, 128, 2, false);
                prepare(&mut weights, &config)?;
                let args: ModelArgs = serde_json::from_value(config)?;
                assert_eq!(
                    args.text_config
                        .get("tie_word_embeddings")
                        .and_then(serde_json::Value::as_bool),
                    Some(is_tied)
                );
                assert_eq!(load_head(&weights, &args)?.is_none(), is_tied);
            }
        }
        Ok(())
    }

    #[test]
    fn malformed_or_conflicting_tying_fails_at_both_configuration_boundaries() {
        for root in [
            serde_json::json!(true),
            serde_json::Value::Null,
            serde_json::json!("false"),
        ] {
            let mut config = config();
            config["tie_word_embeddings"] = root;
            assert!(validate(&config).is_err());
            assert!(serde_json::from_value::<ModelArgs>(config).is_err());
        }
        for nested in [
            serde_json::Value::Null,
            serde_json::json!("false"),
            serde_json::json!(0),
        ] {
            let mut config = config();
            config["tie_word_embeddings"] = serde_json::json!(true);
            config["text_config"]["tie_word_embeddings"] = nested;
            assert!(validate(&config).is_err());
            assert!(serde_json::from_value::<ModelArgs>(config).is_err());
        }
    }

    #[test]
    fn invalid_late_module_leaves_source_tensor_map_unchanged() {
        let mut weights = WeightMap::new();
        add_packed(
            &mut weights,
            "language_model.model.embed_tokens",
            32,
            128,
            2,
            true,
        );
        add_packed(
            &mut weights,
            "vision_tower.encoder.layers.0.unrecognized.linear",
            128,
            128,
            8,
            false,
        );
        let keys: std::collections::HashSet<_> = weights.keys().cloned().collect();
        assert!(prepare(&mut weights, &config()).is_err());
        assert_eq!(
            weights
                .keys()
                .cloned()
                .collect::<std::collections::HashSet<_>>(),
            keys
        );
        assert_eq!(
            weights
                .get("language_model.model.embed_tokens.embedding_quantized")
                .map(|value| mlxcel_core::array_dtype(value)),
            Some(mlxcel_core::dtype::UINT8)
        );
    }

    #[test]
    fn mixed_foreign_planes_and_orphan_scales_are_rejected() {
        for key in [
            "language_model.model.layers.0.mlp.gate_proj.scales",
            "language_model.model.layers.0.mlp.gate_proj.input_activation_scale",
            "language_model.model.embed_tokens.input_activation_scale",
            "language_model.model.layers.0.mlp.gate_proj.weight_scale_2",
        ] {
            let mut weights = WeightMap::new();
            add_packed(
                &mut weights,
                "language_model.model.embed_tokens",
                32,
                128,
                2,
                true,
            );
            weights.insert(
                key.to_owned(),
                mlxcel_core::ones(&[], mlxcel_core::dtype::FLOAT32),
            );
            assert!(prepare(&mut weights, &config()).is_err());
        }
    }

    #[test]
    fn explicit_fusion_request_does_not_bypass_mobile_rounding() {
        let mut weights = WeightMap::new();
        assert!(fused_qkv_allowed(&weights, "attention", true));
        weights.insert(
            "attention.k_proj.gemma_mobile_layout".to_owned(),
            mlxcel_core::from_slice_i32(&[128, 128, 128, 4], &[4]),
        );
        assert!(!fused_qkv_allowed(&weights, "attention", true));
        assert!(!fused_qkv_allowed(&weights, "attention", false));
    }

    #[test]
    fn output_projection_requires_consistent_tying_metadata() -> Result<(), Error> {
        assert!(!tied(
            &serde_json::json!({"tie_word_embeddings":false,"text_config":{"tie_word_embeddings":false}})
        )?);
        assert!(tied(
            &serde_json::json!({"text_config":{"tie_word_embeddings":true}})
        )?);
        assert!(tied(&serde_json::json!({"tie_word_embeddings":true,"text_config":{"tie_word_embeddings":false}})).is_err());
        assert!(tied(&serde_json::json!({})).is_err());
        Ok(())
    }

    #[test]
    fn module_shapes_reject_unknown_and_out_of_range_layers()
    -> Result<(), Box<dyn std::error::Error>> {
        let value = serde_json::json!({"model_type":"gemma4_text","hidden_size":128,"num_hidden_layers":2,"intermediate_size":256,"num_attention_heads":2,"head_dim":64,"rms_norm_eps":0.000001,"vocab_size":32,"num_key_value_heads":1,"rope_parameters":{},"sliding_window":64,"max_position_embeddings":256,"layer_types":["sliding_attention","full_attention"],"num_kv_shared_layers":1,"use_double_wide_mlp":true});
        let config = serde_json::from_value(value)?;
        assert_eq!(
            text_shape(
                &config,
                "language_model.model.layers.1.mlp.down_proj",
                false
            )?,
            (128, 512)
        );
        assert!(
            text_shape(
                &config,
                "language_model.model.layers.2.self_attn.q_proj",
                false
            )
            .is_err()
        );
        assert!(
            text_shape(
                &config,
                "language_model.model.layers.1.self_attn.k_proj",
                false
            )
            .is_err()
        );
        assert!(text_shape(&config, "language_model.model.norm", false).is_err());
        Ok(())
    }
}
