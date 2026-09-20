//! Model registry and local cache resolution.
//!
//! This crate never downloads model weights and never embeds them. A later
//! fetcher can use the resolved metadata while keeping acquisition policy out
//! of the inference core.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, path::PathBuf};
use thiserror::Error;

const EMBEDDED_REGISTRY: &str = include_str!("../../../models/models.toml");

#[derive(Debug, Clone, Deserialize)]
pub struct Registry {
    pub models: BTreeMap<String, ModelSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub display_name: String,
    pub provider: String,
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub tokenizer_file: String,
    pub license: String,
    pub status: String,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    pub runtime: RuntimeSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RuntimeSpec {
    pub backend: String,
    pub recommended_gpu_layers: u32,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub id: String,
    pub spec: ModelSpec,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LocalVerification {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub model_sha256: Option<String>,
    pub tokenizer_sha256: Option<String>,
    pub verified: bool,
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed to parse embedded model registry: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("unknown model id: {0}")]
    UnknownModel(String),
    #[error("failed to read model file {0}: {1}")]
    Read(PathBuf, String),
}

impl Registry {
    pub fn embedded() -> Result<Self, RegistryError> {
        Ok(toml::from_str(EMBEDDED_REGISTRY)?)
    }

    pub fn get(&self, id: &str) -> Result<&ModelSpec, RegistryError> {
        self.models
            .get(id)
            .ok_or_else(|| RegistryError::UnknownModel(id.into()))
    }

    pub fn resolve(&self, id: &str) -> Result<ResolvedModel, RegistryError> {
        let spec = self.get(id)?.clone();
        let root = cache_root();
        let model_dir = root.join(id).join(&spec.revision);
        Ok(ResolvedModel {
            id: id.into(),
            model_path: model_dir.join(&spec.file),
            tokenizer_path: model_dir.join(&spec.tokenizer_file),
            spec,
        })
    }

    pub fn verify_local(&self, id: &str) -> Result<LocalVerification, RegistryError> {
        let resolved = self.resolve(id)?;
        let model_sha256 = file_sha256(&resolved.model_path)?;
        let tokenizer_sha256 = file_sha256(&resolved.tokenizer_path)?;
        let verified = resolved
            .spec
            .sha256
            .as_deref()
            .is_some_and(|expected| model_sha256.as_deref() == Some(expected))
            && resolved
                .spec
                .tokenizer_sha256
                .as_deref()
                .is_some_and(|expected| tokenizer_sha256.as_deref() == Some(expected));
        Ok(LocalVerification {
            model_path: resolved.model_path,
            tokenizer_path: resolved.tokenizer_path,
            model_sha256,
            tokenizer_sha256,
            verified,
        })
    }
}

fn file_sha256(path: &PathBuf) -> Result<Option<String>, RegistryError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(RegistryError::Read(path.clone(), error.to_string())),
    };
    let digest = Sha256::digest(bytes);
    Ok(Some(format!("{digest:x}")))
}

pub fn cache_root() -> PathBuf {
    if let Some(path) = env::var_os("GANACHE_MODEL_DIR") {
        return PathBuf::from(path);
    }
    if let Some(path) = env::var_os("GANACHE_REPO_ROOT") {
        return PathBuf::from(path).join("models").join("cache");
    }
    PathBuf::from("models").join("cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_contains_baseline() {
        let registry = Registry::embedded().unwrap();
        let model = registry.get("jinen_v1_xsmall_q5").unwrap();
        assert_eq!(model.provider, "huggingface");
        assert_eq!(model.file, "jinen-v1-xsmall-Q5_K_M.gguf");
    }

    #[test]
    fn resolve_uses_revision_and_cache_root() {
        let registry = Registry::embedded().unwrap();
        let resolved = registry.resolve("jinen_v1_xsmall_q5").unwrap();
        assert!(resolved.model_path.ends_with("jinen-v1-xsmall-Q5_K_M.gguf"));
        assert!(resolved.tokenizer_path.ends_with("tokenizer.json"));
        assert!(resolved.model_path.to_string_lossy().contains("main"));
    }
}
