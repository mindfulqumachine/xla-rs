use std::path::Path;
use tokenizers::Tokenizer;

pub struct GemmaTokenizer {
    tokenizer: Tokenizer,
}

impl GemmaTokenizer {
    pub fn new<P: AsRef<Path>>(path: P) -> crate::tensor::Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> crate::tensor::Result<Vec<usize>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    pub fn decode(&self, ids: &[usize]) -> crate::tensor::Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.tokenizer
            .decode(&u32_ids, true)
            .map_err(|e| crate::tensor::TensorError::Unsupported(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_tokenizer_load_encode_decode() {
        // Create a minimal tokenizer JSON
        let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "hello": 0,
      "world": 1
    },
    "unk_token": "[UNK]"
  }
}"#;
        let path = "test_tokenizer.json";
        let mut file = File::create(path).unwrap();
        file.write_all(json.as_bytes()).unwrap();

        // Test loading
        let tokenizer = GemmaTokenizer::new(path).unwrap();

        // Test encode
        // Note: Basic BPE might not handle spaces or unknown tokens well without proper config.
        // But "hello" should be 0.
        let ids = tokenizer.encode("hello").unwrap();
        assert_eq!(ids, vec![0]);

        // Test decode
        let text = tokenizer.decode(&[1]).unwrap();
        assert_eq!(text, "world");

        // Cleanup
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_tokenizer_load_error() {
        let result = GemmaTokenizer::new("non_existent_tokenizer.json");
        assert!(result.is_err());
    }
}
