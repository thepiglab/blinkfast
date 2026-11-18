//! Token-id parity with HF `tokenizers` (the runtime Python model2vec uses), on built-in edge cases and, when
//! BLINKFAST_TEXTS points at a jsonl {"text": ...} file, on every text in it. Models from BLINKFAST_MODELS
//! (a directory holding blink-8M/ and blink-multilingual-64M/). Skips when the models are absent.
use blinkfast::{Scratch, Tokenizer};
use std::path::PathBuf;

const EDGE: &[&str] = &[
    "",
    "Hello, World!  How are you?!?",
    "Métformine réduit l'HbA1c — naïve café; Ångström Œuvre ß straße İstanbul ǅ",
    "野口里佳 Noguchi Rika 一个人在东京 年会費 ティアード 한국어 텍스트",
    "tab\there\nnewline\r\nzero\u{0}width\u{200b}joiner\u{200d}soft\u{ad}hyphen \u{fffd} bell\u{7}",
    "supercalifragilisticexpialidocious-antidisestablishmentarianism_pneumonoultramicroscopic",
    "[UNK] [PAD] literal special tokens [UNK]x",
    "IL-6/STAT3 (p<0.05), TNF-α, β-catenin, 5′-UTR, CD4+ T cells, Ca²⁺, μmol·L⁻¹, ≥95%",
    "😀 emoji 👩‍⚕️ and ZWJ; Ελληνικά Русский العربية हिन्दी ไทย",
    "COMBINING: e\u{301} a\u{308}\u{301} \u{1100}\u{1161}\u{11A8}",
];

fn models() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("BLINKFAST_MODELS").ok()?);
    d.exists().then_some(d)
}

fn check(model: &str, texts: &[String]) {
    let Some(root) = models() else { return };
    let mut p = root.join(model).join("tokenizer.json");
    if !p.exists() {
        p = root.join(model).join("0_StaticEmbedding").join("tokenizer.json");
    }
    if !p.exists() {
        return;
    }
    let ours = Tokenizer::from_json(&std::fs::read(&p).unwrap()).unwrap();
    let mut reference = Tokenizer::from_json(&std::fs::read(&p).unwrap()).unwrap();
    reference.fused = false;
    let mut hf = tokenizers::Tokenizer::from_file(&p).unwrap();
    hf.with_truncation(None).unwrap();
    hf.with_padding(None);
    let mut s = Scratch::default();
    let mut bad = 0;
    for t in texts {
        let a = ours.ids(t, &mut s);
        assert_eq!(a, reference.ids(t, &mut s), "fused vs reference path differ on {t:?}");
        let b = hf.encode_fast(t.as_str(), false).unwrap().get_ids().to_vec();
        if a != b {
            bad += 1;
            if bad <= 5 {
                eprintln!("{model} MISMATCH on {:?}\n ours {:?}\n hf   {:?}", t.chars().take(200).collect::<String>(), a, b);
            }
        }
    }
    assert_eq!(bad, 0, "{model}: {bad}/{} texts differ", texts.len());
    eprintln!("{model}: {} texts, ids identical", texts.len());
}

fn texts() -> Vec<String> {
    let mut v: Vec<String> = EDGE.iter().map(|s| s.to_string()).collect();
    v.push("x".repeat(250));
    v.push(format!("{}é{}", "a".repeat(150), "日本".repeat(80)));
    if let Ok(f) = std::env::var("BLINKFAST_TEXTS") {
        for l in std::fs::read_to_string(f).unwrap().lines() {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(l) {
                if let Some(t) = j["text"].as_str() {
                    v.push(t.to_string());
                }
            }
        }
    }
    v
}

#[test]
fn ids_blink_8m() {
    check("blink-8M", &texts());
}

#[test]
fn ids_blink_multilingual_64m() {
    check("blink-multilingual-64M", &texts());
}
