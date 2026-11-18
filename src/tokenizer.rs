//! A WordPiece tokenizer specialised for the pipeline the Blink models ship in `tokenizer.json`:
//!
//! 1. added tokens with `normalized: false` are split out of the raw text first (leftmost-longest, as HF
//!    `tokenizers`' `AddedVocabulary`); Blink-multilingual has 30,000 CJK words plus `[PAD]`/`[UNK]`;
//! 2. `BertNormalizer` (clean_text, handle_chinese_chars, strip_accents, lowercase) on each remaining piece;
//! 3. `BertPreTokenizer` (split on whitespace, isolate punctuation), then optionally `FixedLength(n)`;
//! 4. `WordPiece` greedy longest-match-first with the `##` continuation prefix and `max_input_chars_per_word`.
//!
//! It emits the same ids as `tokenizers` 0.23 for these pipelines (tests/parity.rs), without building
//! `NormalizedString`s, offsets or per-word `String`s. Pure-ASCII pieces take a single-pass fast path.
//! Any pipeline outside this shape is rejected at load time instead of being approximated.
use anyhow::{bail, Context, Result};
use foldhash::fast::FixedState;
use hashbrown::HashMap;
use serde_json::Value;
use unicode_categories::UnicodeCategories;
use unicode_normalization::char::decompose_canonical;

type Map = HashMap<Box<str>, u32, FixedState>;

pub struct Tokenizer {
    /// word-initial pieces
    head: Map,
    /// continuation pieces, stored WITHOUT the `##` prefix
    cont: Map,
    head_max: usize,
    cont_max: usize,
    pub unk_id: u32,
    max_chars: usize,
    fixed_len: Option<usize>,
    clean_text: bool,
    chinese: bool,
    strip_accents: bool,
    lowercase: bool,
    added: Option<(aho_corasick::AhoCorasick, Vec<u32>)>,
    /// median vocabulary token length in chars (Python model2vec's truncation unit)
    pub median_token_len: usize,
    /// model2vec-rs 0.4's unit instead: median byte length over the model vocabulary only (no added tokens),
    /// upper middle. Same as `median_token_len` for Blink-8M (6); 6 vs 4 for Blink-multilingual-64M.
    pub median_token_len_m2v_rs: usize,
    /// per ASCII byte: what the normaliser + pre-tokeniser do with it (built from the exact predicates)
    ascii: [Class; 128],
    /// Some(..) when the pipeline is outside the fast path: HF `tokenizers` does the ids (feature `generic`)
    #[cfg(feature = "generic")]
    generic: Option<tokenizers::Tokenizer>,
    /// true when the WordPiece fast path is in use
    pub fast: bool,
    /// false = the straightforward reference path (normalise the piece into a string with the exact Unicode
    /// predicates, then pre-tokenise it); used for the ablation and as a cross-check. Same ids either way.
    pub fused: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// removed by clean_text
    Drop,
    /// whitespace: ends a word
    Space,
    /// punctuation: ends a word and is a word of its own
    Punct,
    /// word byte (already lowercased when the normaliser lowercases)
    Word(u8),
}

/// Reusable per-thread buffers: no allocation per document once warm.
#[derive(Default)]
pub struct Scratch {
    norm: String,
    word: String,
    /// pooled ids of the current text (reused)
    pub ids: Vec<u32>,
}

fn is_ws(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}

fn is_chinese_char(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F
        | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

fn is_punc(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}

impl Tokenizer {
    /// Fast WordPiece path when the pipeline is one it reproduces exactly; otherwise (feature `generic`) the HF
    /// `tokenizers` pipeline, so every model2vec / StaticEmbedding model loads.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        match Self::fast_from_json(bytes) {
            Ok(t) => Ok(t),
            #[cfg(feature = "generic")]
            Err(e) => Self::generic_from_json(bytes).with_context(|| format!("fast path not applicable ({e}); generic load failed")),
            #[cfg(not(feature = "generic"))]
            Err(e) => Err(e),
        }
    }

    #[cfg(feature = "generic")]
    fn generic_from_json(bytes: &[u8]) -> Result<Self> {
        let mut t = tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        t.with_truncation(None).map_err(|e| anyhow::anyhow!("{e}"))?;
        t.with_padding(None);
        let j: Value = serde_json::from_slice(bytes)?;
        let vocab = t.get_vocab(true);
        let unk_name = j["model"]["unk_token"].as_str().map(String::from).or_else(|| {
            j["model"]["unk_id"].as_u64().and_then(|id| t.id_to_token(id as u32))
        });
        let unk_id = unk_name.and_then(|n| vocab.get(&n).copied()).unwrap_or(u32::MAX);
        let mut lens: Vec<usize> = vocab.keys().map(|k| k.chars().count()).collect();
        lens.sort_unstable();
        let n = lens.len();
        let median_token_len = if n == 0 { 1 } else if n % 2 == 1 { lens[n / 2] } else { (lens[n / 2 - 1] + lens[n / 2]) / 2 }.max(1);
        let mut bytelens: Vec<usize> = t.get_vocab(false).keys().map(|k| k.len()).collect();
        bytelens.sort_unstable();
        let median_token_len_m2v_rs = bytelens.get(bytelens.len() / 2).copied().unwrap_or(1);
        Ok(Tokenizer {
            head: Map::default(),
            cont: Map::default(),
            head_max: 0,
            cont_max: 0,
            unk_id,
            max_chars: 0,
            fixed_len: None,
            clean_text: false,
            chinese: false,
            strip_accents: false,
            lowercase: false,
            added: None,
            median_token_len,
            median_token_len_m2v_rs,
            ascii: [Class::Drop; 128],
            generic: Some(t),
            fast: false,
            fused: true,
        })
    }

    fn fast_from_json(bytes: &[u8]) -> Result<Self> {
        let j: Value = serde_json::from_slice(bytes).context("tokenizer.json")?;
        let m = &j["model"];
        if m["type"] != "WordPiece" {
            bail!("unsupported model type {} (blinkfast handles WordPiece)", m["type"]);
        }
        if m["continuing_subword_prefix"] != "##" {
            bail!("unsupported continuing_subword_prefix {}", m["continuing_subword_prefix"]);
        }
        let n = &j["normalizer"];
        if n["type"] != "BertNormalizer" {
            bail!("unsupported normalizer {}", n["type"]);
        }
        let lowercase = n["lowercase"].as_bool().unwrap_or(true);
        let strip_accents = n["strip_accents"].as_bool().unwrap_or(lowercase);
        let pre = &j["pre_tokenizer"];
        let fixed_len = match pre["type"].as_str() {
            Some("BertPreTokenizer") => None,
            Some("Sequence") => {
                let p = pre["pretokenizers"].as_array().context("pretokenizers")?;
                match p.as_slice() {
                    [a, b] if a["type"] == "BertPreTokenizer" && b["type"] == "FixedLength" => {
                        Some(b["length"].as_u64().unwrap_or(5) as usize)
                    }
                    _ => bail!("unsupported pre_tokenizer sequence {pre}"),
                }
            }
            _ => bail!("unsupported pre_tokenizer {pre}"),
        };
        if !j["post_processor"].is_null() {
            // model2vec encodes without special tokens, so a post-processor would not be applied anyway
        }
        let mut head = Map::default();
        let mut cont = Map::default();
        let mut by_id: HashMap<u32, usize, FixedState> = HashMap::default();
        let mut bytelens: Vec<usize> = m["vocab"].as_object().context("vocab")?.keys().map(|k| k.len()).collect();
        bytelens.sort_unstable();
        let median_token_len_m2v_rs = bytelens.get(bytelens.len() / 2).copied().unwrap_or(1);
        for (k, v) in m["vocab"].as_object().context("vocab")? {
            let id = v.as_u64().context("vocab id")? as u32;
            by_id.insert(id, k.chars().count());
            match k.strip_prefix("##") {
                Some(rest) => {
                    cont.insert(rest.into(), id);
                }
                None => {}
            }
            head.insert(k.as_str().into(), id);
        }
        let mut added_pats = Vec::new();
        let mut added_ids = Vec::new();
        for t in j["added_tokens"].as_array().into_iter().flatten() {
            let content = t["content"].as_str().context("added content")?;
            let id = t["id"].as_u64().context("added id")? as u32;
            by_id.insert(id, content.chars().count());
            if t["single_word"] == true || t["lstrip"] == true || t["rstrip"] == true {
                bail!("unsupported added-token flags on {content}");
            }
            if t["normalized"] == true {
                bail!("normalized added tokens are not supported ({content})");
            }
            added_pats.push(content.to_string());
            added_ids.push(id);
        }
        let added = if added_pats.is_empty() {
            None
        } else {
            let ac = aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(&added_pats)?;
            Some((ac, added_ids))
        };
        let unk = m["unk_token"].as_str().context("unk_token")?;
        let unk_id = *head.get(unk).context("unk token not in vocab")?;
        // model2vec (Python): int(np.median([len(t) for t in tokens])) over the unique vocabulary, in characters
        let mut lens: Vec<usize> = by_id.into_values().collect();
        lens.sort_unstable();
        let median_token_len = match lens.len() {
            0 => 1,
            n if n % 2 == 1 => lens[n / 2],
            n => (lens[n / 2 - 1] + lens[n / 2]) / 2, // np.median averages the middle pair; int() floors
        }
        .max(1);
        let clean_text = n["clean_text"].as_bool().unwrap_or(true);
        let mut ascii = [Class::Drop; 128];
        for b in 0u8..128 {
            let c = b as char;
            ascii[b as usize] = if clean_text && (b == 0 || is_control(c)) {
                Class::Drop
            } else if (clean_text && is_ws(c)) || c.is_whitespace() {
                Class::Space
            } else if is_punc(c) {
                Class::Punct
            } else {
                Class::Word(if lowercase { b.to_ascii_lowercase() } else { b })
            };
        }
        Ok(Tokenizer {
            #[cfg(feature = "generic")]
            generic: None,
            fast: true,
            fused: true,
            ascii,
            head_max: head.keys().map(|k| k.len()).max().unwrap_or(1),
            cont_max: cont.keys().map(|k| k.len()).max().unwrap_or(1),
            head,
            cont,
            unk_id,
            max_chars: m["max_input_chars_per_word"].as_u64().unwrap_or(100) as usize,
            fixed_len,
            clean_text,
            chinese: n["handle_chinese_chars"].as_bool().unwrap_or(true),
            strip_accents,
            lowercase,
            added,
            median_token_len,
            median_token_len_m2v_rs,
        })
    }

    /// Calls `emit(id)` for every token of `text`, in order. Returns early if `emit` returns false.
    pub fn for_each_id(&self, text: &str, s: &mut Scratch, emit: &mut impl FnMut(u32) -> bool) {
        #[cfg(feature = "generic")]
        if let Some(t) = &self.generic {
            if let Ok(enc) = t.encode_fast(text, false) {
                for &id in enc.get_ids() {
                    if !emit(id) {
                        return;
                    }
                }
            }
            return;
        }
        match &self.added {
            None => {
                self.piece(text, s, emit);
            }
            Some((ac, ids)) => {
                let mut last = 0;
                for m in ac.find_iter(text) {
                    if m.start() > last && !self.piece(&text[last..m.start()], s, emit) {
                        return;
                    }
                    if !emit(ids[m.pattern().as_usize()]) {
                        return;
                    }
                    last = m.end();
                }
                if last < text.len() {
                    self.piece(&text[last..], s, emit);
                }
            }
        }
    }

    pub fn ids(&self, text: &str, s: &mut Scratch) -> Vec<u32> {
        let mut v = Vec::new();
        self.for_each_id(text, s, &mut |id| {
            v.push(id);
            true
        });
        v
    }

    /// normalise + pre-tokenise + WordPiece one piece of raw text (no added tokens inside), in one pass.
    /// Every BertNormalizer step is a per-char map, so each raw char is normalised and immediately classified by
    /// the pre-tokeniser; words collect in a reused buffer.
    fn piece(&self, raw: &str, s: &mut Scratch, emit: &mut impl FnMut(u32) -> bool) -> bool {
        if !self.fused {
            return self.piece_reference(raw, s, emit);
        }
        let mut word = std::mem::take(&mut s.word);
        let mut norm = std::mem::take(&mut s.norm);
        word.clear();
        let ok = self.piece_inner(raw, &mut word, &mut norm, s, emit);
        s.word = word;
        s.norm = norm;
        ok
    }

    #[inline]
    fn piece_inner(&self, raw: &str, word: &mut String, norm: &mut String, s: &mut Scratch, emit: &mut impl FnMut(u32) -> bool) -> bool {
        let bytes = raw.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b < 128 {
                i += 1;
                match self.ascii[b as usize] {
                    Class::Word(l) => word.push(l as char),
                    Class::Drop => {}
                    Class::Space => {
                        if !word.is_empty() {
                            if !self.word(word, s, emit) {
                                return false;
                            }
                            word.clear();
                        }
                    }
                    Class::Punct => {
                        if !word.is_empty() {
                            if !self.word(word, s, emit) {
                                return false;
                            }
                            word.clear();
                        }
                        let p = [b];
                        if !self.word(std::str::from_utf8(&p).unwrap(), s, emit) {
                            return false;
                        }
                    }
                }
                continue;
            }
            let c = raw[i..].chars().next().unwrap();
            i += c.len_utf8();
            // normalise this char into `norm`, then pre-tokenise what it became
            norm.clear();
            if self.clean_text {
                if c as u32 == 0xFFFD || is_control(c) {
                    continue;
                }
                if is_ws(c) {
                    norm.push(' ');
                } else {
                    self.push_post_clean(c, norm);
                }
            } else {
                self.push_post_clean(c, norm);
            }
            for d in norm.chars() {
                if d.is_whitespace() {
                    if !word.is_empty() {
                        if !self.word(word, s, emit) {
                            return false;
                        }
                        word.clear();
                    }
                } else if is_punc(d) {
                    if !word.is_empty() {
                        if !self.word(word, s, emit) {
                            return false;
                        }
                        word.clear();
                    }
                    let mut buf = [0u8; 4];
                    if !self.word(d.encode_utf8(&mut buf), s, emit) {
                        return false;
                    }
                } else {
                    word.push(d);
                }
            }
        }
        if !word.is_empty() {
            return self.word(word, s, emit);
        }
        true
    }

    /// Reference path: BertNormalizer over the whole piece into a string, then BertPreTokenizer over that string.
    fn piece_reference(&self, raw: &str, s: &mut Scratch, emit: &mut impl FnMut(u32) -> bool) -> bool {
        let mut norm = std::mem::take(&mut s.norm);
        norm.clear();
        for c in raw.chars() {
            if self.clean_text {
                if c as u32 == 0 || c as u32 == 0xFFFD || is_control(c) {
                    continue;
                }
                if is_ws(c) {
                    norm.push(' ');
                    continue;
                }
            }
            self.push_post_clean(c, &mut norm);
        }
        let mut ok = true;
        let mut start: Option<usize> = None;
        for (i, c) in norm.char_indices() {
            let ws = c.is_whitespace();
            let p = !ws && is_punc(c);
            if ws || p {
                if let Some(st) = start.take() {
                    if !self.word(&norm[st..i], s, emit) {
                        ok = false;
                        break;
                    }
                }
                if p && !self.word(&norm[i..i + c.len_utf8()], s, emit) {
                    ok = false;
                    break;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if ok {
            if let Some(st) = start {
                ok = self.word(&norm[st..], s, emit);
            }
        }
        s.norm = norm;
        ok
    }

    /// handle_chinese_chars, strip_accents (NFD minus Mn), lowercase (per char), applied to one cleaned char.
    #[inline]
    fn push_post_clean(&self, c: char, out: &mut String) {
        let chinese = self.chinese && is_chinese_char(c);
        if chinese {
            out.push(' ');
        }
        if self.strip_accents {
            decompose_canonical(c, |d| {
                if !d.is_mark_nonspacing() {
                    self.push_lower(d, out);
                }
            });
        } else {
            self.push_lower(c, out);
        }
        if chinese {
            out.push(' ');
        }
    }

    #[inline]
    fn push_lower(&self, c: char, out: &mut String) {
        if self.lowercase {
            for l in c.to_lowercase() {
                out.push(l);
            }
        } else {
            out.push(c);
        }
    }

    /// FixedLength split (if any), then WordPiece.
    fn word(&self, w: &str, s: &mut Scratch, emit: &mut impl FnMut(u32) -> bool) -> bool {
        match self.fixed_len {
            Some(n) if w.len() > n => {
                // byte length > n is necessary for char length > n
                let mut chunk_start = 0;
                let mut k = 0;
                for (i, _) in w.char_indices() {
                    if k == n {
                        if !self.wordpiece(&w[chunk_start..i], s, emit) {
                            return false;
                        }
                        chunk_start = i;
                        k = 0;
                    }
                    k += 1;
                }
                self.wordpiece(&w[chunk_start..], s, emit)
            }
            _ => self.wordpiece(w, s, emit),
        }
    }

    fn wordpiece(&self, w: &str, s: &mut Scratch, emit: &mut impl FnMut(u32) -> bool) -> bool {
        if w.is_empty() {
            return true;
        }
        let ascii = w.is_ascii();
        let nchars = if ascii { w.len() } else { w.chars().count() };
        if nchars > self.max_chars {
            return emit(self.unk_id);
        }
        // fast path: the whole word is one vocab entry
        if w.len() <= self.head_max {
            if let Some(&id) = self.head.get(w) {
                return emit(id);
            }
        }
        // collect pieces first: a failure anywhere makes the whole word [UNK]
        let mut ids: [u32; 128] = [0; 128];
        let mut n = 0usize;
        let mut overflow: Vec<u32> = Vec::new();
        let mut start = 0;
        let _ = &mut s.word;
        while start < w.len() {
            let (map, maxl) = if start == 0 { (&self.head, self.head_max) } else { (&self.cont, self.cont_max) };
            let mut end = (start + maxl).min(w.len());
            while !w.is_char_boundary(end) {
                end -= 1;
            }
            let mut found = None;
            while end > start {
                if let Some(&id) = map.get(&w[start..end]) {
                    found = Some(id);
                    break;
                }
                end -= 1;
                if !ascii {
                    while end > start && !w.is_char_boundary(end) {
                        end -= 1;
                    }
                }
            }
            match found {
                Some(id) => {
                    if n < ids.len() {
                        ids[n] = id;
                    } else {
                        overflow.push(id);
                    }
                    n += 1;
                    start = end;
                }
                None => return emit(self.unk_id),
            }
        }
        for &id in ids[..n.min(ids.len())].iter().chain(overflow.iter()) {
            if !emit(id) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn mn_detection_matches_combining() {
        // sanity: combining acute accent is Mn and a combining mark
        let c = '\u{0301}';
        use unicode_categories::UnicodeCategories;
        assert!(c.is_mark_nonspacing());
        assert!(unicode_normalization::char::is_combining_mark(c));
    }
}
