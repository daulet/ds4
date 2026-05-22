pub const DS4_N_VOCAB: usize = 129_280;
pub const DS4_NEG_INF: f32 = -1.0e30;
pub const DS4_DEFAULT_TEMPERATURE: f32 = 1.0;
pub const DS4_DEFAULT_TOP_P: f32 = 1.0;
pub const DS4_DEFAULT_MIN_P: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

impl SamplingParams {
    pub const fn defaults() -> Self {
        Self {
            temperature: DS4_DEFAULT_TEMPERATURE,
            top_k: 0,
            top_p: DS4_DEFAULT_TOP_P,
            min_p: DS4_DEFAULT_MIN_P,
        }
    }

    pub fn apply_thinking_defaults(&mut self) {
        *self = Self::defaults();
    }

    pub fn apply_dsml_structural(&mut self) {
        self.temperature = 0.0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceCandidate {
    pub id: i32,
    pub logit: f32,
    pub weight: f32,
    pub normalized_prob: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingTrace {
    pub selected: i32,
    pub actual_selected: i32,
    pub matches_actual: bool,
    pub rng_before: u64,
    pub rng_after: u64,
    pub actual_rng_after: u64,
    pub greedy: bool,
    pub effective_top_k: i32,
    pub effective_top_p: f32,
    pub effective_min_p: f32,
    pub finite_count: u32,
    pub max_logit: f32,
    pub sum: f32,
    pub filtered_sum: f32,
    pub rng_unit: f32,
    pub filtered: Vec<TraceCandidate>,
}

impl SamplingTrace {
    pub fn new(seed: u64, params: SamplingParams) -> Self {
        Self {
            selected: -1,
            actual_selected: -1,
            matches_actual: false,
            rng_before: seed,
            rng_after: seed,
            actual_rng_after: seed,
            greedy: false,
            effective_top_k: params.top_k,
            effective_top_p: params.top_p,
            effective_min_p: params.min_p,
            finite_count: 0,
            max_logit: DS4_NEG_INF,
            sum: 0.0,
            filtered_sum: 0.0,
            rng_unit: 0.0,
            filtered: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenScore {
    pub id: i32,
    pub logit: f32,
    pub logprob: f32,
}

pub fn sample_argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    let mut best_v = DS4_NEG_INF;
    for (idx, &value) in logits.iter().enumerate() {
        if value > best_v {
            best_v = value;
            best = idx as i32;
        }
    }
    best
}

pub fn sample_rng_next(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9e37_79b9_7f4a_7c15;
    }
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

pub fn sample_rng_f32(state: &mut u64) -> f32 {
    let x = sample_rng_next(state);
    ((x >> 40) & 0x00ff_ffff) as f32 / 16_777_216.0
}

pub fn sample_top_p_min_p(
    logits: &[f32],
    params: SamplingParams,
    rng: &mut u64,
    trace: Option<&mut SamplingTrace>,
) -> i32 {
    sample_top_p_min_p_impl(logits, params, rng, trace)
}

fn trace_return(trace: Option<&mut SamplingTrace>, selected: i32, rng: &u64) -> i32 {
    if let Some(trace) = trace {
        trace.selected = selected;
        trace.rng_after = *rng;
    }
    selected
}

fn sample_top_p_min_p_impl(
    logits: &[f32],
    mut params: SamplingParams,
    rng: &mut u64,
    trace: Option<&mut SamplingTrace>,
) -> i32 {
    if params.temperature <= 0.0 {
        if let Some(trace) = trace {
            trace.greedy = true;
            return trace_return(Some(trace), sample_argmax(logits), rng);
        }
        return sample_argmax(logits);
    }

    if params.top_p <= 0.0 || params.top_p > 1.0 {
        params.top_p = 1.0;
    }
    if params.min_p < 0.0 {
        params.min_p = 0.0;
    }

    if params.top_k <= 0 {
        if let Some(trace) = trace {
            trace.effective_top_k = params.top_k;
            trace.effective_top_p = params.top_p;
            trace.effective_min_p = params.min_p;
            return sample_full_vocab(logits, params, rng, Some(trace));
        }
        return sample_full_vocab(logits, params, rng, None);
    }

    if params.top_k > 1024 {
        params.top_k = 1024;
    }
    if params.top_k as usize > logits.len() {
        params.top_k = logits.len() as i32;
    }
    if let Some(trace) = trace {
        trace.effective_top_k = params.top_k;
        trace.effective_top_p = params.top_p;
        trace.effective_min_p = params.min_p;
        return sample_top_k(logits, params, rng, Some(trace));
    }
    sample_top_k(logits, params, rng, None)
}

fn sample_full_vocab(
    logits: &[f32],
    params: SamplingParams,
    rng: &mut u64,
    mut trace: Option<&mut SamplingTrace>,
) -> i32 {
    let mut max_logit = DS4_NEG_INF;
    let mut best = 0;
    let mut finite = 0_u32;
    for (idx, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        finite += 1;
        if value > max_logit {
            max_logit = value;
            best = idx as i32;
        }
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.finite_count = finite;
        trace.max_logit = max_logit;
    }
    if finite == 0 {
        return trace_return(trace, sample_argmax(logits), rng);
    }

    if params.top_p >= 1.0 {
        let min_rel = if params.min_p > 0.0 {
            params.min_p
        } else {
            0.0
        };
        let mut sum = 0.0_f32;
        for (idx, &value) in logits.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let weight = ((value - max_logit) / params.temperature).exp();
            if weight < min_rel {
                continue;
            }
            sum += weight;
            if let Some(trace) = trace.as_deref_mut() {
                trace.filtered.push(TraceCandidate {
                    id: idx as i32,
                    logit: value,
                    weight,
                    normalized_prob: 0.0,
                });
            }
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.sum = sum;
            trace.filtered_sum = sum;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return trace_return(trace, best, rng);
        }
        if let Some(trace) = trace.as_deref_mut() {
            for candidate in &mut trace.filtered {
                candidate.normalized_prob = candidate.weight / sum;
            }
        }
        let rng_unit = sample_rng_f32(rng);
        if let Some(trace) = trace.as_deref_mut() {
            trace.rng_unit = rng_unit;
            trace.rng_after = *rng;
        }
        let mut draw = rng_unit * sum;
        for (idx, &value) in logits.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let weight = ((value - max_logit) / params.temperature).exp();
            if weight < min_rel {
                continue;
            }
            draw -= weight;
            if draw <= 0.0 {
                return trace_return(trace, idx as i32, rng);
            }
        }
        return trace_return(trace, best, rng);
    }

    let mut candidates: Vec<TraceCandidate> = Vec::with_capacity(finite as usize);
    let mut sum = 0.0_f32;
    for (idx, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        let weight = ((value - max_logit) / params.temperature).exp();
        candidates.push(TraceCandidate {
            id: idx as i32,
            logit: value,
            weight,
            normalized_prob: 0.0,
        });
        sum += weight;
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.sum = sum;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return trace_return(trace, best, rng);
    }

    candidates.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap());
    let min_prob = (candidates[0].weight / sum) * params.min_p.max(0.0);
    let mut filtered_sum = 0.0_f32;
    let mut filtered = 0_usize;
    for candidate in &candidates {
        let normalized = candidate.weight / sum;
        if filtered > 0 && normalized < min_prob {
            break;
        }
        filtered_sum += candidate.weight;
        if let Some(trace) = trace.as_deref_mut() {
            trace.filtered.push(TraceCandidate {
                normalized_prob: normalized,
                ..*candidate
            });
        }
        filtered += 1;
        if filtered_sum / sum >= params.top_p {
            break;
        }
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.filtered_sum = filtered_sum;
    }
    if filtered == 0 {
        return trace_return(trace, best, rng);
    }

    let rng_unit = sample_rng_f32(rng);
    if let Some(trace) = trace.as_deref_mut() {
        trace.rng_unit = rng_unit;
        trace.rng_after = *rng;
    }
    let mut draw = rng_unit * filtered_sum;
    for candidate in candidates.iter().take(filtered) {
        draw -= candidate.weight;
        if draw <= 0.0 {
            return trace_return(trace, candidate.id, rng);
        }
    }
    trace_return(trace, candidates[filtered - 1].id, rng)
}

fn sample_top_k(
    logits: &[f32],
    params: SamplingParams,
    rng: &mut u64,
    mut trace: Option<&mut SamplingTrace>,
) -> i32 {
    let top_k = params.top_k as usize;
    let mut ids = [0_i32; 1024];
    let mut vals = [0.0_f32; 1024];
    let mut n = 0_usize;
    let mut max_logit = DS4_NEG_INF;
    let mut finite = 0_u32;
    for (idx, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        finite += 1;
        if value > max_logit {
            max_logit = value;
        }
        if n == top_k && value <= vals[n - 1] {
            continue;
        }
        let mut j = if n < top_k {
            let j = n;
            n += 1;
            j
        } else {
            top_k - 1
        };
        while j > 0 && vals[j - 1] < value {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j -= 1;
        }
        vals[j] = value;
        ids[j] = idx as i32;
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.finite_count = finite;
        trace.max_logit = max_logit;
    }
    if n == 0 {
        return trace_return(trace, sample_argmax(logits), rng);
    }

    max_logit = vals[0];
    if let Some(trace) = trace.as_deref_mut() {
        trace.max_logit = max_logit;
    }
    let mut probs = [0.0_f32; 1024];
    let mut sum = 0.0_f32;
    for idx in 0..n {
        probs[idx] = ((vals[idx] - max_logit) / params.temperature).exp();
        sum += probs[idx];
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.sum = sum;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return trace_return(trace, ids[0], rng);
    }

    let min_prob = (probs[0] / sum) * params.min_p;
    let mut filtered_sum = 0.0_f32;
    let mut filtered = 0_usize;
    for idx in 0..n {
        let normalized = probs[idx] / sum;
        if idx > 0 && normalized < min_prob {
            break;
        }
        filtered_sum += probs[idx];
        if let Some(trace) = trace.as_deref_mut() {
            trace.filtered.push(TraceCandidate {
                id: ids[idx],
                logit: vals[idx],
                weight: probs[idx],
                normalized_prob: normalized,
            });
        }
        filtered += 1;
        if filtered_sum / sum >= params.top_p {
            break;
        }
    }
    if let Some(trace) = trace.as_deref_mut() {
        trace.filtered_sum = filtered_sum;
    }
    if filtered == 0 {
        return trace_return(trace, ids[0], rng);
    }

    let rng_unit = sample_rng_f32(rng);
    if let Some(trace) = trace.as_deref_mut() {
        trace.rng_unit = rng_unit;
        trace.rng_after = *rng;
    }
    let mut draw = rng_unit * filtered_sum;
    for idx in 0..filtered {
        draw -= probs[idx];
        if draw <= 0.0 {
            return trace_return(trace, ids[idx], rng);
        }
    }
    trace_return(trace, ids[filtered - 1], rng)
}

pub fn top_logprobs(logits: &[f32], k: usize) -> (usize, Vec<TokenScore>) {
    if k == 0 {
        return (0, Vec::new());
    }
    let k = k.min(DS4_N_VOCAB);
    let mut out = vec![
        TokenScore {
            id: -1,
            logit: DS4_NEG_INF,
            logprob: DS4_NEG_INF,
        };
        k
    ];

    let mut max_logit = DS4_NEG_INF;
    for (idx, &value) in logits.iter().enumerate().take(DS4_N_VOCAB) {
        if !value.is_finite() {
            continue;
        }
        if value > max_logit {
            max_logit = value;
        }
        for slot in 0..k {
            if out[slot].id < 0 || value > out[slot].logit {
                for shift in (slot + 1..k).rev() {
                    out[shift] = out[shift - 1];
                }
                out[slot].id = idx as i32;
                out[slot].logit = value;
                break;
            }
        }
    }
    if !max_logit.is_finite() {
        return (0, out);
    }

    let sum: f64 = logits
        .iter()
        .take(DS4_N_VOCAB)
        .filter(|value| value.is_finite())
        .map(|&value| ((value as f64) - (max_logit as f64)).exp())
        .sum();
    let logsum = (max_logit as f64) + sum.ln();
    for score in &mut out {
        if score.id < 0 {
            continue;
        }
        score.logprob = if score.logit.is_finite() {
            ((score.logit as f64) - logsum) as f32
        } else {
            DS4_NEG_INF
        };
    }
    (k, out)
}

pub fn token_logprob(logits: &[f32], token: usize) -> Option<TokenScore> {
    if token >= DS4_N_VOCAB || token >= logits.len() {
        return None;
    }
    let mut max_logit = DS4_NEG_INF;
    for &value in logits.iter().take(DS4_N_VOCAB) {
        if value.is_finite() && value > max_logit {
            max_logit = value;
        }
    }
    if !max_logit.is_finite() {
        return None;
    }
    let sum: f64 = logits
        .iter()
        .take(DS4_N_VOCAB)
        .filter(|value| value.is_finite())
        .map(|&value| ((value as f64) - (max_logit as f64)).exp())
        .sum();
    let logsum = (max_logit as f64) + sum.ln();
    let logit = logits[token];
    let logprob = if logit.is_finite() {
        ((logit as f64) - logsum) as f32
    } else {
        DS4_NEG_INF
    };
    Some(TokenScore {
        id: token as i32,
        logit,
        logprob,
    })
}

pub fn fill_full_logits(logits: &[f32]) -> Vec<f32> {
    let mut full = vec![DS4_NEG_INF; DS4_N_VOCAB];
    for (idx, value) in logits.iter().enumerate().take(DS4_N_VOCAB) {
        full[idx] = *value;
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_c_fixture_seed() {
        let mut seed = 0x0123_4567_89ab_cdef;
        let value = sample_rng_f32(&mut seed);
        assert_eq!(seed, 12_005_736_692_549_773_564);
        assert!((value - 0.486_641_05).abs() < 1.0e-7);
    }

    #[test]
    fn full_vocab_sampling_records_candidates_from_sampler_path() {
        let logits = [0.0, 1.25, 0.25, 3.5, 2.0, -0.5];
        let params = SamplingParams {
            temperature: 0.9,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.05,
        };
        let mut seed = 0x5555_5555_5555_5555;
        let mut trace = SamplingTrace::new(seed, params);
        let selected = sample_top_p_min_p(&logits, params, &mut seed, Some(&mut trace));
        assert_eq!(selected, 3);
        assert_eq!(trace.effective_top_p, 1.0);
        assert_eq!(trace.filtered.len(), 3);
        assert_eq!(trace.filtered[2].id, 4);
        assert!(trace.filtered[2].normalized_prob > 0.14);
    }

    #[test]
    fn top_logprobs_preserves_tie_order() {
        let full = fill_full_logits(&[2.0, 2.0, 1.0, 0.0]);
        let (returned, scores) = top_logprobs(&full, 4);
        assert_eq!(returned, 4);
        assert_eq!(
            scores.iter().map(|score| score.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }
}
