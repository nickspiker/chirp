//! Deterministic procedural notification sounds.
//!
//! A [`Chirp`] is a short audio source synthesized from a 256-bit hash. Equal
//! hashes always produce the same chirp; different hashes produce audibly
//! distinct ones. The crate is intended for things like per-event UI
//! notifications where you want the *kind* of event to be recognizable from
//! the sound alone, without curating a sound bank.
//!
//! `Chirp` implements [`rodio::Source`], so it can be played directly thru
//! a rodio sink, mixer, or player. See the `play` example.

use std::fmt::Write as _;
use std::num::NonZero;
use std::time::Duration;

use rodio::{ChannelCount, Sample, SampleRate, Source};

const SAMPLE_RATE_HZ: u32 = 44_100;

/// Chord pitch layers: every voice identity plays all three, at base multipliers 3 : 4 : 5.
/// The multiplier sits INSIDE the FM term (`x·(P + wobble)`), exactly as in the source formula.
const PITCHES: [f64; 3] = [3.0, 4.0, 5.0];

/// Number of voice identities. Each identity owns seven randoms shared by all three of its pitch layers, so an identity's three notes detune, phase, shape, and gate together.
const N_IDENTITIES: usize = 9;

/// One voice identity: seven random parameters in `[-1, 1)`.
/// r0: FM depth, r1: FM rate, r2: phase offset, r3: waveshaper exponent, r4: waveshaper bias, r5: attack offset, r6: decay-rate stretch.
#[derive(Clone, Copy)]
struct Voice {
    r: [f64; 7],
}

impl Voice {
    /// One pitch layer of this identity at plot-space `x` — a verbatim port of the Plotypus formula (one summand of the 27):
    /// `tanh((sin((x·(P + r0·tanh(x·r1 + 1>>2)>>2) + r2·π)<<10) + 1)^(3+r3) − 2 + r4>>1)`
    /// `· sqrt(1 − (max(min(½ + x + (r5>>4), 1), 0) − 1)²)`   (quarter-circle attack)
    /// `· min(0, (x+½)·(r6>>2 + 1) − 1)²`                     (parabolic decay tail)
    fn sample(&self, pitch: f64, x: f64) -> f64 {
        let [r0, r1, r2, r3, r4, r5, r6] = self.r;

        // Tone: slow tanh FM inside a sine, pushed thru an odd waveshaper for harmonics.
        let angle =
            (x * (pitch + r0 * (x * r1 + 0.25).tanh() * 0.25) + r2 * std::f64::consts::PI) * 1024.0;
        let tone = ((angle.sin() + 1.0).powf(3.0 + r3) - 2.0 + r4 * 0.5).tanh();

        // Quarter-circle attack: silent below x ≈ -0.5, full by x ≈ 0.5, edge jittered by r5.
        let t = (0.5 + x + r5 * 0.0625).clamp(0.0, 1.0);
        let attack = (1.0 - (t - 1.0) * (t - 1.0)).max(0.0).sqrt();

        // Parabolic decay: 1 at x = -0.5, zero once (x+0.5)·(r6/4+1) reaches 1, then silent.
        let d = ((x + 0.5) * (r6 * 0.25 + 1.0) - 1.0).min(0.0);
        let tail = d * d;

        tone * attack * tail
    }
}

/// A deterministic notification chirp: nine voice identities, each sounding the
/// 3 : 4 : 5 chord (27 summands), summed and divided by 8 — a verbatim port of the
/// Plotypus-designed formula. All synthesis runs in `f64`; only the emitted sample is `f32`.
///
/// Construct with [`Chirp::from_hash`] (or [`Chirp::from_bytes`] for arbitrary
/// input). The struct itself is the audio source: iterate it for raw `f32`
/// samples, or hand it to rodio for playback.
#[derive(Clone)]
pub struct Chirp {
    sample_rate: u32,
    total_samples: u32,
    cursor: u32,
    /// Plot-space x at t=0 and its rate of change per second — the voice formula is evaluated in x-space.
    x_start: f64,
    x_rate: f64,
    voices: [Voice; 9],
    master: f64,
}

impl Chirp {
    /// Build a chirp deterministically from a 256-bit hash.
    ///
    /// The hash folds to a 64-bit seed; every per-voice parameter is an
    /// independent SplitMix64 expansion of `(seed, index)`. The mapping is
    /// stable across releases that share a major version.
    pub fn from_hash(hash: [u8; 32]) -> Self {
        // Fold the four 64-bit lanes so every hash bit influences the seed.
        let mut seed = 0u64;
        for lane in 0..4 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&hash[lane * 8..lane * 8 + 8]);
            seed ^= u64::from_le_bytes(b).rotate_left(lane as u32 * 17);
        }
        // Duration wobbles per hash (0.75-1.05 s).
        let duration = 0.75 + (mix_unit(seed, 100) * 0.5 + 0.5) * 0.3;
        Self::from_params([0.0; 7], 1.0, seed, duration)
    }

    /// Build a chirp from explicit voice parameters — the tuning-UI entry point.
    ///
    /// `base` is the seven formula knobs (FM depth, FM rate, phase, shape, bias, attack offset, decay stretch), each nominally `-1..1`.
    /// Every voice identity takes `base[j] + jitter·spread` where jitter is an independent `(seed, identity, param)` hash in `-1..1` — `spread = 0` makes all nine identities share the base values exactly, `spread = 1` with zero base reproduces [`Chirp::from_hash`]'s full-random voicing.
    /// `duration` is the clip length in seconds; the formula's x domain `-1..1` sweeps over it.
    pub fn from_params(base: [f64; 7], spread: f64, seed: u64, duration: f64) -> Self {
        let mut voices = [Voice { r: [0.0; 7] }; N_IDENTITIES];
        for (k, v) in voices.iter_mut().enumerate() {
            for (j, r) in v.r.iter_mut().enumerate() {
                *r = (base[j] + mix_unit(seed, (k * 8 + j) as u64) * spread).clamp(-1.0, 1.0);
            }
        }

        let duration = duration.clamp(0.05, 10.0);
        let total_samples = (SAMPLE_RATE_HZ as f64 * duration) as u32;

        Self {
            sample_rate: SAMPLE_RATE_HZ,
            total_samples,
            cursor: 0,
            // The formula's time domain is x ∈ [-1, 1], swept linearly over the duration.
            x_start: -1.0,
            x_rate: 2.0 / duration,
            voices,
            // The source formula's final `>>3`.
            master: 1.0 / 8.0,
        }
    }

    /// Hash arbitrary bytes with a stable, non-cryptographic mixer and build
    /// a chirp from the result. Useful when the caller has a string event id
    /// rather than a precomputed hash.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_hash(hash256(bytes))
    }

    /// The instantaneous signal at formula-space `x ∈ [-1, 1]`, clamped to `[-1, 1]`.
    /// This is the pure function the sample iterator walks — exposed so viewers can
    /// supersample the waveform at arbitrary x positions (stochastic anti-aliasing).
    /// Sum of all identities × all pitch layers (27 summands), scaled by the master `>>3`.
    pub fn signal(&self, x: f64) -> f64 {
        let mut acc = 0.0f64;
        for v in &self.voices {
            for &p in &PITCHES {
                acc += v.sample(p, x);
            }
        }
        (acc * self.master).clamp(-1.0, 1.0)
    }

    /// Render the complete waveform as an SVG string.
    ///
    /// Every sample is included as a point on a white polyline over a black
    /// background. The SVG is 1200×200 pixels.
    pub fn to_svg(self) -> String {
        let samples: Vec<f32> = self.collect();
        samples_to_svg(&samples)
    }
}

/// SplitMix64 of `(seed, idx)` mapped to `[-1, 1)`.
fn mix_unit(seed: u64, idx: u64) -> f64 {
    let mut z = seed ^ idx.wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Render a slice of `f32` samples as an SVG waveform.
///
/// Every sample becomes a point on a white polyline over a black background.
/// The output is a complete 1200×200 SVG document.
pub fn samples_to_svg(samples: &[f32]) -> String {
    let n = samples.len();
    let width = 1200.0_f32;
    let height = 200.0_f32;

    let mut points = String::new();
    for (i, &s) in samples.iter().enumerate() {
        let x = if n > 1 {
            (i as f32 / (n - 1) as f32) * width
        } else {
            width / 2.0
        };
        let y = height / 2.0 - s * (height / 2.0);
        if !points.is_empty() {
            points.push(' ');
        }
        let _ = write!(points, "{x:.1},{y:.1}");
    }

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"200\">\
        \n  <rect width=\"100%\" height=\"100%\" fill=\"black\"/>\
        \n  <polyline points=\"{points}\" fill=\"none\" stroke=\"white\" stroke-width=\"0.1\"/>\
        \n</svg>"
    )
}

impl Iterator for Chirp {
    type Item = Sample;

    fn next(&mut self) -> Option<Sample> {
        if self.cursor >= self.total_samples {
            return None;
        }
        let t = self.cursor as f64 / self.sample_rate as f64;
        let x = self.x_start + self.x_rate * t;
        self.cursor += 1;
        Some(self.signal(x) as f32)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.total_samples - self.cursor) as usize;
        (remaining, Some(remaining))
    }
}

impl Source for Chirp {
    fn current_span_len(&self) -> Option<usize> {
        Some((self.total_samples - self.cursor) as usize)
    }

    fn channels(&self) -> ChannelCount {
        NonZero::new(1).unwrap()
    }

    fn sample_rate(&self) -> SampleRate {
        NonZero::new(self.sample_rate).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f32(
            self.total_samples as f32 / self.sample_rate as f32,
        ))
    }
}

/// Produce a 256-bit hash from arbitrary bytes. Not cryptographic; we only
/// need stability and good avalanche so that small input differences produce
/// audibly different chirps. Runs four independent FxHash-style passes with
/// distinct seeds.
fn hash256(bytes: &[u8]) -> [u8; 32] {
    const SEEDS: [u64; 4] = [
        0xcbf29ce484222325,
        0x517cc1b727220a95,
        0x6c62272e07bb0142,
        0x9e3779b97f4a7c15,
    ];
    const ROT: u32 = 5;
    const PRIME: u64 = 0x100000001b3;
    let mut out = [0u8; 32];
    for (i, &seed) in SEEDS.iter().enumerate() {
        let mut h = seed;
        for &b in bytes {
            h = (h.rotate_left(ROT) ^ b as u64).wrapping_mul(PRIME);
        }
        out[i * 8..i * 8 + 8].copy_from_slice(&h.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: [u8; 32] = [
        0xde, 0xad, 0xbe, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba,
        0x98, 0x76, 0x54, 0x32, 0x10, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96,
        0xa5, 0xb4,
    ];

    const HASH_B: [u8; 32] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x20,
    ];

    #[test]
    fn same_hash_same_samples() {
        let a: Vec<f32> = Chirp::from_hash(HASH_A).collect();
        let b: Vec<f32> = Chirp::from_hash(HASH_A).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn different_hashes_differ() {
        let a: Vec<f32> = Chirp::from_hash(HASH_A).collect();
        let b: Vec<f32> = Chirp::from_hash(HASH_B).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn samples_are_bounded() {
        for s in Chirp::from_hash(HASH_A) {
            assert!(s.abs() <= 1.0, "sample {s} out of range");
        }
    }

    #[test]
    fn not_silent() {
        let peak = Chirp::from_hash(HASH_A).fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.05, "chirp is near-silent (peak {peak})");
    }

    #[test]
    fn svg_contains_all_samples() {
        let chirp = Chirp::from_hash(HASH_A);
        let expected_points = chirp.total_samples as usize;
        let svg = Chirp::from_hash(HASH_A).to_svg();
        assert!(svg.starts_with("<svg"));
        // Each sample becomes a point; points are space-separated.
        let start = svg.find("points=\"").unwrap() + 8;
        let end = svg[start..].find('"').unwrap() + start;
        let point_count = svg[start..end].split(' ').count();
        assert_eq!(point_count, expected_points);
    }
}
