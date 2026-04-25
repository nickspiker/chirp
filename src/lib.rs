//! Deterministic procedural notification sounds.
//!
//! A [`Chirp`] is a short audio source synthesized from a 256-bit hash. Equal
//! hashes always produce the same chirp; different hashes produce audibly
//! distinct ones. The crate is intended for things like per-event UI
//! notifications where you want the *kind* of event to be recognizable from
//! the sound alone, without curating a sound bank.
//!
//! `Chirp` implements [`rodio::Source`], so it can be played directly through
//! a rodio sink, mixer, or player. See the `play` example.

use std::fmt::Write as _;
use std::num::NonZero;
use std::time::Duration;

use rodio::{ChannelCount, Sample, SampleRate, Source};

const SAMPLE_RATE_HZ: u32 = 44_100;

/// A deterministic notification chirp.
///
/// Construct with [`Chirp::from_hash`] (or [`Chirp::from_bytes`] for arbitrary
/// input). The struct itself is the audio source: iterate it for raw `f32`
/// samples, or hand it to rodio for playback.
pub struct Chirp {
    sample_rate: u32,
    total_samples: u32,
    cursor: u32,
    start_freq: f32,
    end_freq: f32,
    decay: f32,
    phase: f32,
}

impl Chirp {
    /// Build a chirp deterministically from a 256-bit hash.
    ///
    /// The hash is sliced into four 32-bit fields that drive the start
    /// frequency, sweep ratio, duration, and amplitude decay. The mapping is
    /// stable across releases that share a major version.
    pub fn from_hash(hash: [u8; 32]) -> Self {
        let f0_bits = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);
        let f1_bits = u32::from_le_bytes([hash[4], hash[5], hash[6], hash[7]]);
        let dur_bits = u32::from_le_bytes([hash[8], hash[9], hash[10], hash[11]]);
        let decay_bits = u32::from_le_bytes([hash[12], hash[13], hash[14], hash[15]]);

        let start_freq = 400.0 + (f0_bits as f32 / u32::MAX as f32) * 1200.0;
        let ratio = 0.5 + (f1_bits as f32 / u32::MAX as f32) * 1.5;
        let end_freq = start_freq * ratio;
        let duration_ms = 80.0 + (dur_bits as f32 / u32::MAX as f32) * 200.0;
        let total_samples = (SAMPLE_RATE_HZ as f32 * duration_ms / 1000.0) as u32;
        let decay = 2.0 + (decay_bits as f32 / u32::MAX as f32) * 6.0;

        Self {
            sample_rate: SAMPLE_RATE_HZ,
            total_samples,
            cursor: 0,
            start_freq,
            end_freq,
            decay,
            phase: 0.0,
        }
    }

    /// Hash arbitrary bytes with a stable, non-cryptographic mixer and build
    /// a chirp from the result. Useful when the caller has a string event id
    /// rather than a precomputed hash.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_hash(hash256(bytes))
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
        let progress = self.cursor as f32 / self.total_samples as f32;
        let freq = self.start_freq + (self.end_freq - self.start_freq) * progress;
        let envelope = (-self.decay * progress).exp();
        let sample = self.phase.sin() * envelope * 0.4;
        self.phase += 2.0 * std::f32::consts::PI * freq / self.sample_rate as f32;
        self.cursor += 1;
        Some(sample)
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
