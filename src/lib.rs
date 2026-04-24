//! Deterministic procedural notification sounds.
//!
//! A [`Chirp`] is a short audio source synthesized from a 64-bit hash. Equal
//! hashes always produce the same chirp; different hashes produce audibly
//! distinct ones. The crate is intended for things like per-event UI
//! notifications where you want the *kind* of event to be recognizable from
//! the sound alone, without curating a sound bank.
//!
//! `Chirp` implements [`rodio::Source`], so it can be played directly through
//! a rodio sink, mixer, or player. See the `play` example.

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
    /// Build a chirp deterministically from a 64-bit hash.
    ///
    /// The hash is sliced into four 16-bit fields that drive the start
    /// frequency, sweep ratio, duration, and amplitude decay. The mapping is
    /// stable across releases that share a major version.
    pub fn from_hash(hash: u64) -> Self {
        let f0_bits = (hash & 0xFFFF) as u32;
        let f1_bits = ((hash >> 16) & 0xFFFF) as u32;
        let dur_bits = ((hash >> 32) & 0xFFFF) as u32;
        let decay_bits = ((hash >> 48) & 0xFFFF) as u32;

        let start_freq = 400.0 + (f0_bits as f32 / 65535.0) * 1200.0;
        let ratio = 0.5 + (f1_bits as f32 / 65535.0) * 1.5;
        let end_freq = start_freq * ratio;
        let duration_ms = 80.0 + (dur_bits as f32 / 65535.0) * 200.0;
        let total_samples = (SAMPLE_RATE_HZ as f32 * duration_ms / 1000.0) as u32;
        let decay = 2.0 + (decay_bits as f32 / 65535.0) * 6.0;

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
        Self::from_hash(fxhash64(bytes))
    }
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

/// FxHash-style 64-bit hash. Not cryptographic; we only need stability and
/// good avalanche so that small input differences produce audibly different
/// chirps.
fn fxhash64(bytes: &[u8]) -> u64 {
    const SEED: u64 = 0xcbf29ce484222325;
    const ROT: u32 = 5;
    const PRIME: u64 = 0x100000001b3;
    let mut h = SEED;
    for &b in bytes {
        h = (h.rotate_left(ROT) ^ b as u64).wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_hash_same_samples() {
        let a: Vec<f32> = Chirp::from_hash(0xdeadbeef).collect();
        let b: Vec<f32> = Chirp::from_hash(0xdeadbeef).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn different_hashes_differ() {
        let a: Vec<f32> = Chirp::from_hash(1).collect();
        let b: Vec<f32> = Chirp::from_hash(2).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn samples_are_bounded() {
        for s in Chirp::from_hash(0x1234_5678_9abc_def0) {
            assert!(s.abs() <= 1.0, "sample {s} out of range");
        }
    }
}
