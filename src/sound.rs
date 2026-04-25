// ── Tweak these constants, change the formula, then `cargo run`. ──

pub const SAMPLE_RATE: u32 = 44100;
pub const DURATION: f64 = 4.;
pub const START_FREQ: f64 = 800.;
pub const END_FREQ: f64 = 1200.;
pub const AMPLITUDE: f64 = 0.4;

/// Generate audio samples. Edit the math to shape the waveform.
pub fn samples() -> Vec<f32> {
    let total = (SAMPLE_RATE as f64 * DURATION) as usize;
    let mut out = Vec::with_capacity(total);
    let mut phase: f64 = 0.;

    for i in 0..total {
        let seconds = i as f64 / SAMPLE_RATE as f64-1.;
        let progress = i as f64 / total as f64;
        let freq = START_FREQ + (END_FREQ - START_FREQ) * progress;
        let envelope = 1. - (1. - seconds * seconds*seconds).sqrt();
        let sample = phase.sin() * envelope * AMPLITUDE;
        phase += 2.0 * std::f64::consts::PI * freq / SAMPLE_RATE as f64;
        out.push(sample as f32);
    }

    out
}
