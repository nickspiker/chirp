// ── Ring prototype playground: tweak the seed text or ring_from_hash in lib.rs, then `cargo run --bin chirp`. ──
//
// Renders `Chirp::ring_from_hash` — the identity voice struck in a "ring-ring" call phrase (2 single strikes rung long under a full-depth 4.5 Hz tremolo, one shared room) — looped three times with a gap, the way an incoming call would repeat it until answered.

pub const SAMPLE_RATE: u32 = 44100;

/// The event id whose ring the playground renders. Change it to audition a different roll of the per-voice randoms.
pub const EVENT: &[u8] = b"photon";

/// Seconds of silence between the A and B candidates and between cadence repeats.
const REPEAT_GAP: f64 = 1.2;

/// Cadence repeats of the production ring (variation 1, half-tau hits), the way a call would loop it.
const REPEATS: usize = 3;

pub fn samples() -> Vec<f32> {
    let digest = *blake3::hash(EVENT).as_bytes();
    let ring = chirp::Chirp::ring_from_hash(digest);

    let pad = SAMPLE_RATE as usize / 4;
    let gap = (REPEAT_GAP * SAMPLE_RATE as f64) as usize;
    let mut out = Vec::new();
    out.extend(std::iter::repeat(0.0).take(pad));
    for k in 0..REPEATS {
        if k > 0 {
            out.extend(std::iter::repeat(0.0).take(gap));
        }
        out.extend_from_slice(ring.samples());
    }
    out.extend(std::iter::repeat(0.0).take(pad));
    out
}
