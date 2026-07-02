// ── Tweak the seed text or the Chirp voice math in lib.rs, then `cargo run`. ──

pub const SAMPLE_RATE: u32 = 44100;

/// The event id whose chirp the playground renders. Change it to audition a different roll of the per-voice randoms.
pub const EVENT: &[u8] = b"photon";

/// The layered-major chirp for [`EVENT`], padded with a little silence on each side so the audio stack is fully spun up before the signal arrives and isn't truncated on drop.
pub fn samples() -> Vec<f32> {
    let body: Vec<f32> = chirp::Chirp::from_bytes(EVENT).collect();
    let pad = SAMPLE_RATE as usize / 4;
    let mut out = Vec::with_capacity(2 * pad + body.len());
    out.extend(std::iter::repeat(0.0).take(pad));
    out.extend(body);
    out.extend(std::iter::repeat(0.0).take(pad));
    out
}
