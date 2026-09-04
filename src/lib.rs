//! Deterministic procedural notification sounds — a struck metal bell, synthesized modally.
//!
//! A [`Chirp`] is a short audio source synthesized from a 256-bit hash. Equal
//! hashes always produce the same chirp; different hashes produce audibly
//! distinct ones. The crate is intended for things like per-event UI
//! notifications where you want the *kind* of event to be recognizable from
//! the sound alone, without curating a sound bank.
//!
//! **Synthesis model: modal.** A struck bell/bar/glass is a small set of exponentially
//! decaying sine partials: `y(t) = Σ aₖ·sin(2π·fₖ·t)·exp(−t/τₖ)`. The partial frequency
//! ratios come from measured material tables (bar / church bell / glass, morphable),
//! decay times fall with frequency (highs die first — the strongest "struck metal" cue),
//! amplitudes tilt with strike hardness, and a short noise burst supplies the hammer clank.
//! A deterministic post chain (sympathetic resonators, echo, Schroeder reverb) adds the room.
//! The finished clip is peak-normalized (no clamping anywhere) and trimmed to its sounding span.
//!
//! `Chirp` implements [`rodio::Source`], so it can be played directly thru
//! a rodio sink, mixer, or player. See the `play` example.

use std::fmt::Write as _;
#[cfg(feature = "playback")]
use std::num::NonZero;
#[cfg(feature = "playback")]
use std::time::Duration;

// rodio's `Sample` is an alias for f32; the Iterator impl below uses f32 directly so it stays
// available without the playback feature (to_svg / haptic / to_wav all need the sample stream).
#[cfg(feature = "playback")]
use rodio::{ChannelCount, SampleRate, Source};

const SAMPLE_RATE_HZ: u32 = 44_100;

/// Maximum partial count (the `partials` knob selects 3..=10 of these).
pub const MAX_PARTIALS: usize = 10;

/// Partial frequency ratio tables, lowest mode = 1.0 (except the bell's hum note below the prime).
/// BAR: free-free uniform bar (glockenspiel / tine) — the classic clean "ding".
const R_BAR: [f64; MAX_PARTIALS] = [1.0, 2.756, 5.404, 8.933, 13.345, 18.638, 24.813, 31.870, 39.808, 48.629];
/// BELL: church-bell voicing — hum (0.5), prime, minor-third tierce, quint, nominal, and upper partials.
const R_BELL: [f64; MAX_PARTIALS] = [0.5, 1.0, 1.188, 1.506, 2.0, 2.662, 3.011, 4.166, 5.433, 6.796];
/// GLASS: wine-glass / thin shell modes — sparse, glassy, nearly-pure.
const R_GLASS: [f64; MAX_PARTIALS] = [1.0, 2.32, 4.25, 6.63, 9.38, 12.22, 15.53, 19.22, 23.28, 27.70];

/// One decaying sine mode of the struck object. `det` is a small frequency offset for a doubled
/// partial pair — two near-identical modes beating slowly against each other (bell warble).
#[derive(Clone, Copy)]
struct Partial {
    freq: f64,
    amp: f64,
    tau: f64,
    det: f64,
}

/// Macro knobs for the bell voice, all in `0..1` (sliders map 1:1; internal ranges documented per field).
#[derive(Clone, Copy, Debug)]
pub struct BellParams {
    /// Fundamental pitch: exp-mapped 220..2000 Hz.
    pub pitch: f64,
    /// Material morph: 0 = glass, 0.5 = bar (glockenspiel), 1 = church bell. Interpolates the ratio tables.
    pub material: f64,
    /// Fundamental ring time τ₀: exp-mapped ~0.08..4.4 s.
    pub decay: f64,
    /// Spectral decay slope α in τₖ = τ₀/(rₖ)^α: 0 → highs ring nearly as long as lows (glassy sustain), 1 → highs vanish instantly (dull thud).
    pub slope: f64,
    /// Strike hardness: tilts energy into the high modes. 0 = soft mallet (mostly fundamental), 1 = hard hammer (bright clank spectrum).
    pub strike: f64,
    /// Inharmonic stretch: raises each ratio to (1 + 0.08·inharm) — sharpens the upper partials the way real thick plates stretch.
    pub inharm: f64,
    /// Warble: doubles each partial with a slow-beating detuned twin, up to ~5 cents.
    pub shimmer: f64,
    /// Mode count: 0 → 3 partials (pure), 1 → all 10 (rich).
    pub partials: f64,
    /// Hammer noise burst amount: a few ms of decaying noise at the strike.
    pub clank: f64,
    /// Sub-octave hum partial (0.5×f₀, rings 1.5× longer than the fundamental) — church-bell undertone.
    pub hum: f64,
    /// Phase-mod depth β in sin(θ + β·sin(2θ)) — the DX7 electric-bell sound. 0 = acoustic, 1 = full electric.
    /// Also alias-guarded per partial (sidebands stay under Nyquist).
    pub fm: f64,
    /// Swept-sine "zap" transient: a 3-octave exponential sweep to/from the fundamental over ~25 ms (direction is a per-bell seed draw).
    /// The laser/droid vocabulary, where clank is the hammer's.
    pub blip: f64,
}

/// Post-chain macro parameters, each in `0..1`. All are plain numbers — the chain has no runtime randomness, so equal params + seed give bit-identical output.
#[derive(Clone, Copy, Debug)]
pub struct PostParams {
    /// Room size: scales the reverb comb delays AND their feedback (bigger room rings longer).
    pub room: f64,
    /// Wall damping: one-pole lowpass in the comb feedback. 0 = bright/metallic, 1 = warm/dead.
    pub damp: f64,
    /// Echo delay time: 0..1 → 5..30 ms (tight slapback; anything longer reads as lag on a notification).
    pub echo_time: f64,
    /// Echo feedback amount: 0 = single slap, 1 = long repeating trail.
    pub echo_gain: f64,
    /// Harmonic resonance: sympathetic combs tuned to the bell's lowest partials sing along.
    pub resonance: f64,
    /// Reverb wet mix — the ambiance floor. 0 bypasses the reverb entirely.
    pub wet: f64,
    /// Ring-mod depth on the dry voice (before the room): ×(1−d + d·sin(2π·f_rm·t)), f_rm seed-drawn 40..226 Hz.
    /// Even shallow depth reads retro-futurist metallic (the Dalek patina).
    pub ring: f64,
}

/// Feedback comb: `y[n] = buf[n-D]`, `buf[n] = x + y·g`. The echo line and the tuned harmonic resonators.
#[derive(Clone)]
struct Comb {
    buf: Vec<f64>,
    idx: usize,
    g: f64,
}

impl Comb {
    fn new(delay: usize, g: f64) -> Self {
        Self { buf: vec![0.0; delay.max(2)], idx: 0, g }
    }
    fn process(&mut self, x: f64) -> f64 {
        let y = self.buf[self.idx];
        self.buf[self.idx] = x + y * self.g;
        self.idx = (self.idx + 1) % self.buf.len();
        y
    }
}

/// Damped feedback comb (Freeverb-style): the feedback path runs thru a one-pole lowpass, so high frequencies die faster than lows — the "absorbent walls" of the reverb.
#[derive(Clone)]
struct DampComb {
    buf: Vec<f64>,
    idx: usize,
    feedback: f64,
    damp: f64,
    store: f64,
}

impl DampComb {
    fn new(delay: usize, feedback: f64, damp: f64) -> Self {
        Self { buf: vec![0.0; delay.max(2)], idx: 0, feedback, damp, store: 0.0 }
    }
    fn process(&mut self, x: f64) -> f64 {
        let y = self.buf[self.idx];
        self.store = y * (1.0 - self.damp) + self.store * self.damp;
        self.buf[self.idx] = x + self.store * self.feedback;
        self.idx = (self.idx + 1) % self.buf.len();
        y
    }
}

/// Schroeder allpass diffuser: smears comb echoes into a dense tail without colouring the spectrum.
#[derive(Clone)]
struct Allpass {
    buf: Vec<f64>,
    idx: usize,
    g: f64,
}

impl Allpass {
    fn new(delay: usize, g: f64) -> Self {
        Self { buf: vec![0.0; delay.max(2)], idx: 0, g }
    }
    fn process(&mut self, x: f64) -> f64 {
        let d = self.buf[self.idx];
        let y = d - self.g * x;
        self.buf[self.idx] = x + self.g * d;
        self.idx = (self.idx + 1) % self.buf.len();
        y
    }
}

/// The assembled mono post chain: dry → (+ tuned resonators) → (+ echo) → Schroeder reverb → wet mix.
/// State starts zeroed and every step is pure DSP, so the output is a deterministic function of (params, seed, input stream).
#[derive(Clone)]
struct Post {
    res: [Comb; 3],
    res_gain: f64,
    echo: Comb,
    echo_gain: f64,
    combs: [DampComb; 4],
    aps: [Allpass; 2],
    wet: f64,
}

impl Post {
    /// `res_freqs` are the sounding frequencies (Hz) the sympathetic resonators tune to — the bell's lowest partials.
    fn new(p: PostParams, digest: &[u8; 32], res_freqs: [f64; 3]) -> Self {
        let sr = SAMPLE_RATE_HZ as f64;

        let res_g = 0.82 + p.resonance * 0.15;
        let res = std::array::from_fn(|i| {
            let delay = (sr / res_freqs[i].max(20.0)).round() as usize;
            Comb::new(delay, res_g)
        });

        // Echo: 5..30 ms — a tight slapback thickener, not a laggy repeat.
        // Seed-jittered ±5% so no two hashes share a slap time exactly.
        // The comb's first tap emerges at FULL input amplitude, so the audible slap level must be gated by the output gain (echo_gain param), not the comb feedback — feedback only shapes how the trail of repeats decays.
        let echo_delay = ((0.005 + p.echo_time * 0.025)
            * (1.0 + channel("echo delay jitter", digest) * 0.05)
            * sr) as usize;
        let echo = Comb::new(echo_delay, p.echo_gain * 0.55);

        // Reverb: Freeverb comb tunings scaled by room size, each seed-jittered ±3% (mutually prime-ish lengths keep the tail dense instead of fluttery).
        let scale = 0.6 + p.room * 1.4;
        let base_delays = [1116.0, 1188.0, 1277.0, 1356.0];
        let feedback = (0.70 + p.room * 0.28).min(0.98);
        let damp = 0.15 + p.damp * 0.55;
        let combs = std::array::from_fn(|i| {
            let jitter = 1.0 + channel(&format!("reverb comb{i} jitter"), digest) * 0.03;
            DampComb::new((base_delays[i] * scale * jitter) as usize, feedback, damp)
        });
        let aps = [Allpass::new(556, 0.5), Allpass::new(441, 0.5)];

        Self {
            res,
            res_gain: p.resonance * 0.45,
            echo,
            echo_gain: p.echo_gain * 0.8,
            combs,
            aps,
            wet: p.wet,
        }
    }

    /// Zero every delay line and filter state — back to the pristine just-constructed chain.
    fn reset(&mut self) {
        for r in &mut self.res {
            r.buf.fill(0.0);
            r.idx = 0;
        }
        self.echo.buf.fill(0.0);
        self.echo.idx = 0;
        for c in &mut self.combs {
            c.buf.fill(0.0);
            c.idx = 0;
            c.store = 0.0;
        }
        for a in &mut self.aps {
            a.buf.fill(0.0);
            a.idx = 0;
        }
    }

    /// One sample thru the chain. Order: sympathetic resonance rides on the dry, echo taps that, reverb diffuses the sum, wet mixes the room back in.
    fn process(&mut self, dry: f64) -> f64 {
        let mut res_sum = 0.0;
        for r in &mut self.res {
            res_sum += r.process(dry);
        }
        let voiced = dry + res_sum * self.res_gain;

        let echoed = voiced + self.echo.process(voiced) * self.echo_gain;

        let mut rev = 0.0;
        for c in &mut self.combs {
            rev += c.process(echoed);
        }
        rev *= 0.25;
        for a in &mut self.aps {
            rev = a.process(rev);
        }

        echoed + rev * self.wet
    }
}

/// One struck bell within the chime: its modes, its hammer, and when it gets hit.
#[derive(Clone)]
struct Strike {
    /// Strike time in seconds from clip start.
    offset: f64,
    partials: Vec<Partial>,
    /// Overall level of this bell (root slightly louder than the harmony bells).
    level: f64,
    /// Hammer noise burst: amplitude, and the seed the per-sample noise hashes from.
    clank: f64,
    clank_seed: u64,
    /// This bell's fundamental (Hz) — anchor for the blip sweep.
    f0: f64,
    /// Electric voice knobs, locked across the chime like the other instrument-identity params: phase-mod depth, swept-blip amount, and the blip's seed-drawn sweep direction (sign: ≥0 sweeps up from f₀, <0 falls onto it).
    fm: f64,
    blip: f64,
    blip_dir: f64,
    /// This bell's sawtooth-gate randoms `[r0, r1, r2]` in `-1..1` (rate stretch, phase offset, duty bias).
    saw: [f64; 3],
    /// This bell's sounding life in seconds (4τ of its slowest mode) — the gate's x runs 0..1 over it.
    saw_life: f64,
}

impl Strike {
    /// The bell's dry output at clip time `t` (silent before its strike offset).
    fn sample(&self, t: f64) -> f64 {
        let t = t - self.offset;
        if t < 0.0 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for q in &self.partials {
            // Envelope: (x-1)²·√(1-(x-1)²) over x ∈ 0..1, x = t normalized to this mode's lifetime (4τ, so higher modes still die first).
            // Zero at BOTH ends (no exponential tail, no click), peak 2/(3√3) at x ≈ 0.184, scaled to unity.
            let x = t / (q.tau * 4.0);
            if x >= 1.0 {
                continue;
            }
            let u = x - 1.0;
            let u2 = u * u;
            let env = u2 * (1.0 - u2).max(0.0).sqrt() * 2.598_076_211_353_316;
            // Alias guard: FM sidebands extend the partial's bandwidth, so the depth fades back toward pure sine as the partial climbs.
            let beta = self.fm * 2.5 * (3000.0 / q.freq).min(1.0);
            let theta = std::f64::consts::TAU * q.freq * t;
            // Phase mod at 2:1 — sin(θ + β·sin(2θ)), the classic electric-bell operator stack.
            let a = (theta + (theta * 2.0).sin() * beta).sin();
            let s = if q.det != 0.0 {
                let theta2 = std::f64::consts::TAU * (q.freq + q.det) * t;
                (a + (theta2 + (theta2 * 2.0).sin() * beta).sin()) * 0.5
            } else {
                a
            };
            acc += q.amp * s * env;
        }
        // Hammer: white noise hashed per sample index, 4 ms exponential burst.
        if self.clank > 0.0 && t < 0.02 {
            let n = (t * SAMPLE_RATE_HZ as f64) as u64;
            acc += mix_unit(self.clank_seed, 700 + n) * self.clank * 0.8 * (-t / 0.004).exp();
        }
        // Zap: swept sine, 3 exponential octaves to/from f₀ over 25 ms (direction seed-drawn).
        // Envelope decays fast AND is pinned to zero at the window edge — no cutoff click.
        if self.blip > 0.0 && t < 0.025 {
            const T_BLIP: f64 = 0.025;
            let (f1, f2) = if self.blip_dir >= 0.0 {
                (self.f0, self.f0 * 8.0)
            } else {
                (self.f0 * 8.0, self.f0)
            };
            let r = f2 / f1;
            let tt = t / T_BLIP;
            // Constant-Q exponential sweep: phase = 2π·f1·T/ln r · (r^(t/T) − 1).
            let phase = std::f64::consts::TAU * f1 * T_BLIP / r.ln() * (r.powf(tt) - 1.0);
            acc += phase.sin() * self.blip * 0.9 * (-t / 0.006).exp() * (1.0 - tt);
        }
        // Per-bell sawtooth gate, x ∈ 0..1 over THIS bell's sounding life (the formula's own +1 shifts the exponent to 4..8):
        // (tanh(sin(2^((x·(1+r0/4) + 1 + r1/16)·4))·4 + r2·2) + 1)/2
        // sin of an exponentially growing phase = accelerating gate cycles; ×4 into tanh saturates them square (digital chop); r2 biases the duty toward open or shut.
        let [s0, s1, s2] = self.saw;
        let x = t / self.saw_life;
        let phase = 2.0f64.powf((x * (1.0 + s0 * 0.25) + 1.0 + s1 * 0.0625) * 4.0);
        let env = ((phase.sin() * 4.0 + s2 * 2.0).tanh() + 1.0) * 0.5;
        acc * self.level * env
    }
}

/// Chime-level knobs, all in `0..1`: how the three bells relate in time and character. Pitch relation is fixed sus4 (see [`CHORD`]).
#[derive(Clone, Copy, Debug)]
pub struct ChimeParams {
    /// Strike scatter: each bell's strike time is an independent seed-drawn offset in a ±(spacing × 20 ms) window — near-simultaneous, any order (whichever bell's draw lands earliest strikes first).
    pub spacing: f64,
    /// How much the three castings differ: scales each bell's per-partial jitter. 0 = identical copies, 1 = clearly individual bells.
    pub spread: f64,
    /// Arpeggiation: morphs the ±20 ms scatter flam into a deliberate ascending ladder (root → 4/3 → 3/2 at 45 ms steps).
    /// 1 reads as "computer acknowledgment", not "bell".
    pub arp: f64,
}

/// A deterministic notification chime: three matched modal bells in harmony (shared
/// material / damping / mallet, per-bell pitch / seed / strike time), struck into ONE
/// shared room. All synthesis runs in `f64`; the finished clip is peak-normalized and trimmed.
#[derive(Clone)]
pub struct Chirp {
    sample_rate: u32,
    total_samples: u32,
    cursor: u32,
    strikes: Vec<Strike>,
    /// Deterministic post FX chain (resonators + echo + reverb). `None` = dry.
    post: Option<Post>,
    /// Sample count of the dry voice span. `total_samples` extends past this for the post tail.
    dry_samples: u32,
    /// Sounding frequencies of the root bell's three lowest partials — the post resonators tune to these.
    res_freqs: [f64; 3],
    /// Ring-mod on the dry voice (depth 0..1, carrier Hz) — applied before the room so the resonators/reverb ring the modulated tone, not a clean bell with a warble pasted on.
    ring_depth: f64,
    ring_freq: f64,
    /// The finished clip: rendered in f64 with NO clamping, peak-normalized to full scale, then trimmed to the span that actually sounds (every patch's lead-in delay and ring-out length differ).
    /// Built by `finalize()` at construction; the Iterator/Source impls just stream it.
    rendered: Vec<f32>,
}

/// Interpolate the material ratio tables: 0 = glass, 0.5 = bar, 1 = bell.
fn material_ratio(material: f64, k: usize) -> f64 {
    let m = material.clamp(0.0, 1.0);
    if m < 0.5 {
        let t = m * 2.0;
        R_GLASS[k] * (1.0 - t) + R_BAR[k] * t
    } else {
        let t = (m - 0.5) * 2.0;
        R_BAR[k] * (1.0 - t) + R_BELL[k] * t
    }
}

/// The chime chord: strictly sus4 (1 : 4/3 : 3/2) — no third, unresolved, attention-getting; the PA/transit-chime family. Beats least against casting jitter of the just triads.
const CHORD: [f64; 3] = [1.0, 4.0 / 3.0, 1.5];

/// The call-ring tremolo: one sine arc of this many half-cycles (0 → 9π) across the finished ding clip — 10 zeros counting the ends, 4 inverted lobes (Nick 2026-09-03).
const RING_HALF_CYCLES: f64 = 9.0;

/// Build one bell's mode set at fundamental `f0`.
/// `jitter` scales the per-partial randomness (the chime's `spread`: 0 = exact casting, 1 = full individuality).
/// `prefix` namespaces this bell's channels (e.g. "bell0 "), so the three castings draw independent jitters from one digest.
fn build_partials(p: &BellParams, f0: f64, prefix: &str, digest: &[u8; 32], jitter: f64) -> Vec<Partial> {
    let tau0 = 0.02 * (p.decay.clamp(0.0, 1.0) * 2.5).exp(); // 0.02..0.24 s — the dry ring (4τ) tops out around a second
    let alpha = 0.3 + p.slope.clamp(0.0, 1.0) * 1.7;
    let tilt = 1.6 - p.strike.clamp(0.0, 1.0) * 1.2; // hard strike flattens the spectrum
    let stretch = 1.0 + p.inharm.clamp(0.0, 1.0) * 0.08;
    let n_partials = 3 + (p.partials.clamp(0.0, 1.0) * (MAX_PARTIALS - 3) as f64).round() as usize;

    let mut partials = Vec::with_capacity(n_partials + 1);
    for k in 0..n_partials {
        let jf = 1.0 + channel(&format!("{prefix}partial{k} freq"), digest) * 0.003 * jitter;
        let jt = 1.0 + channel(&format!("{prefix}partial{k} tau"), digest) * 0.25 * jitter;
        let ja = 1.0 + channel(&format!("{prefix}partial{k} amp"), digest) * 0.2 * jitter;
        let ratio = material_ratio(p.material, k).powf(stretch) * jf;
        let freq = f0 * ratio;
        // Nyquist guard: partials that would alias are dropped, not folded.
        if freq > SAMPLE_RATE_HZ as f64 * 0.45 {
            continue;
        }
        // Warble: beating twin up to ~5 cents (0.3% of the partial), seed-signed so pairs drift both ways.
        let det = p.shimmer.clamp(0.0, 1.0)
            * freq
            * 0.003
            * channel(&format!("{prefix}partial{k} warble"), digest);
        partials.push(Partial {
            freq,
            amp: ja / ratio.max(0.4).powf(tilt),
            tau: (tau0 / ratio.max(0.4).powf(alpha)) * jt,
            det,
        });
    }
    // Sub-octave hum: rings 1.5x longer than the fundamental — the church-bell undertone.
    if p.hum > 0.0 {
        partials.push(Partial {
            freq: f0 * 0.5,
            amp: p.hum * 0.7,
            tau: tau0 * 1.5,
            det: 0.0,
        });
    }
    partials
}

impl Chirp {
    /// Build a chime deterministically from a 32-byte digest (e.g. `spaghettify(theirs ‖ mine)`): bells, harmony, timing, and room all derive from named blake3 channels of it.
    pub fn from_hash(hash: [u8; 32]) -> Self {
        // Every param is a channel mapped into its ear-locked range (July 2026 electric tuning session,
        // zone sliders in the tune bin; supersedes the 29-keeper acoustic ranges). The tuner's RANGES
        // table mirrors this exactly — keep them in sync when re-tuning.
        let r = |name: &str, lo: f64, hi: f64| lo + channel_unit(name, &hash) * (hi - lo);
        let bell = BellParams {
            pitch: r("pitch", 0.42, 0.90),
            material: r("material", 0.56, 0.79),
            decay: r("decay", 0.10, 0.49),
            slope: r("slope", 0.0, 1.0),
            strike: r("strike", 0.0, 1.0),
            inharm: r("inharm", 0.45, 0.95),
            shimmer: r("shimmer", 0.0, 1.0),
            partials: r("partials", 0.0, 1.0),
            clank: r("clank", 0.0, 1.0),
            hum: r("hum", 0.0, 1.0),
            fm: r("fm", 0.40, 0.80),
            blip: r("blip", 0.0, 1.0),
        };
        let chime = ChimeParams {
            spacing: r("chime spacing", 0.0, 1.0),
            spread: r("chime spread", 0.0, 1.0),
            arp: r("chime arp", 0.0, 0.60),
        };
        let post = PostParams {
            room: r("room size", 0.05, 0.16),
            damp: r("room damp", 0.38, 1.0),
            echo_time: r("echo time", 0.18, 0.45),
            echo_gain: r("echo gain", 0.21, 0.58),
            resonance: r("resonance", 0.60, 1.0),
            wet: r("wet", 0.38, 0.78),
            ring: r("ring", 0.22, 0.77),
        };
        Self::from_chime(bell, chime, &hash).with_post(post, &hash)
    }

    /// Build a single bell (a chime with zero spacing and one strike) — kept for API symmetry and quick tests.
    pub fn from_bell(p: BellParams, digest: &[u8; 32]) -> Self {
        Self::from_chime(p, ChimeParams { spacing: 0.0, spread: 0.0, arp: 0.0 }, digest)
    }

    /// Build a three-bell chime from explicit macro knobs — the tuning-UI entry point.
    ///
    /// Locked across the three bells: material, damping slope, mallet (strike + clank), inharmonic stretch, partial count, shimmer amount — the instrument identity.
    /// Broken apart per bell: pitch (chord ratios off the root), strike time (spacing × order), channel namespace (each bell its own casting, scaled by `spread`), decay (higher bell rings shorter, physics), and level (root slightly louder, ±4% human jitter).
    /// The root fundamental maps to 1024..2048 Hz — desk bell, not church tower.
    pub fn from_chime(p: BellParams, c: ChimeParams, digest: &[u8; 32]) -> Self {
        Self::chime_scheduled(p, c, digest, &[(0.0, 1.0)], 1.1)
    }

    /// The ring: the ding itself — the bit-identical [`from_hash`] render, post chain and all —
    /// multiplied by ONE sine arc spanning the whole clip, phase 0 → 9π (Nick 2026-09-03). It opens
    /// and closes on a zero, dips to silence 10 times counting the ends, and inverts thru 4 of its
    /// 9 lobes. No phrase schedule, no envelope of its own; the tremolo IS the articulation.
    ///
    /// The clip is one cadence; the caller loops it (with silence between repeats if desired)
    /// until the call is answered. Same digest → same ring, and it audibly IS the contact whose
    /// messages chirp — just conjugated from "ding" into "answer me".
    pub fn ring_from_hash(hash: [u8; 32]) -> Self {
        let mut ring = Self::from_hash(hash);
        let n = ring.rendered.len();
        if n > 1 {
            let mut peak = 0.0f32;
            for (i, s) in ring.rendered.iter_mut().enumerate() {
                let phase = std::f64::consts::PI * RING_HALF_CYCLES * i as f64 / (n - 1) as f64;
                *s *= phase.sin() as f32;
                peak = peak.max(s.abs());
            }
            // Re-normalize: the tremolo shaved the peak unless it happened to land mid-lobe.
            if peak > 0.0 {
                let g = 1.0 / peak;
                for s in &mut ring.rendered {
                    *s *= g;
                }
            }
        }
        ring
    }

    /// Strike the whole three-bell chime once per `(onset_secs, level)` schedule entry, all into one
    /// clip. The castings (partials) are built once per bell and shared across hits — one instrument,
    /// many strikes; only per-hit things (scatter draw, hammer noise stream) get hit-namespaced
    /// channels. `dry_cap` bounds the dry span in seconds (1.1 for the notification, longer for rings).
    fn chime_scheduled(
        p: BellParams,
        c: ChimeParams,
        digest: &[u8; 32],
        hits: &[(f64, f64)],
        dry_cap: f64,
    ) -> Self {
        Self::chime_scheduled_var(p, c, digest, hits, dry_cap, 0.0, 1.0)
    }

    /// [`chime_scheduled`] plus the per-hit smidgle: `hit_var` scales a per-hit re-roll of every
    /// partial's frequency (±0.2%, a few cents), decay (±10%), and amplitude (±10%) — each hit is
    /// the same bell struck by a hand, not a copy-paste. 0 disables it (bit-identical castings).
    /// `tau_scale` uniformly scales every partial's τ (0.5 = each strike's envelope runs in half
    /// the time) — the ring's hits are the ding damped, nothing else touched.
    fn chime_scheduled_var(
        p: BellParams,
        c: ChimeParams,
        digest: &[u8; 32],
        hits: &[(f64, f64)],
        dry_cap: f64,
        hit_var: f64,
        tau_scale: f64,
    ) -> Self {
        let f_root = 1024.0 * 2.0f64.powf(p.pitch.clamp(0.0, 1.0)); // 1024..2048 Hz
        // Strike scatter: each bell draws an independent offset in ±(spacing × 20 ms) — near-simultaneous flam, any order.
        let window = c.spacing.clamp(0.0, 1.0) * 0.020;

        let mut strikes = Vec::with_capacity(3 * hits.len());
        for k in 0..3 {
            let ratio = CHORD[k];
            // Each bell is its own casting: its channel names are prefixed "bell{k} ", so the three draw independent jitters from the one digest.
            let bp = format!("bell{k} ");
            let mut partials = build_partials(&p, f_root * ratio, &bp, digest, c.spread);
            // Physics: the higher (smaller) bell rings shorter. Scale every mode's tau down by the
            // chord ratio — then by the caller's envelope scale (the ring damps its hits to half).
            let bell_tau = tau_scale / ratio.powf(0.75);
            for q in &mut partials {
                q.tau *= bell_tau;
            }
            // Level: root a touch louder than the harmony bells, ±4% per-strike human jitter.
            let level = (1.0 / ratio.powf(0.3))
                * (1.0 + channel(&format!("{bp}level"), digest) * 0.04);
            // Each bell's sawtooth gate runs over its OWN sounding life with its OWN randoms — the three chop patterns interleave instead of one gate slicing the mix.
            let saw_life = partials
                .iter()
                .map(|q| q.tau)
                .fold(0.0f64, f64::max)
                .max(0.01)
                * 4.0;
            let arp = c.arp.clamp(0.0, 1.0);
            for (h, &(onset, hit_level)) in hits.iter().enumerate() {
                // Per-hit namespace suffix. Hit 0 keeps the bare names so a one-hit schedule is
                // channel-for-channel identical to the original chime — same digest, same ding.
                let hs = if h == 0 { String::new() } else { format!("hit{h} ") };
                // The smidgle: re-roll this hit's casting a hair. Hit 0 stays the exact casting.
                let mut hit_partials = partials.clone();
                if hit_var > 0.0 && h > 0 {
                    for (qi, q) in hit_partials.iter_mut().enumerate() {
                        q.freq *= 1.0 + channel(&format!("{bp}{hs}p{qi} freq"), digest) * 0.002 * hit_var;
                        q.tau *= 1.0 + channel(&format!("{bp}{hs}p{qi} tau"), digest) * 0.10 * hit_var;
                        q.amp *= 1.0 + channel(&format!("{bp}{hs}p{qi} amp"), digest) * 0.10 * hit_var;
                    }
                }
                // Arp morphs the scatter draw toward a deliberate ascending ladder (45 ms steps).
                let scatter = channel(&format!("{bp}{hs}offset"), digest) * window;
                strikes.push(Strike {
                    offset: onset + scatter * (1.0 - arp) + k as f64 * 0.045 * arp,
                    partials: hit_partials,
                    level: level * hit_level,
                    clank: p.clank.clamp(0.0, 1.0),
                    clank_seed: channel_u64(&format!("{bp}{hs}clank noise"), digest),
                    f0: f_root * ratio,
                    fm: p.fm.clamp(0.0, 1.0),
                    blip: p.blip.clamp(0.0, 1.0),
                    blip_dir: channel(&format!("{bp}blip dir"), digest),
                    saw: [
                        channel(&format!("{bp}saw rate"), digest),
                        channel(&format!("{bp}saw phase"), digest),
                        channel(&format!("{bp}saw duty"), digest),
                    ],
                    saw_life,
                });
            }
        }
        // Shift so the earliest strike lands at t = 0 (offsets were drawn in ±window).
        let min_offset = strikes.iter().map(|s| s.offset).fold(f64::MAX, f64::min);
        for s in &mut strikes {
            s.offset -= min_offset;
        }

        // Clip length: last strike + ~4 time constants of the slowest mode anywhere.
        let max_tau = strikes
            .iter()
            .flat_map(|s| s.partials.iter().map(|q| q.tau))
            .fold(0.0f64, f64::max);
        let last_offset = strikes.iter().map(|s| s.offset).fold(0.0f64, f64::max);
        // Dry span capped by the caller: ~1.1 s for the notification (a desk bell, not a resonating hall), longer for ring phrases.
        let duration = (last_offset + max_tau * 4.0 + 0.1).clamp(0.2, dry_cap);
        let total_samples = (SAMPLE_RATE_HZ as f64 * duration) as u32;

        // Resonator targets: the root bell's three lowest sounding partials.
        let mut freqs: Vec<f64> = strikes[0].partials.iter().map(|q| q.freq).collect();
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let res_freqs = [
            *freqs.first().unwrap_or(&f_root),
            *freqs.get(1).unwrap_or(&(f_root * 2.0)),
            *freqs.get(2).unwrap_or(&(f_root * 3.0)),
        ];

        Self {
            sample_rate: SAMPLE_RATE_HZ,
            total_samples,
            cursor: 0,
            strikes,
            post: None,
            dry_samples: total_samples,
            res_freqs,
            ring_depth: 0.0,
            ring_freq: 0.0,
            rendered: Vec::new(),
        }
        .finalize()
    }

    /// Install the deterministic post chain (resonators + echo + reverb) and extend the clip so the room tail rings out instead of being chopped at the dry end.
    /// `digest` jitters the comb/echo delays (named channels) so distinct digests get audibly distinct rooms; equal (params, digest) is bit-identical.
    pub fn with_post(mut self, params: PostParams, digest: &[u8; 32]) -> Self {
        // Tail length scales with how live the room is, capped at ~0.9 s so dry + tail stays inside 2 s total.
        let tail_secs = (0.15
            + params.wet * (0.5 + params.room * 1.2)
            + params.echo_gain * (0.005 + params.echo_time * 0.025) * 3.0
            + params.resonance * 0.3)
            .min(0.9);
        self.total_samples = self.dry_samples + (SAMPLE_RATE_HZ as f64 * tail_secs) as u32;
        self.post = Some(Post::new(params, digest, self.res_freqs));
        self.ring_depth = params.ring.clamp(0.0, 1.0);
        // Carrier 40..226 Hz, exp-mapped and seed-drawn: low enough to read as metallic sidebands, not pitch.
        self.ring_freq = 40.0 * 2f64.powf(channel_unit("ring freq", digest) * 2.5);
        self.finalize()
    }

    /// Hash arbitrary bytes and build a bell from the result.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_hash(hash256(bytes))
    }

    /// The dry chime at time `t` seconds — a pure function (each strike's modal sum is memoryless).
    pub fn signal(&self, t: f64) -> f64 {
        self.strikes.iter().map(|s| s.sample(t)).sum()
    }

    /// Render, normalize, and trim — the second pass that turns the synth into the finished clip.
    ///
    /// 1. Raw render in f64 with NO clamping: dry voice over its span, post chain (if any) over the whole extended length.
    /// 2. Peak-normalize: full scale is defined by the clip's own min/max, so the post chain can swing past ±1 internally without distortion and quiet patches still fill the range.
    /// 3. Trim: find the first and last samples above -60 dB of peak and cut the silence outside them (each patch's lead-in delay and ring-out length differ, so start/stop are found per clip, not assumed).
    /// Short edge fades (1.5 ms in, 12 ms out) keep the trimmed boundaries click-free.
    fn finalize(mut self) -> Self {
        let n = self.total_samples as usize;
        let sr = self.sample_rate as f64;
        let mut raw = Vec::with_capacity(n);
        for i in 0..n {
            let mut dry = if (i as u32) < self.dry_samples {
                self.signal(i as f64 / sr)
            } else {
                0.0
            };
            if self.ring_depth > 0.0 {
                let lfo = (std::f64::consts::TAU * self.ring_freq * i as f64 / sr).sin();
                dry *= 1.0 - self.ring_depth + self.ring_depth * lfo;
            }
            let out = match &mut self.post {
                Some(p) => p.process(dry),
                None => dry,
            };
            raw.push(out);
        }
        // Reset post state so a hypothetical second finalize starts clean.
        if let Some(p) = &mut self.post {
            p.reset();
        }

        let peak = raw.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        if peak > 0.0 {
            let scale = 1.0 / peak;
            for s in &mut raw {
                *s *= scale;
            }
        }

        // Soft saturation: √(1 − x²/2)·x·√2 — naturally odd (the bare x carries the sign), monotonic on -1..1, and maps ±1 → ±1 EXACTLY (√(1/2)·√2 = 1), so no re-normalize is needed. Slope √2 at zero: a gentle knee.
        for s in &mut raw {
            *s = (1.0 - *s * *s * 0.5).sqrt() * *s * std::f64::consts::SQRT_2;
        }

        // -60 dB threshold relative to the (now unit) peak.
        let thr = 1.0 / 1024.0;
        let first = raw.iter().position(|s| s.abs() > thr).unwrap_or(0);
        let last = raw.iter().rposition(|s| s.abs() > thr).unwrap_or(n.saturating_sub(1));
        // Keep ~1.5 ms of pre-roll so a fast strike's leading edge isn't shaved.
        let start = first.saturating_sub((sr * 0.0015) as usize);
        let end = (last + 1).min(n);
        let mut clip: Vec<f32> = raw[start..end].iter().map(|&s| s as f32).collect();

        // Edge fades: 1.5 ms in, 12 ms out.
        let fade_in = ((sr * 0.0015) as usize).min(clip.len());
        for i in 0..fade_in {
            clip[i] *= i as f32 / fade_in as f32;
        }
        let fade_out = ((sr * 0.012) as usize).min(clip.len());
        let len = clip.len();
        for i in 0..fade_out {
            clip[len - 1 - i] *= i as f32 / fade_out as f32;
        }

        self.total_samples = clip.len() as u32;
        self.cursor = 0;
        self.rendered = clip;
        self
    }

    /// Render the complete waveform as an SVG string (1200×200, white polyline on black).
    pub fn to_svg(self) -> String {
        let samples: Vec<f32> = self.collect();
        samples_to_svg(&samples)
    }

    /// The chirp's amplitude envelope as a vibration waveform, for driving a phone's haptic motor
    /// in step with the sound. `rate_hz` is the motor's amplitude update rate (~200 Hz is a good
    /// default; the LRA smears anything faster, which just reads as texture).
    ///
    /// Derivation, deliberately dumb so ALL the flavor survives — the clank, the beating warble, the
    /// sawtooth chop, the reverb tail all ride thru as texture rather than being averaged away by a
    /// per-partial envelope model:
    /// 1. **Abs** the finished clip → `|sample|`, the raw loudness stream.
    /// 2. **Full-width bin** 44.1 kHz → `rate_hz`: each output step is the MEAN of every input sample
    ///    in its window (a box-filter anti-alias — NOT nearest-neighbor, which would alias the carrier
    ///    into garbage). One bin ≈ 44100/rate_hz input samples.
    /// 3. **Normalize** to the envelope's own peak so the strongest buzz hits motor max.
    /// 4. **Quantize** to `1..=255` (0 reserved for genuine silence) — the `VibrationEffect.createWaveform`
    ///    amplitude range. Paired `timings` are all `step_ms = 1000/rate_hz`.
    ///
    /// Returns `(timings_ms, amplitudes)`, equal-length, ready for `createWaveform(timings, amplitudes, -1)`.
    /// Same hash → same envelope (the clip is deterministic), so a contact's haptic matches their chirp.
    pub fn haptic_waveform(&self, rate_hz: u32) -> (Vec<u32>, Vec<u8>) {
        let rate_hz = rate_hz.max(1);
        let bin = (self.sample_rate as f64 / rate_hz as f64).max(1.0);
        let n = self.rendered.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let n_bins = ((n as f64) / bin).ceil() as usize;

        // Full-width binning: mean of sample² (power) over each bin. Bin edges are floating so the last
        // (partial) bin averages only the samples it actually covers — no zero-padding tail.
        let mut env = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let start = (b as f64 * bin).floor() as usize;
            let end = (((b + 1) as f64 * bin).floor() as usize).min(n);
            if end <= start {
                continue;
            }
            let sum: f64 = self.rendered[start..end].iter().map(|s| (*s as f64) * (*s as f64)).sum();
            env.push(sum / (end - start) as f64);
        }

        let peak = env.iter().fold(0.0f64, |m, &v| m.max(v));
        let step_ms = (1000.0 / rate_hz as f64).round().max(1.0) as u32;
        let amplitudes: Vec<u8> = env
            .iter()
            .map(|&v| {
                if peak <= 0.0 {
                    0
                } else {
                    // Normalize to peak, then map (0,1] → 1..=255. Genuine silence stays 0.
                    let norm = v / peak;
                    if norm <= 0.0 {
                        0
                    } else {
                        (norm * 255.0).round().clamp(1.0, 255.0) as u8
                    }
                }
            })
            .collect();
        let timings = vec![step_ms; amplitudes.len()];
        (timings, amplitudes)
    }

    /// The finished, peak-normalized f32 clip (−1..=1), for a caller that wants the raw samples —
    /// e.g. to hand to a platform audio API. `sample_rate()` gives the rate. Deterministic per hash.
    pub fn samples(&self) -> &[f32] {
        &self.rendered
    }

    /// The finished clip as a self-contained mono 16-bit PCM WAV (44-byte header + samples), so the
    /// host app can hand the bytes straight to a platform player (Android MediaPlayer/AudioTrack reads
    /// this as-is). No extra deps — the header is hand-built. Same hash → byte-identical WAV.
    pub fn to_wav(&self) -> Vec<u8> {
        let sr = self.sample_rate;
        let n = self.rendered.len();
        let data_len = (n * 2) as u32; // 16-bit mono → 2 bytes/sample
        let mut out = Vec::with_capacity(44 + n * 2);

        // RIFF header
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes()); // chunk size = 36 + data
        out.extend_from_slice(b"WAVE");
        // fmt subchunk (PCM)
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size (16 for PCM)
        out.extend_from_slice(&1u16.to_le_bytes()); // audio format = 1 (PCM)
        out.extend_from_slice(&1u16.to_le_bytes()); // channels = 1 (mono)
        out.extend_from_slice(&sr.to_le_bytes()); // sample rate
        out.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate = sr * channels * bytes/sample
        out.extend_from_slice(&2u16.to_le_bytes()); // block align = channels * bytes/sample
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // data subchunk
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for &s in &self.rendered {
            // f32 −1..=1 → i16, clamped so a sample sitting exactly at ±1 doesn't wrap.
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Play on the default output device, blocking until the clip finishes.
    #[cfg(feature = "playback")]
    pub fn play_blocking(self) -> Result<(), String> {
        let mut sink = rodio::DeviceSinkBuilder::open_default_sink()
            .map_err(|e| format!("no default audio output device: {e}"))?;
        // Dropping the sink after playback is deliberate; silence rodio's drop warning.
        sink.log_on_drop(false);
        let player = rodio::Player::connect_new(&sink.mixer());
        player.append(self);
        player.sleep_until_end();
        Ok(())
    }

    /// Fire-and-forget playback on a detached thread — the notification-app entry point.
    /// Errors (e.g. no audio device) are logged to stderr, never fatal: a missing sound card must not take the host app down.
    #[cfg(feature = "playback")]
    pub fn play_detached(self) {
        std::thread::spawn(move || {
            if let Err(e) = self.play_blocking() {
                eprintln!("chirp: {e}");
            }
        });
    }
}

/// One deterministic aesthetic channel from a 32-byte relationship digest: `blake3(name ‖ digest)`, first 8 bytes as i64, divided by `i64::MIN`.
/// Range `(-1, +1]` with ±1 actually reachable (i64::MIN is the one value whose magnitude is exactly 2^63).
/// Distinct names = statistically independent channels — the name IS the domain separation, so add channels freely (colour, haptics, …) without a registry.
pub fn channel(name: &str, digest: &[u8; 32]) -> f64 {
    channel_u64(name, digest) as i64 as f64 / i64::MIN as f64
}

/// Right-handed unipolar variant: same derivation, u64 divided by u64::MAX → `[0, 1]`.
pub fn channel_unit(name: &str, digest: &[u8; 32]) -> f64 {
    channel_u64(name, digest) as f64 / u64::MAX as f64
}

/// Raw 64-bit channel value (the first quarter of `blake3(name ‖ digest)`) — for seeding per-sample noise streams.
pub fn channel_u64(name: &str, digest: &[u8; 32]) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(name.as_bytes());
    h.update(digest);
    let mut out = [0u8; 8];
    out.copy_from_slice(&h.finalize().as_bytes()[..8]);
    u64::from_le_bytes(out)
}

/// SplitMix64 of `(seed, idx)` mapped to `[-1, 1)` — the per-sample NOISE STREAM generator (hammer clank).
/// Parameters use the named blake3 [`channel`]s; this stays for streams where one blake3 per sample would be silly. Seed it from [`channel_u64`].
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
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.cursor >= self.total_samples {
            return None;
        }
        // Stream the finalized clip — rendering, normalization, and trimming all happened at construction.
        let s = self.rendered[self.cursor as usize];
        self.cursor += 1;
        Some(s)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.total_samples - self.cursor) as usize;
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "playback")]
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
    fn normalized_to_full_scale() {
        let peak = Chirp::from_hash(HASH_A).fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.95, "clip should be peak-normalized (peak {peak})");
    }

    #[test]
    #[cfg(feature = "playback")]
    #[ignore] // requires audio output — run with: cargo test -- --ignored play_audible
    fn play_audible() {
        Chirp::from_hash(HASH_A).play_blocking().expect("playback A failed");
        Chirp::from_hash(HASH_B).play_blocking().expect("playback B failed");
    }

    #[test]
    fn haptic_same_hash_same_waveform() {
        let a = Chirp::from_hash(HASH_A).haptic_waveform(200);
        let b = Chirp::from_hash(HASH_A).haptic_waveform(200);
        assert_eq!(a, b);
    }

    #[test]
    fn haptic_different_hashes_differ() {
        let a = Chirp::from_hash(HASH_A).haptic_waveform(200);
        let b = Chirp::from_hash(HASH_B).haptic_waveform(200);
        assert_ne!(a, b);
    }

    #[test]
    fn haptic_arrays_equal_length_and_stepped() {
        let (timings, amps) = Chirp::from_hash(HASH_A).haptic_waveform(200);
        assert_eq!(timings.len(), amps.len(), "createWaveform needs equal-length arrays");
        assert!(!amps.is_empty());
        // 200 Hz → 5 ms steps.
        assert!(timings.iter().all(|&t| t == 5), "all steps should be 1000/rate ms");
    }

    #[test]
    fn haptic_bin_count_matches_rate() {
        // ~200 Hz over the clip's duration ≈ duration_secs * 200 bins (±1 for the partial last bin).
        let chirp = Chirp::from_hash(HASH_A);
        let secs = chirp.total_samples as f64 / chirp.sample_rate as f64;
        let (_, amps) = chirp.haptic_waveform(200);
        let expected = (secs * 200.0).round() as i64;
        assert!(
            (amps.len() as i64 - expected).abs() <= 1,
            "bins {} should be ~{} for a {:.3}s clip at 200Hz",
            amps.len(), expected, secs,
        );
    }

    #[test]
    fn haptic_is_binned_not_nearest_neighbor() {
        // Nearest-neighbor would pick single raw |samples| — mostly near 0 or the sparse peaks, and
        // it would resample the ±1 CARRIER, so consecutive steps would swing wildly. Full-width
        // binning averages ~220 samples/bin into a smooth envelope: adjacent steps move gradually and
        // the whole thing is far from the raw carrier's variance. Assert the envelope is smooth —
        // mean absolute step-to-step change is small relative to the range.
        let (_, amps) = Chirp::from_hash(HASH_A).haptic_waveform(200);
        assert!(amps.len() > 10);
        let mut jumps = 0u64;
        let mut big = 0u64;
        for w in amps.windows(2) {
            let d = (w[1] as i32 - w[0] as i32).unsigned_abs();
            jumps += 1;
            if d > 128 {
                big += 1;
            }
        }
        // A binned envelope almost never jumps more than half-scale between adjacent 5ms steps;
        // a nearest-neighbor carrier resample would do so constantly.
        assert!(
            big * 5 < jumps,
            "too many half-scale jumps ({big}/{jumps}) — looks like carrier aliasing, not binning",
        );
    }

    #[test]
    fn haptic_normalized_to_full_scale() {
        let (_, amps) = Chirp::from_hash(HASH_A).haptic_waveform(200);
        assert_eq!(*amps.iter().max().unwrap(), 255, "envelope should hit motor max after normalize");
    }

    #[test]
    fn samples_match_rendered_length() {
        let chirp = Chirp::from_hash(HASH_A);
        assert_eq!(chirp.samples().len(), chirp.total_samples as usize);
        assert!(!chirp.samples().is_empty());
    }

    #[test]
    fn wav_header_is_valid_pcm_mono() {
        let chirp = Chirp::from_hash(HASH_A);
        let wav = chirp.to_wav();
        let n = chirp.samples().len();
        // Header + 2 bytes/sample (16-bit mono).
        assert_eq!(wav.len(), 44 + n * 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1, "PCM format");
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1, "mono");
        assert_eq!(u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]), chirp.sample_rate);
        assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16, "16-bit");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), (n * 2) as u32);
    }

    #[test]
    fn wav_is_deterministic() {
        assert_eq!(Chirp::from_hash(HASH_A).to_wav(), Chirp::from_hash(HASH_A).to_wav());
        assert_ne!(Chirp::from_hash(HASH_A).to_wav(), Chirp::from_hash(HASH_B).to_wav());
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
