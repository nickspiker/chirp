//! Bell tuning UI — modal-synthesis knobs + room, one slider column.
//!
//! Waveform on top (rendered buffer, post chain included, supersampled with stratified jitter, green = exact zero).
//! Each knob row is three sliders: ⚄-zone min, the knob itself, ⚄-zone max — drag the outer pair to tune the randomizer's range live.
//! ⚄ random rolls every knob inside its zone; `n` rerolls the per-partial jitter seed at the same knob settings; ▶ play (or Enter / `p`) renders + plays; `l` logs the knobs; ⎙ ranges prints the zone table paste-ready.

use std::num::NonZero;

use fluor::canvas::{Canvas, PixelRect};
use fluor::coord::Coord;
use fluor::event::{CursorIcon, ElementState, Event, Key, KeyEvent, MouseButton, NamedKey};
use fluor::geom::Viewport;
use fluor::host::app::{Context, EventResponse, FluorApp};
use fluor::host::chrome::{self, HIT_NONE, HitId, ResizeEdge};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::host::widget::{self as widget, Container, TabDir, Widget};
use fluor::pixel::Blend;
use fluor::text::TextStyle;
use fluor::widgets::{Button, Slider};
use fluor::BlendMode;

/// Deterministic per-(column, subsample) jitter in `[0, 1)` — SplitMix64 hash, same as plotypus's.
#[inline]
fn sample_jitter(px: usize, ss: usize) -> f64 {
    let mut h = (px as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((ss as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Slider order: twelve bell knobs, three chime knobs, seven room knobs. All 0..1, passed straight thru. Chord is fixed sus4.
const SLIDER_LABELS: [&str; 22] = [
    "pitch", "material", "decay", "slope", "strike", "inharm", "shimmer", "partials", "clank",
    "hum", "fm", "blip", "spacing", "spread", "arp", "room", "walls", "echo·t", "echo·g",
    "reson", "wet", "ring",
];
const N_SLIDERS: usize = SLIDER_LABELS.len();

/// Default knob positions: a near-simultaneous sus4 desk-bell chime in a small, mostly-dry room, fm centered in its locked band.
const DEFAULTS: [f32; N_SLIDERS] = [
    0.5, 0.45, 0.5, 0.4, 0.6, 0.1, 0.2, 0.5, 0.35, 0.1, // bell (acoustic)
    0.6, 0.5, // bell (electric): fm, blip
    0.5, 0.5, 0.0, // chime: ±10 ms scatter, individual castings, no arp
    0.3, 0.5, 0.2, 0.05, 0.1, 0.12, // room
    0.0, // room (electric): ring
];

/// ⚄-zone presets — the min/max flanking sliders start here; the roll draws inside whatever they currently read.
/// Knob sliders stay full 0..1 for manual exploration. These are the SHIPPED ranges (July 2026 electric session) and mirror `Chirp::from_hash` exactly — keep them in sync when re-tuning.
const RANGES: [(f32, f32); N_SLIDERS] = [
    (0.42, 0.90), // pitch
    (0.56, 0.79), // material
    (0.10, 0.49), // decay
    (0.00, 1.00), // slope
    (0.00, 1.00), // strike
    (0.45, 0.95), // inharm
    (0.00, 1.00), // shimmer
    (0.00, 1.00), // partials
    (0.00, 1.00), // clank
    (0.00, 1.00), // hum
    (0.40, 0.80), // fm
    (0.00, 1.00), // blip
    (0.00, 1.00), // spacing
    (0.00, 1.00), // spread
    (0.00, 0.60), // arp
    (0.05, 0.16), // room
    (0.38, 1.00), // walls
    (0.18, 0.45), // echo·t
    (0.21, 0.58), // echo·g
    (0.60, 1.00), // reson
    (0.38, 0.78), // wet
    (0.22, 0.77), // ring
];

const PINK:         u32 = 0xFF00_0000 | (0x00FF_8080 ^ 0x00FF_FFFF);
const BLUE:         u32 = 0xFF00_0000 | (0x0080_80FF ^ 0x00FF_FFFF);
const PLOT_BG:      u32 = 0xFFFF_FFFF;
const ZERO_LINE:    u32 = 0xFF00_0000 | (0x0040_4040 ^ 0x00FF_FFFF);
const ZERO_FILL:    u32 = 0xFF00_0000 | (0x0000_E0_00 ^ 0x00FF_FFFF); // green — exact-zero samples, plotypus convention
const LABEL_COLOUR: u32 = 0xFF00_0000 | (0x00C0_C0C0 ^ 0x00FF_FFFF);

/// Digest source for the per-partial jitter + room comb channels — fixed while dialing knobs so the instrument holds still; `n` rerolls it (digest = blake3(digest)).
const BASE_DIGEST_INPUT: &[u8] = b"bell tune base digest";

struct TuneApp {
    title: String,
    chrome: DefaultChrome,
    sliders: Vec<Slider>,
    /// ⚄-zone bounds flanking each knob: min slider on the left, max on the right. Randomizer draws inside them.
    lo_sliders: Vec<Slider>,
    hi_sliders: Vec<Slider>,
    play_button: Button,
    random_button: Button,
    ranges_button: Button,
    ring_button: Button,
    rng: u64,
    hit_counter: HitId,
    current_focus: Option<HitId>,
    /// (vec, row): vec 0 = knob, 1 = zone min, 2 = zone max.
    dragging: Option<(usize, usize)>,
    modifiers: fluor::event::ModifiersState,

    /// Rendered clip for the current knobs — the plot source and the play buffer.
    samples: Vec<f32>,
    samples_dirty: bool,
    /// Jitter digest for partials + room comb channels; `n` rerolls it (blake3 of itself).
    digest: [u8; 32],

    plot_rect: (usize, usize, usize, usize),
    slider_area_top: Coord,
    row_pitch: Coord,
}

impl TuneApp {
    fn new() -> Self {
        let viewport = Viewport::new(1100, 900);
        let mut hit_counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(
            viewport,
            "Bell Tune".to_string(),
            None,
            Some("ready".to_string()),
            &mut hit_counter,
        );

        let mut sliders = Vec::with_capacity(N_SLIDERS);
        let mut lo_sliders = Vec::with_capacity(N_SLIDERS);
        let mut hi_sliders = Vec::with_capacity(N_SLIDERS);
        for i in 0..N_SLIDERS {
            sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, DEFAULTS[i]));
            lo_sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, RANGES[i].0));
            hi_sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, RANGES[i].1));
        }
        let play_button   = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "▶ play");
        let random_button = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "⚄ random");
        let ranges_button = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "⎙ ranges");
        let ring_button   = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "☎ ring");

        let rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5EED_5EED_5EED_5EED);

        Self {
            title: "Bell Tune".to_string(),
            chrome,
            sliders,
            lo_sliders,
            hi_sliders,
            play_button,
            random_button,
            ranges_button,
            ring_button,
            rng,
            hit_counter,
            current_focus: None,
            dragging: None,
            modifiers: fluor::event::ModifiersState::default(),
            samples: Vec::new(),
            samples_dirty: true,
            digest: *blake3::hash(BASE_DIGEST_INPUT).as_bytes(),
            plot_rect: (0, 0, 0, 0),
            slider_area_top: 0.0,
            row_pitch: 0.0,
        }
    }

    /// Slider values → BellParams + PostParams, all 0..1 straight thru.
    fn rebuild_samples(&mut self) {
        let v = |i: usize| self.sliders[i].value() as f64;
        let bell = chirp::BellParams {
            pitch: v(0),
            material: v(1),
            decay: v(2),
            slope: v(3),
            strike: v(4),
            inharm: v(5),
            shimmer: v(6),
            partials: v(7),
            clank: v(8),
            hum: v(9),
            fm: v(10),
            blip: v(11),
        };
        let chime = chirp::ChimeParams {
            spacing: v(12),
            spread: v(13),
            arp: v(14),
        };
        let post = chirp::PostParams {
            room: v(15),
            damp: v(16),
            echo_time: v(17),
            echo_gain: v(18),
            resonance: v(19),
            wet: v(20),
            ring: v(21),
        };
        let synth =
            chirp::Chirp::from_chime(bell, chime, &self.digest).with_post(post, &self.digest);
        self.samples = synth.collect();
        self.samples_dirty = false;
    }

    /// Play the current buffer on a background thread.
    fn play(&self) {
        let samples = self.samples.clone();
        std::thread::spawn(move || {
            let Ok(mut handle) = rodio::DeviceSinkBuilder::open_default_sink() else {
                eprintln!("audio: no default output device");
                return;
            };
            // We drop the sink deliberately after sleep_until_end; silence rodio's drop warning.
            handle.log_on_drop(false);
            let player = rodio::Player::connect_new(&handle.mixer());
            let buf = rodio::buffer::SamplesBuffer::new(
                NonZero::new(1u16).unwrap(),
                NonZero::new(44_100u32).unwrap(),
                samples,
            );
            player.append(buf);
            player.sleep_until_end();
        });
    }

    fn update_layout(&mut self, ctx: &mut Context) {
        let vp = ctx.viewport;
        let w = vp.width_px as Coord;
        let h = vp.height_px as Coord;
        let span = vp.effective_span();
        // RU unit — visual weight only (band height, fonts, buttons). Positions are viewport fractions.
        let bw = span / 32.0;
        let margin = w * 0.03;

        // Waveform plot: 7%..38%.
        let plot_top = h * 0.07;
        let plot_bottom = h * 0.38;
        self.plot_rect = (
            margin as usize,
            plot_top as usize,
            (w - margin * 2.0).max(8.0) as usize,
            (plot_bottom - plot_top).max(8.0) as usize,
        );

        // Slider stack: 41%..91%, one row per knob.
        self.slider_area_top = h * 0.41;
        let slider_area_h = h * 0.50;
        self.row_pitch = slider_area_h / N_SLIDERS as Coord;
        let band_h = (bw * 1.1).max(10.0).min(self.row_pitch * 0.9);

        // Horizontal: label 3%, min-value 10.5%, zone-min 14%..21.5%, knob 23%..70%, zone-max 71.5%..79%, max-value 79.5%, knob value 83.5%.
        let lo_cx = w * 0.1775;
        let lo_w = w * 0.075;
        let knob_left = w * 0.23;
        let knob_w = w * 0.47;
        let knob_cx = knob_left + knob_w * 0.5;
        let hi_cx = w * 0.7525;
        for i in 0..N_SLIDERS {
            let cy = self.slider_area_top + (i as Coord + 0.5) * self.row_pitch;
            self.lo_sliders[i].set_rect(lo_cx, cy, lo_w, band_h * 0.8);
            self.sliders[i].set_rect(knob_cx, cy, knob_w, band_h);
            self.hi_sliders[i].set_rect(hi_cx, cy, lo_w, band_h * 0.8);
        }

        let btn_w = bw * 7.0;
        let btn_h = bw * 1.7;
        let btn_cy = h * 0.955;
        self.play_button
            .set_rect(w - margin - btn_w * 0.5, btn_cy, btn_w, btn_h);
        self.play_button.set_font_size(bw * 0.9);
        self.random_button
            .set_rect(w - margin - btn_w * 1.5 - bw * 0.5, btn_cy, btn_w, btn_h);
        self.random_button.set_font_size(bw * 0.9);
        self.ranges_button
            .set_rect(w - margin - btn_w * 2.5 - bw * 1.0, btn_cy, btn_w, btn_h);
        self.ranges_button.set_font_size(bw * 0.9);
        self.ring_button
            .set_rect(w - margin - btn_w * 3.5 - bw * 1.5, btn_cy, btn_w, btn_h);
        self.ring_button.set_font_size(bw * 0.9);
    }

    /// Next uniform in `0..1` from the UI RNG (splitmix64 churn).
    fn next_rand(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Poll every slider's change counter. Knob moves dirty the clip; zone moves only need a redraw (they shape the NEXT ⚄ roll, not the current sound).
    fn poll_slider_changes(&mut self) -> bool {
        let mut changed = false;
        for s in &mut self.sliders {
            if s.take_change() {
                changed = true;
                self.samples_dirty = true;
            }
        }
        for s in self.lo_sliders.iter_mut().chain(self.hi_sliders.iter_mut()) {
            if s.take_change() {
                changed = true;
            }
        }
        changed
    }

    /// Row `i`'s ⚄ zone from the live flanking sliders, order-normalized so a crossed pair still works.
    fn zone(&self, i: usize) -> (f32, f32) {
        let a = self.lo_sliders[i].value();
        let b = self.hi_sliders[i].value();
        (a.min(b), a.max(b))
    }

    /// Poll both buttons; fires their actions. Called after any dispatch that could have clicked them.
    fn poll_buttons(&mut self, ctx: &mut Context) {
        if self.ranges_button.take_click() {
            // Paste-ready RANGES table from the live zone sliders.
            println!("const RANGES: [(f32, f32); N_SLIDERS] = [");
            for i in 0..N_SLIDERS {
                let (lo, hi) = self.zone(i);
                println!("    ({lo:.2}, {hi:.2}), // {}", SLIDER_LABELS[i]);
            }
            println!("];");
        }
        if self.random_button.take_click() {
            for i in 0..N_SLIDERS {
                let (lo, hi) = self.zone(i);
                let v = lo + self.next_rand() * (hi - lo);
                self.sliders[i].set_value(v);
            }
            self.poll_slider_changes();
            // Audition immediately — no separate play press per roll.
            self.rebuild_samples();
            self.play();
            ctx.window.request_redraw();
        }
        if self.play_button.take_click() {
            if self.samples_dirty {
                self.rebuild_samples();
            }
            self.play();
        }
        if self.ring_button.take_click() {
            self.audition_ring(ctx);
        }
    }

    /// ☎ (or `r`): roll a fresh digest and audition its production RING — sliders are ignored, the ring derives every knob from the digest exactly as a real caller's identity does. Each press is a new rando; the digest prints for reproducibility.
    fn audition_ring(&mut self, ctx: &mut Context) {
        self.digest = *blake3::hash(&self.digest).as_bytes();
        self.samples = chirp::Chirp::ring_from_hash(self.digest).collect();
        self.samples_dirty = true;
        let hex: String = self.digest.iter().map(|b| format!("{b:02x}")).collect();
        println!("ring digest: {hex}");
        self.play();
        ctx.window.request_redraw();
    }

    fn change_focus(&mut self, new_focus: Option<HitId>, ctx: &mut Context) {
        if new_focus == self.current_focus {
            return;
        }
        let prior = self.current_focus;
        widget::apply_focus_change(self as &mut dyn Container, prior, new_focus);
        self.current_focus = new_focus;
        ctx.window.request_redraw();
    }

    fn handle_key(&mut self, kev: &KeyEvent, ctx: &mut Context) -> EventResponse {
        if kev.state != ElementState::Pressed {
            return EventResponse::Pass;
        }
        if matches!(kev.logical_key, Key::Named(NamedKey::Tab)) {
            let dir = if self.modifiers.shift_key() {
                TabDir::Backward
            } else {
                TabDir::Forward
            };
            let current = self.current_focus;
            let next = widget::linear_tab_next(self as &mut dyn Container, current, dir);
            self.change_focus(next, ctx);
            return EventResponse::Handled;
        }
        if matches!(kev.logical_key, Key::Named(NamedKey::Escape)) {
            self.change_focus(None, ctx);
            return EventResponse::Handled;
        }
        if matches!(kev.logical_key, Key::Named(NamedKey::Enter)) {
            if self.samples_dirty {
                self.rebuild_samples();
            }
            self.play();
            return EventResponse::Handled;
        }
        // `n` = reroll the jitter digest: same knobs, sibling instrument.
        if let Key::Character(c) = &kev.logical_key {
            // `r` = roll a rando and audition its ring (mirrors the ☎ button).
            if c.eq_ignore_ascii_case("r") {
                self.audition_ring(ctx);
                return EventResponse::Handled;
            }
            if c.eq_ignore_ascii_case("n") {
                self.digest = *blake3::hash(&self.digest).as_bytes();
                self.samples_dirty = true;
                ctx.window.request_redraw();
                return EventResponse::Handled;
            }
            // `p` = play (mirrors Enter) — reachable with the left hand while the right drags sliders.
            if c.eq_ignore_ascii_case("p") {
                if self.samples_dirty {
                    self.rebuild_samples();
                }
                self.play();
                return EventResponse::Handled;
            }
            // `l` = log the current knobs to stdout as paste-ready param blocks.
            if c.eq_ignore_ascii_case("l") {
                let v = |i: usize| self.sliders[i].value();
                println!(
                    "BellParams {{ pitch: {:.3}, material: {:.3}, decay: {:.3}, slope: {:.3}, strike: {:.3}, inharm: {:.3}, shimmer: {:.3}, partials: {:.3}, clank: {:.3}, hum: {:.3}, fm: {:.3}, blip: {:.3} }}",
                    v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11)
                );
                println!(
                    "ChimeParams {{ spacing: {:.3}, spread: {:.3}, arp: {:.3} }}",
                    v(12), v(13), v(14)
                );
                let digest_hex: String =
                    self.digest.iter().map(|b| format!("{b:02x}")).collect();
                println!(
                    "PostParams {{ room: {:.3}, damp: {:.3}, echo_time: {:.3}, echo_gain: {:.3}, resonance: {:.3}, wet: {:.3}, ring: {:.3} }}  digest: {digest_hex}",
                    v(15), v(16), v(17), v(18), v(19), v(20), v(21)
                );
                return EventResponse::Handled;
            }
        }
        let Some(focus_id) = self.current_focus else {
            return EventResponse::Pass;
        };
        let mods = self.modifiers;
        let text = &mut *ctx.text;
        let response = widget::dispatch_key(self as &mut dyn Container, focus_id, kev, mods, text);
        if matches!(response, EventResponse::Handled) {
            if self.poll_slider_changes() {
                ctx.window.request_redraw();
            }
            self.poll_buttons(ctx);
        }
        response
    }

    /// Plotypus-style supersampled waveform of the RENDERED buffer (post chain visible): 32 stratified-jittered evals per column, per-row area coverage split pink/blue by sign with a green bucket for exact zeros (IEEE `== 0.0` matches both signed zeros), alpha = coverage/32.
    fn draw_waveform(&mut self, target: &mut [u32], buf_w: usize, damage: &mut fluor::canvas::Damage) {
        let (px, py, pw, ph) = self.plot_rect;
        if pw < 2 || ph < 2 {
            return;
        }
        // Topmost-first doctrine: zero line, then the area fill, then the black background LAST.
        let mut canvas = Canvas::new(target, buf_w, py + ph, damage);
        let mid_y = py as isize + (ph / 2) as isize;
        fluor::paint::fill_rect(&mut canvas, px as isize, mid_y, pw as isize, 0, ZERO_LINE, None, None);

        if self.samples.len() >= 2 {
            const SUBSAMPLES: usize = 32;
            let pink_rgb = PINK & 0x00FF_FFFF;
            let blue_rgb = BLUE & 0x00FF_FFFF;
            let zero_rgb = ZERO_FILL & 0x00FF_FFFF;
            let n = self.samples.len();
            let mut cov_pos = vec![0f32; ph];
            let mut cov_neg = vec![0f32; ph];
            let mut cov_zero = vec![0f32; ph];
            let bottom_row = py + ph;
            for col in 0..pw {
                cov_pos.iter_mut().for_each(|c| *c = 0.0);
                cov_neg.iter_mut().for_each(|c| *c = 0.0);
                cov_zero.iter_mut().for_each(|c| *c = 0.0);
                for ss in 0..SUBSAMPLES {
                    let frac = (ss as f64 + sample_jitter(col, ss)) / SUBSAMPLES as f64;
                    let pos = (col as f64 + frac) / pw as f64 * (n - 1) as f64;
                    let i0 = pos as usize;
                    let t = pos - i0 as f64;
                    let y = self.samples[i0] as f64 * (1.0 - t)
                        + self.samples[(i0 + 1).min(n - 1)] as f64 * t;
                    let curve_frac = ((y + 1.0) * 0.5) as f32;
                    let cov = if y == 0.0 {
                        &mut cov_zero
                    } else if y > 0.0 {
                        &mut cov_pos
                    } else {
                        &mut cov_neg
                    };
                    let curve_pix_y = py as f32 + (1.0 - curve_frac) * ph as f32;
                    let top_row = curve_pix_y as usize;
                    let top_cov = 1.0 - (curve_pix_y - curve_pix_y.floor());
                    if top_row >= py && top_row < bottom_row {
                        cov[top_row - py] += top_cov;
                    }
                    for r in (top_row + 1).max(py)..bottom_row {
                        cov[r - py] += 1.0;
                    }
                }
                let inv = 255.0 / SUBSAMPLES as f32;
                let abs_x = px + col;
                for i in 0..ph {
                    let idx = (py + i) * buf_w + abs_x;
                    let cp = cov_pos[i];
                    if cp > 0.0 {
                        let a = (cp * inv).min(255.0) as u32;
                        canvas.pixels[idx] = canvas.pixels[idx].under((a << 24) | pink_rgb, BlendMode::Normal);
                    }
                    let cn = cov_neg[i];
                    if cn > 0.0 {
                        let a = (cn * inv).min(255.0) as u32;
                        canvas.pixels[idx] = canvas.pixels[idx].under((a << 24) | blue_rgb, BlendMode::Normal);
                    }
                    let cz = cov_zero[i];
                    if cz > 0.0 {
                        let a = (cz * inv).min(255.0) as u32;
                        canvas.pixels[idx] = canvas.pixels[idx].under((a << 24) | zero_rgb, BlendMode::Normal);
                    }
                }
            }
        }
        fluor::paint::fill_rect(
            &mut canvas, px as isize, py as isize, pw as isize, ph as isize, PLOT_BG, None, None,
        );
    }
}

impl Container for TuneApp {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn Widget)) {
        // Row-wise: min, knob, max — so Tab walks each row left to right.
        for i in 0..self.sliders.len() {
            f(&mut self.lo_sliders[i]);
            f(&mut self.sliders[i]);
            f(&mut self.hi_sliders[i]);
        }
        f(&mut self.play_button);
        f(&mut self.random_button);
        f(&mut self.ranges_button);
        f(&mut self.ring_button);
        self.chrome.visit(f);
    }
}

impl FluorApp for TuneApp {
    type UserEvent = ();

    fn title(&self) -> &str {
        &self.title
    }

    fn init(&mut self, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
        self.update_layout(ctx);
        self.rebuild_samples();
    }

    fn on_resize(&mut self, _w: u32, _h: u32, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
        self.chrome.set_full_edge(ctx.is_maximized);
        self.update_layout(ctx);
    }

    fn on_event(&mut self, event: &Event, ctx: &mut Context) -> EventResponse {
        match event {
            Event::ModifiersChanged(m) => {
                self.modifiers = *m;
                EventResponse::Pass
            }

            Event::CursorMoved { .. } => {
                let x = ctx.cursor_x;
                let y = ctx.cursor_y;
                if let Some((vec_id, i)) = self.dragging {
                    match vec_id {
                        0 => self.sliders[i].set_value_from_x(x),
                        1 => self.lo_sliders[i].set_value_from_x(x),
                        _ => self.hi_sliders[i].set_value_from_x(x),
                    }
                    if self.poll_slider_changes() {
                        ctx.window.request_redraw();
                    }
                    return EventResponse::Handled;
                }
                let new_hit = self.chrome.hit_at(x, y);
                let mut changed = self.chrome.set_hover(new_hit);
                for btn in [&mut self.play_button, &mut self.random_button, &mut self.ranges_button, &mut self.ring_button] {
                    let want = new_hit == btn.hit_id();
                    if btn.is_hovered() != want {
                        btn.set_hovered(want);
                        changed = true;
                    }
                }
                if changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }

            Event::CursorLeft => {
                if self.chrome.set_hover(HIT_NONE) {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }

            Event::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
            } => {
                let x = ctx.cursor_x;
                let y = ctx.cursor_y;
                let hit_id = self.chrome.hit_at(x, y);

                if hit_id != HIT_NONE {
                    let mods = self.modifiers;
                    let response =
                        widget::dispatch_click(self as &mut dyn Container, hit_id, x, y, mods);
                    let mut focusable = false;
                    self.visit(&mut |w| {
                        if w.id() == hit_id && w.focus().is_some() {
                            focusable = true;
                        }
                    });
                    self.change_focus(if focusable { Some(hit_id) } else { None }, ctx);
                    self.dragging = self
                        .sliders
                        .iter()
                        .position(|s| s.hit_id() == hit_id)
                        .map(|i| (0, i))
                        .or_else(|| {
                            self.lo_sliders
                                .iter()
                                .position(|s| s.hit_id() == hit_id)
                                .map(|i| (1, i))
                        })
                        .or_else(|| {
                            self.hi_sliders
                                .iter()
                                .position(|s| s.hit_id() == hit_id)
                                .map(|i| (2, i))
                        });
                    if self.poll_slider_changes() {
                        ctx.window.request_redraw();
                    }
                    self.poll_buttons(ctx);
                    ctx.window.request_redraw();
                    return response;
                }

                let edge = chrome::get_resize_edge(ctx.viewport, x, y);
                if edge != ResizeEdge::None {
                    return EventResponse::StartResize(edge);
                }
                self.change_focus(None, ctx);
                EventResponse::StartWindowDrag
            }

            Event::MouseInput {
                state: ElementState::Released,
                ..
            } => {
                if self.dragging.take().is_some() {
                    return EventResponse::Handled;
                }
                EventResponse::Pass
            }

            Event::KeyboardInput { event: kev } => self.handle_key(kev, ctx),

            Event::Focused(focused) => {
                let mut redraw = self.chrome.set_focused(*focused);
                if !*focused && self.current_focus.is_some() {
                    self.change_focus(None, ctx);
                    redraw = true;
                }
                if redraw {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }

            _ => EventResponse::Pass,
        }
    }

    fn damage_rect(&mut self, viewport: Viewport) -> Option<PixelRect> {
        let w = viewport.width_px as usize;
        let h = viewport.height_px as usize;
        Some(PixelRect::new(0, 0, w, h))
    }

    fn hit_test_map(&self) -> Option<(&[HitId], usize, usize)> {
        let (w, h) = self.chrome.dims();
        Some((self.chrome.hit_test_map(), w, h))
    }

    fn overlay_deltas(&mut self) -> Vec<u32> {
        let count = self.hit_counter as usize + 1;
        widget::build_overlay_deltas(self, count)
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let buf_w = ctx.viewport.width_px as usize;
        let buf_h = ctx.viewport.height_px as usize;

        if self.samples_dirty {
            self.rebuild_samples();
        }

        // Direct assignment, NOT under(): the bg closure must fully overwrite the chrome's bg cache.
        self.chrome.rasterize_bg(ctx.damage, |canvas| {
            let px = fluor::paint::pack_argb(10, 14, 22, 255);
            canvas.pixels.fill(px);
        });
        self.chrome
            .rasterize_perimeter(target, buf_w, buf_h, ctx.clip_mask);
        self.chrome
            .rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);

        self.draw_waveform(target, buf_w, ctx.damage);

        // Sliders + label / value texts.
        let span = ctx.viewport.effective_span();
        let font_size = (span / 52.0).max(10.0);
        for i in 0..N_SLIDERS {
            for s in [&mut self.lo_sliders[i], &mut self.sliders[i], &mut self.hi_sliders[i]] {
                let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
                let id = s.hit_id();
                s.render_content_into(&mut canvas, Some(&mut self.chrome.hit_test_map), id);
            }
            let s = &self.sliders[i];
            let bbox = self.hi_sliders[i].bbox();
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_left(
                &mut canvas,
                SLIDER_LABELS[i],
                buf_w as Coord * 0.03,
                bbox.y + bbox.h * 0.5,
                &TextStyle::new(font_size, LABEL_COLOUR),
                None,
                None,
            );
            // Numerics: zone min left of its slider, zone max right of its slider, knob value at the far right.
            let zone_font = font_size * 0.85;
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_left(
                &mut canvas,
                &format!("{:.2}", self.lo_sliders[i].value()),
                buf_w as Coord * 0.105,
                bbox.y + bbox.h * 0.5,
                &TextStyle::new(zone_font, LABEL_COLOUR),
                None,
                None,
            );
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_left(
                &mut canvas,
                &format!("{:.2}", self.hi_sliders[i].value()),
                buf_w as Coord * 0.795,
                bbox.y + bbox.h * 0.5,
                &TextStyle::new(zone_font, LABEL_COLOUR),
                None,
                None,
            );
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_left(
                &mut canvas,
                &format!("{:.3}", s.value()),
                buf_w as Coord * 0.835,
                bbox.y + bbox.h * 0.5,
                &TextStyle::new(font_size, LABEL_COLOUR),
                None,
                None,
            );
        }

        {
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            let id = self.play_button.hit_id();
            self.play_button.render_content_into(
                &mut canvas,
                0.0,
                0.0,
                ctx.text,
                None,
                Some(&mut self.chrome.hit_test_map),
                id,
            );
        }
        {
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            let id = self.random_button.hit_id();
            self.random_button.render_content_into(
                &mut canvas,
                0.0,
                0.0,
                ctx.text,
                None,
                Some(&mut self.chrome.hit_test_map),
                id,
            );
        }
        {
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            let id = self.ranges_button.hit_id();
            self.ranges_button.render_content_into(
                &mut canvas,
                0.0,
                0.0,
                ctx.text,
                None,
                Some(&mut self.chrome.hit_test_map),
                id,
            );
        }
        {
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            let id = self.ring_button.hit_id();
            self.ring_button.render_content_into(
                &mut canvas,
                0.0,
                0.0,
                ctx.text,
                None,
                Some(&mut self.chrome.hit_test_map),
                id,
            );
        }

        // Composite the chrome group (bg fill, titlebar, window controls) UNDER everything painted above.
        self.chrome.flatten_into(target, buf_w, buf_h, None);
    }

    fn cursor_for(&self, x: Coord, y: Coord, ctx: &Context) -> CursorIcon {
        let hit = self.chrome.hit_at(x, y);
        if self.chrome.owns_hit(hit)
            || hit == self.play_button.hit_id()
            || hit == self.random_button.hit_id()
            || hit == self.ranges_button.hit_id()
            || hit == self.ring_button.hit_id()
            || self.sliders.iter().any(|s| s.hit_id() == hit)
            || self.lo_sliders.iter().any(|s| s.hit_id() == hit)
            || self.hi_sliders.iter().any(|s| s.hit_id() == hit)
        {
            return CursorIcon::Pointer;
        }
        match chrome::get_resize_edge(ctx.viewport, x, y) {
            ResizeEdge::Top | ResizeEdge::Bottom => CursorIcon::NsResize,
            ResizeEdge::Left | ResizeEdge::Right => CursorIcon::EwResize,
            ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorIcon::NwseResize,
            ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorIcon::NeswResize,
            ResizeEdge::None => CursorIcon::Default,
        }
    }
}

fn main() {
    fluor::host::app::run_app(TuneApp::new()).expect("event loop");
}
