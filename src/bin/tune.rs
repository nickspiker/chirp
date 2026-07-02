//! Chirp tuning UI — a fluor app for dialing in the voice formula by ear and eye.
//!
//! Plotypus-style layout: the waveform fills the top of the window (time x ∈ -1..1 left to right, amplitude y ∈ -1..1), a stack of lumis-style sliders (white left of the handle, black right) drives the seven voice knobs plus per-voice spread and clip length, and a play button renders + plays the current settings thru rodio.
//! Every slider edit re-renders the waveform immediately, so you see what you'll hear before pressing play.

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

/// Slider order: the seven formula knobs, then the per-voice random spread, then clip length.
const SLIDER_LABELS: [&str; 9] = [
    "fm depth", "fm rate", "phase", "shape", "bias", "attack", "decay", "spread", "length",
];
const N_SLIDERS: usize = SLIDER_LABELS.len();

/// Waveform fill colours, plotypus palette (darkness convention, opaque).
const PINK: u32 = 0xFF00_0000 | (0x00FF_8080 ^ 0x00FF_FFFF);
const BLUE: u32 = 0xFF00_0000 | (0x0080_80FF ^ 0x00FF_FFFF);
const PLOT_BG: u32 = 0xFFFF_FFFF; // black
const ZERO_LINE: u32 = 0xFF00_0000 | (0x0040_4040 ^ 0x00FF_FFFF);
const LABEL_COLOUR: u32 = 0xFF00_0000 | (0x00C0_C0C0 ^ 0x00FF_FFFF);

struct TuneApp {
    title: String,
    chrome: DefaultChrome,
    sliders: Vec<Slider>,
    play_button: Button,
    random_button: Button,
    /// UI-local RNG state for the random button (splitmix64 churn, time-seeded — no determinism needed here).
    rng: u64,
    hit_counter: HitId,
    current_focus: Option<HitId>,
    /// Index of the slider a left-press landed on; cursor moves drag its handle until release.
    dragging: Option<usize>,
    modifiers: fluor::event::ModifiersState,

    /// Rendered audio for the current slider settings — the play buffer.
    samples: Vec<f32>,
    /// The synth matching `samples`; the plot supersamples `synth.signal(x)` directly.
    synth: Option<chirp::Chirp>,
    samples_dirty: bool,
    /// Seed for the per-voice random jitter (spread slider scales it). Fixed per run; press `n` for a new one.
    seed: u64,

    plot_rect: (usize, usize, usize, usize),
    slider_area_top: Coord,
}

impl TuneApp {
    fn new() -> Self {
        let viewport = Viewport::new(1100, 760);
        let mut hit_counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(
            viewport,
            "Chirp Tune".to_string(),
            None,
            Some("ready".to_string()),
            &mut hit_counter,
        );

        let mut sliders = Vec::with_capacity(N_SLIDERS);
        for i in 0..N_SLIDERS {
            // Formula knobs default centred (param 0); spread defaults 0.5; length defaults ~0.9 s.
            let v = match SLIDER_LABELS[i] {
                "spread" => 0.5,
                "length" => 0.25,
                _ => 0.5,
            };
            sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, v));
        }
        let play_button = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "▶ play");
        let random_button = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "⚄ random");

        let rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5EED_5EED_5EED_5EED);

        Self {
            title: "Chirp Tune".to_string(),
            chrome,
            sliders,
            play_button,
            random_button,
            rng,
            hit_counter,
            current_focus: None,
            dragging: None,
            modifiers: fluor::event::ModifiersState::default(),
            samples: Vec::new(),
            synth: None,
            samples_dirty: true,
            seed: 0x5EED_C41B_9D0F_A7E3,
            plot_rect: (0, 0, 0, 0),
            slider_area_top: 0.0,
        }
    }

    /// Slider values → Chirp params. Knobs map 0..1 → -1..1; spread is direct; length maps 0..1 → 0.2..3.0 s.
    fn rebuild_samples(&mut self) {
        let mut base = [0.0f64; 7];
        for (j, b) in base.iter_mut().enumerate() {
            *b = self.sliders[j].value() as f64 * 2.0 - 1.0;
        }
        let spread = self.sliders[7].value() as f64;
        let duration = 0.2 + self.sliders[8].value() as f64 * 2.8;
        let synth = chirp::Chirp::from_params(base, spread, self.seed, duration);
        self.samples = synth.clone().collect();
        self.synth = Some(synth);
        self.samples_dirty = false;
    }

    /// Play the current buffer on a background thread (same rodio path as the playground bin).
    fn play(&self) {
        let samples = self.samples.clone();
        std::thread::spawn(move || {
            let Ok(handle) = rodio::DeviceSinkBuilder::open_default_sink() else {
                eprintln!("audio: no default output device");
                return;
            };
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
        // RU unit — drives only visual WEIGHT (handle diameter / track thickness / font / button size).
        // All POSITIONS below are fixed fractions of the viewport so resizing or RU changes never move the rows.
        let bw = span / 32.0;

        // Vertical structure, top to bottom, as fractions of the window height:
        // chrome bar (chrome's own), plot 8%..52%, slider stack 55%..90%, button row at 94%.
        let margin = w * 0.03;
        let plot_top = h * 0.08;
        let plot_bottom = h * 0.52;
        self.plot_rect = (
            margin as usize,
            plot_top as usize,
            (w - margin * 2.0).max(8.0) as usize,
            (plot_bottom - plot_top).max(8.0) as usize,
        );
        self.slider_area_top = h * 0.55;
        let slider_area_h = h * 0.35;
        let row_pitch = slider_area_h / N_SLIDERS as Coord;
        // Grab-band height (= handle diameter) is RU-derived but can't exceed the row pitch.
        let band_h = (bw * 1.2).max(12.0).min(row_pitch * 0.9);

        // Horizontal structure, as fractions of the width: labels 3%..16%, slider 17%..82%, value at 84%.
        let slider_left = w * 0.17;
        let slider_right = w * 0.82;
        let slider_w = slider_right - slider_left;
        let slider_cx = slider_left + slider_w * 0.5;
        for (i, s) in self.sliders.iter_mut().enumerate() {
            let cy = self.slider_area_top + (i as Coord + 0.5) * row_pitch;
            s.set_rect(slider_cx, cy, slider_w, band_h);
        }

        let btn_w = bw * 7.0;
        let btn_h = bw * 1.7;
        let btn_cy = h * 0.94;
        self.play_button
            .set_rect(w - margin - btn_w * 0.5, btn_cy, btn_w, btn_h);
        self.play_button.set_font_size(bw * 0.9);
        self.random_button
            .set_rect(w - margin - btn_w * 1.5 - bw * 0.5, btn_cy, btn_w, btn_h);
        self.random_button.set_font_size(bw * 0.9);
    }

    /// Next uniform in `0..1` from the UI RNG (splitmix64 churn).
    fn next_rand(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Poll both buttons; fires their actions. Called after any dispatch that could have clicked them.
    fn poll_buttons(&mut self, ctx: &mut Context) {
        if self.random_button.take_click() {
            // Randomize the seven formula knobs across their full -1..1 range (slider 0..1 space).
            for i in 0..7 {
                let v = self.next_rand();
                self.sliders[i].set_value(v);
            }
            self.poll_slider_changes();
            ctx.window.request_redraw();
        }
        if self.play_button.take_click() {
            if self.samples_dirty {
                self.rebuild_samples();
            }
            self.play();
        }
    }


    /// Poll every slider's change counter; mark the waveform dirty if any moved.
    fn poll_slider_changes(&mut self) -> bool {
        let mut changed = false;
        for s in &mut self.sliders {
            if s.take_change() {
                changed = true;
            }
        }
        if changed {
            self.samples_dirty = true;
        }
        changed
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
        // Tab / Shift+Tab focus cycle.
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
        // Enter / Space anywhere = play (unless a slider has focus and eats the key first below).
        if matches!(kev.logical_key, Key::Named(NamedKey::Enter)) {
            if self.samples_dirty {
                self.rebuild_samples();
            }
            self.play();
            return EventResponse::Handled;
        }
        // `n` = roll a new jitter seed (audible when spread > 0).
        if let Key::Character(c) = &kev.logical_key {
            if c.eq_ignore_ascii_case("n") {
                self.seed = self
                    .seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(0xD1B5_4A32_D192_ED03);
                self.samples_dirty = true;
                ctx.window.request_redraw();
                return EventResponse::Handled;
            }
        }
        // Deliver to the focused widget (slider arrows, button enter/space).
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

    /// Column min/max waveform, pink above the zero line and blue below, plotypus palette.
    fn draw_waveform(&mut self, target: &mut [u32], buf_w: usize, damage: &mut fluor::canvas::Damage) {
        let (px, py, pw, ph) = self.plot_rect;
        if pw < 2 || ph < 2 {
            return;
        }
        // Topmost-first doctrine: zero line, then the supersampled area fill, then the black background LAST — under() early-outs where anything above already claimed the pixel, so painting the bg first would hide everything.
        let mut canvas = Canvas::new(target, buf_w, py + ph, damage);
        let mid_y = py as isize + (ph / 2) as isize;
        fluor::paint::fill_rect(
            &mut canvas,
            px as isize,
            mid_y,
            pw as isize,
            0,
            ZERO_LINE,
            None,
            None,
        );
        // Stratified-jittered supersampling, same as plotypus's draw_curve: SUBSAMPLES jittered x positions per column evaluated straight from the synth, accumulating per-row area coverage split by sign, alpha = coverage/SUBSAMPLES.
        // The jitter breaks the moiré a regular sample grid makes against the high-frequency voices; deterministic per (column, subsample) so the plot is stable frame to frame.
        if let Some(synth) = &self.synth {
            const SUBSAMPLES: usize = 32;
            let pink_rgb = PINK & 0x00FF_FFFF;
            let blue_rgb = BLUE & 0x00FF_FFFF;
            let mut cov_pos = vec![0f32; ph];
            let mut cov_neg = vec![0f32; ph];
            let bottom_row = py + ph;
            for col in 0..pw {
                cov_pos.iter_mut().for_each(|c| *c = 0.0);
                cov_neg.iter_mut().for_each(|c| *c = 0.0);
                for ss in 0..SUBSAMPLES {
                    let frac_in_col =
                        (ss as f64 + sample_jitter(col, ss)) / SUBSAMPLES as f64;
                    let wx = -1.0 + 2.0 * (col as f64 + frac_in_col) / pw as f64;
                    let y = synth.signal(wx);
                    // y ∈ -1..1 → fraction of plot height from the bottom.
                    let curve_frac = ((y + 1.0) * 0.5) as f32;
                    let cov = if y >= 0.0 { &mut cov_pos } else { &mut cov_neg };
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
                        canvas.pixels[idx] =
                            canvas.pixels[idx].under((a << 24) | pink_rgb, BlendMode::Normal);
                    }
                    let cn = cov_neg[i];
                    if cn > 0.0 {
                        let a = (cn * inv).min(255.0) as u32;
                        canvas.pixels[idx] =
                            canvas.pixels[idx].under((a << 24) | blue_rgb, BlendMode::Normal);
                    }
                }
            }
        }
        fluor::paint::fill_rect(
            &mut canvas,
            px as isize,
            py as isize,
            pw as isize,
            ph as isize,
            PLOT_BG,
            None,
            None,
        );
    }
}

impl Container for TuneApp {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn Widget)) {
        for s in &mut self.sliders {
            f(s);
        }
        f(&mut self.play_button);
        f(&mut self.random_button);
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
                // Active slider drag: feed cursor x to the pressed slider until release.
                if let Some(i) = self.dragging {
                    self.sliders[i].set_value_from_x(x);
                    if self.poll_slider_changes() {
                        ctx.window.request_redraw();
                    }
                    return EventResponse::Handled;
                }
                let new_hit = self.chrome.hit_at(x, y);
                let mut changed = self.chrome.set_hover(new_hit);
                // Sliders have no hover visual (tint_delta 0), so no bookkeeping needed for them.
                let want_btn = new_hit == self.play_button.hit_id();
                if self.play_button.is_hovered() != want_btn {
                    self.play_button.set_hovered(want_btn);
                    changed = true;
                }
                let want_rnd = new_hit == self.random_button.hit_id();
                if self.random_button.is_hovered() != want_rnd {
                    self.random_button.set_hovered(want_rnd);
                    changed = true;
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
                    // Arm slider dragging if the press landed on one.
                    self.dragging = self.sliders.iter().position(|s| s.hit_id() == hit_id);
                    if self.poll_slider_changes() {
                        ctx.window.request_redraw();
                    }
                    self.poll_buttons(ctx);
                    ctx.window.request_redraw();
                    return response;
                }

                let edge =
                    chrome::get_resize_edge(ctx.viewport.width_px, ctx.viewport.height_px, x, y);
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

    fn damage_rect(&self, viewport: Viewport) -> Option<PixelRect> {
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

        // Direct assignment, NOT under(): the bg closure must fully overwrite the chrome's bg cache (same as plotypus's starfield). An under()-fill leaves stale alpha and the window goes transparent.
        self.chrome.rasterize_bg(ctx.damage, |canvas| {
            let px = fluor::paint::pack_argb(10, 14, 22, 255);
            canvas.pixels.fill(px);
        });
        self.chrome
            .rasterize_perimeter(target, buf_w, buf_h, ctx.clip_mask);
        self.chrome
            .rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);

        self.draw_waveform(target, buf_w, ctx.damage);

        // Sliders + their label / value texts.
        let span = ctx.viewport.effective_span();
        let font_size = (span / 48.0).max(10.0);
        for i in 0..N_SLIDERS {
            {
                let s = &mut self.sliders[i];
                let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
                let id = s.hit_id();
                s.render_content_into(
                    &mut canvas,
                    Some(&mut self.chrome.hit_test_map),
                    id,
                );
            }
            let s = &self.sliders[i];
            let bbox = s.bbox();
            let label_x = buf_w as Coord * 0.03;
            let value = match SLIDER_LABELS[i] {
                "spread" => s.value(),
                "length" => 0.2 + s.value() * 2.8,
                _ => s.value() * 2.0 - 1.0,
            };
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_left_u32(
                &mut canvas,
                SLIDER_LABELS[i],
                label_x,
                bbox.y + bbox.h * 0.5,
                font_size,
                400,
                LABEL_COLOUR,
                "Open Sans",
                None,
                None,
                None,
            );
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_left_u32(
                &mut canvas,
                &format!("{value:+.3}"),
                bbox.x + bbox.w + font_size,
                bbox.y + bbox.h * 0.5,
                font_size,
                400,
                LABEL_COLOUR,
                "Open Sans",
                None,
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

        // Composite the chrome group (bg fill, titlebar, window controls) UNDER everything painted above. Without this the window has no background at all.
        self.chrome.flatten_into(target, buf_w, buf_h, None);
    }

    fn cursor_for(&self, x: Coord, y: Coord, ctx: &Context) -> CursorIcon {
        let hit = self.chrome.hit_at(x, y);
        if self.chrome.owns_hit(hit)
            || hit == self.play_button.hit_id()
            || hit == self.random_button.hit_id()
        {
            return CursorIcon::Pointer;
        }
        if self.sliders.iter().any(|s| s.hit_id() == hit) {
            return CursorIcon::Pointer;
        }
        match chrome::get_resize_edge(ctx.viewport.width_px, ctx.viewport.height_px, x, y) {
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
