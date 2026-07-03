//! Chirp tuning UI — full-resolution 135-slider voice grid + 4 global controls.
//!
//! Voice grid: 9 identities × 3 pitch layers × 5 params (phase, shape, bias, atk, dec) = 135.
//! Global row: fm·depth, fm·rate (shared sweep), base freq (one octave, exp-mapped), length.
//! Random fills all 139 sliders independently; every edit re-renders the waveform.

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

const N_IDENTITIES: usize = 9;
const N_PITCHES: usize = 3;
const N_PARAMS: usize = 5;
const N_ROWS: usize = N_IDENTITIES * N_PITCHES; // 27
const N_VOICE_SLIDERS: usize = N_ROWS * N_PARAMS; // 135

// Global slider indices after the voice grid.
const GLOB_FM_DEPTH: usize = N_VOICE_SLIDERS;     // 135
const GLOB_FM_RATE: usize  = N_VOICE_SLIDERS + 1; // 136
const GLOB_BASE:    usize  = N_VOICE_SLIDERS + 2; // 137
const GLOB_LENGTH:  usize  = N_VOICE_SLIDERS + 3; // 138
const N_SLIDERS:    usize  = N_VOICE_SLIDERS + 4; // 139

const PARAM_LABELS: [&str; N_PARAMS] = ["phase", "shape", "bias", "atk", "dec"];
const PITCH_NAMES:  [&str; N_PITCHES] = ["3×", "4×", "5×"];
const GLOB_LABELS:  [&str; 4] = ["fm·d", "fm·r", "base", "len"];

const PINK:         u32 = 0xFF00_0000 | (0x00FF_8080 ^ 0x00FF_FFFF);
const BLUE:         u32 = 0xFF00_0000 | (0x0080_80FF ^ 0x00FF_FFFF);
const PLOT_BG:      u32 = 0xFFFF_FFFF;
const ZERO_LINE:    u32 = 0xFF00_0000 | (0x0040_4040 ^ 0x00FF_FFFF);
const LABEL_COLOUR: u32 = 0xFF00_0000 | (0x00C0_C0C0 ^ 0x00FF_FFFF);
const DIM_LABEL:    u32 = 0xFF00_0000 | (0x00E0_E0E0 ^ 0x00FF_FFFF);

#[inline]
fn slider_idx(k: usize, p: usize, j: usize) -> usize {
    k * (N_PITCHES * N_PARAMS) + p * N_PARAMS + j
}

struct TuneApp {
    title: String,
    chrome: DefaultChrome,
    sliders: Vec<Slider>,
    play_button: Button,
    random_button: Button,
    rng: u64,
    hit_counter: HitId,
    current_focus: Option<HitId>,
    dragging: Option<usize>,
    modifiers: fluor::event::ModifiersState,

    samples: Vec<f32>,
    synth: Option<chirp::Chirp>,
    samples_dirty: bool,

    // Layout cache.
    plot_rect: (usize, usize, usize, usize),
    header_y: Coord,
    grid_top: Coord,
    row_pitch: Coord,
    col_cx: [Coord; N_PARAMS],
    label_right: Coord,
    glob_y: Coord,
    glob_cx: [Coord; 4],
}

impl TuneApp {
    fn new() -> Self {
        let viewport = Viewport::new(1100, 960);
        let mut hit_counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(
            viewport,
            "Chirp Tune".to_string(),
            None,
            Some("ready".to_string()),
            &mut hit_counter,
        );

        let mut sliders = Vec::with_capacity(N_SLIDERS);
        for _ in 0..N_VOICE_SLIDERS {
            sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 0.5));
        }
        // fm·depth, fm·rate default centred; base default 0 (= 1.0× = base pitch); length default 0.25.
        sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 0.5)); // fm·d
        sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 0.5)); // fm·r
        sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 0.0)); // base (1× = no shift)
        sliders.push(Slider::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 0.25)); // length

        let play_button   = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 14.0, "▶ play");
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
            plot_rect: (0, 0, 0, 0),
            header_y: 0.0,
            grid_top: 0.0,
            row_pitch: 0.0,
            col_cx: [0.0; N_PARAMS],
            label_right: 0.0,
            glob_y: 0.0,
            glob_cx: [0.0; 4],
        }
    }

    fn rebuild_samples(&mut self) {
        let fm_depth = self.sliders[GLOB_FM_DEPTH].value() as f64 * 2.0 - 1.0;
        let fm_rate  = self.sliders[GLOB_FM_RATE].value()  as f64 * 2.0 - 1.0;
        let sweep    = [fm_depth, fm_rate];
        // base slider 0..1 → base_freq 1.0..2.0 via 2^v (perceptually linear octave).
        let base_freq = 2.0f64.powf(self.sliders[GLOB_BASE].value() as f64);
        let duration  = 0.2 + self.sliders[GLOB_LENGTH].value() as f64 * 2.8;

        let mut params = [[[0.0f64; N_PARAMS]; N_PITCHES]; N_IDENTITIES];
        for k in 0..N_IDENTITIES {
            for p in 0..N_PITCHES {
                for j in 0..N_PARAMS {
                    params[k][p][j] =
                        self.sliders[slider_idx(k, p, j)].value() as f64 * 2.0 - 1.0;
                }
            }
        }

        let synth = chirp::Chirp::from_raw(params, sweep, base_freq, duration);
        self.samples = synth.clone().collect();
        self.synth = Some(synth);
        self.samples_dirty = false;
    }

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
        let bw = span / 32.0;
        let margin = w * 0.02;

        // Waveform plot: 7%..24%.
        let plot_top    = h * 0.07;
        let plot_bottom = h * 0.24;
        self.plot_rect = (
            margin as usize,
            plot_top as usize,
            (w - margin * 2.0).max(8.0) as usize,
            (plot_bottom - plot_top).max(8.0) as usize,
        );

        // Column header row.
        self.header_y = h * 0.26;

        // Voice slider grid: 28%..88%, 27 rows.
        self.grid_top = h * 0.28;
        let grid_bottom = h * 0.88;
        self.row_pitch = (grid_bottom - self.grid_top) / N_ROWS as Coord;
        let band_h = (self.row_pitch * 0.72).max(6.0);

        // Horizontal split: labels 2%..10%, sliders 11%..98%.
        self.label_right = w * 0.10;
        let slider_left  = w * 0.11;
        let slider_right = w * 0.98;
        let col_w = (slider_right - slider_left) / N_PARAMS as Coord;
        for j in 0..N_PARAMS {
            self.col_cx[j] = slider_left + (j as Coord + 0.5) * col_w;
        }

        for k in 0..N_IDENTITIES {
            for p in 0..N_PITCHES {
                let row = k * N_PITCHES + p;
                let cy = self.grid_top + (row as Coord + 0.5) * self.row_pitch;
                for j in 0..N_PARAMS {
                    self.sliders[slider_idx(k, p, j)].set_rect(
                        self.col_cx[j], cy, col_w * 0.95, band_h,
                    );
                }
            }
        }

        // Global row: 90%.
        self.glob_y = h * 0.90;
        let glob_left  = w * 0.11;
        let glob_right = w * 0.98;
        let glob_w = (glob_right - glob_left) / 4.0;
        for i in 0..4 {
            self.glob_cx[i] = glob_left + (i as Coord + 0.5) * glob_w;
            self.sliders[N_VOICE_SLIDERS + i].set_rect(
                self.glob_cx[i], self.glob_y, glob_w * 0.90, band_h,
            );
        }

        // Buttons at 94%.
        let btn_w  = bw * 7.0;
        let btn_h  = bw * 1.7;
        let btn_cy = h * 0.955;
        self.play_button.set_rect(w - margin - btn_w * 0.5, btn_cy, btn_w, btn_h);
        self.play_button.set_font_size(bw * 0.9);
        self.random_button.set_rect(w - margin - btn_w * 1.5 - bw * 0.5, btn_cy, btn_w, btn_h);
        self.random_button.set_font_size(bw * 0.9);
    }

    fn next_rand(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }

    fn randomize_all(&mut self) {
        // Randomize all 135 voice sliders + fm·d, fm·r, base. Length stays untouched.
        for i in 0..(N_VOICE_SLIDERS + 3) {
            let v = self.next_rand();
            self.sliders[i].set_value(v);
        }
        self.poll_slider_changes();
    }

    fn poll_buttons(&mut self, ctx: &mut Context) {
        if self.random_button.take_click() {
            self.randomize_all();
            ctx.window.request_redraw();
        }
        if self.play_button.take_click() {
            if self.samples_dirty { self.rebuild_samples(); }
            self.play();
        }
    }

    fn poll_slider_changes(&mut self) -> bool {
        let mut changed = false;
        for s in &mut self.sliders {
            if s.take_change() { changed = true; }
        }
        if changed { self.samples_dirty = true; }
        changed
    }

    fn change_focus(&mut self, new_focus: Option<HitId>, ctx: &mut Context) {
        if new_focus == self.current_focus { return; }
        let prior = self.current_focus;
        widget::apply_focus_change(self as &mut dyn Container, prior, new_focus);
        self.current_focus = new_focus;
        ctx.window.request_redraw();
    }

    fn handle_key(&mut self, kev: &KeyEvent, ctx: &mut Context) -> EventResponse {
        if kev.state != ElementState::Pressed { return EventResponse::Pass; }

        if matches!(kev.logical_key, Key::Named(NamedKey::Tab)) {
            let dir = if self.modifiers.shift_key() { TabDir::Backward } else { TabDir::Forward };
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
            if self.samples_dirty { self.rebuild_samples(); }
            self.play();
            return EventResponse::Handled;
        }
        if let Key::Character(c) = &kev.logical_key {
            if c.eq_ignore_ascii_case("r") {
                self.randomize_all();
                ctx.window.request_redraw();
                return EventResponse::Handled;
            }
        }

        let Some(focus_id) = self.current_focus else { return EventResponse::Pass; };
        let mods = self.modifiers;
        let response = widget::dispatch_key(self as &mut dyn Container, focus_id, kev, mods, &mut *ctx.text);
        if matches!(response, EventResponse::Handled) {
            if self.poll_slider_changes() { ctx.window.request_redraw(); }
            self.poll_buttons(ctx);
        }
        response
    }

    fn draw_waveform(&mut self, target: &mut [u32], buf_w: usize, damage: &mut fluor::canvas::Damage) {
        let (px, py, pw, ph) = self.plot_rect;
        if pw < 2 || ph < 2 { return; }
        let mut canvas = Canvas::new(target, buf_w, py + ph, damage);
        let mid_y = py as isize + (ph / 2) as isize;
        fluor::paint::fill_rect(&mut canvas, px as isize, mid_y, pw as isize, 0, ZERO_LINE, None, None);

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
                    let frac = (ss as f64 + sample_jitter(col, ss)) / SUBSAMPLES as f64;
                    let wx = -1.0 + 2.0 * (col as f64 + frac) / pw as f64;
                    let y = synth.signal(wx);
                    let curve_frac = ((y + 1.0) * 0.5) as f32;
                    let cov = if y >= 0.0 { &mut cov_pos } else { &mut cov_neg };
                    let curve_pix_y = py as f32 + (1.0 - curve_frac) * ph as f32;
                    let top_row = curve_pix_y as usize;
                    let top_cov = 1.0 - (curve_pix_y - curve_pix_y.floor());
                    if top_row >= py && top_row < bottom_row { cov[top_row - py] += top_cov; }
                    for r in (top_row + 1).max(py)..bottom_row { cov[r - py] += 1.0; }
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
        for s in &mut self.sliders { f(s); }
        f(&mut self.play_button);
        f(&mut self.random_button);
        self.chrome.visit(f);
    }
}

impl FluorApp for TuneApp {
    type UserEvent = ();

    fn title(&self) -> &str { &self.title }

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
            Event::ModifiersChanged(m) => { self.modifiers = *m; EventResponse::Pass }

            Event::CursorMoved { .. } => {
                let x = ctx.cursor_x;
                let y = ctx.cursor_y;
                if let Some(i) = self.dragging {
                    self.sliders[i].set_value_from_x(x);
                    if self.poll_slider_changes() { ctx.window.request_redraw(); }
                    return EventResponse::Handled;
                }
                let new_hit = self.chrome.hit_at(x, y);
                let mut changed = self.chrome.set_hover(new_hit);
                let want_play = new_hit == self.play_button.hit_id();
                if self.play_button.is_hovered() != want_play {
                    self.play_button.set_hovered(want_play); changed = true;
                }
                let want_rnd = new_hit == self.random_button.hit_id();
                if self.random_button.is_hovered() != want_rnd {
                    self.random_button.set_hovered(want_rnd); changed = true;
                }
                if changed { ctx.window.request_redraw(); }
                EventResponse::Pass
            }

            Event::CursorLeft => {
                if self.chrome.set_hover(HIT_NONE) { ctx.window.request_redraw(); }
                EventResponse::Pass
            }

            Event::MouseInput { state: ElementState::Pressed, button: MouseButton::Left } => {
                let x = ctx.cursor_x;
                let y = ctx.cursor_y;
                let hit_id = self.chrome.hit_at(x, y);
                if hit_id != HIT_NONE {
                    let mods = self.modifiers;
                    let response = widget::dispatch_click(self as &mut dyn Container, hit_id, x, y, mods);
                    let mut focusable = false;
                    self.visit(&mut |w| {
                        if w.id() == hit_id && w.focus().is_some() { focusable = true; }
                    });
                    self.change_focus(if focusable { Some(hit_id) } else { None }, ctx);
                    self.dragging = self.sliders.iter().position(|s| s.hit_id() == hit_id);
                    if self.poll_slider_changes() { ctx.window.request_redraw(); }
                    self.poll_buttons(ctx);
                    ctx.window.request_redraw();
                    return response;
                }
                let edge = chrome::get_resize_edge(ctx.viewport.width_px, ctx.viewport.height_px, x, y);
                if edge != ResizeEdge::None { return EventResponse::StartResize(edge); }
                self.change_focus(None, ctx);
                EventResponse::StartWindowDrag
            }

            Event::MouseInput { state: ElementState::Released, .. } => {
                if self.dragging.take().is_some() { return EventResponse::Handled; }
                EventResponse::Pass
            }

            Event::KeyboardInput { event: kev } => self.handle_key(kev, ctx),

            Event::Focused(focused) => {
                let mut redraw = self.chrome.set_focused(*focused);
                if !*focused && self.current_focus.is_some() {
                    self.change_focus(None, ctx); redraw = true;
                }
                if redraw { ctx.window.request_redraw(); }
                EventResponse::Pass
            }

            _ => EventResponse::Pass,
        }
    }

    fn damage_rect(&self, viewport: Viewport) -> Option<PixelRect> {
        Some(PixelRect::new(0, 0, viewport.width_px as usize, viewport.height_px as usize))
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

        if self.samples_dirty { self.rebuild_samples(); }

        self.chrome.rasterize_bg(ctx.damage, |canvas| {
            canvas.pixels.fill(fluor::paint::pack_argb(10, 14, 22, 255));
        });
        self.chrome.rasterize_perimeter(target, buf_w, buf_h, ctx.clip_mask);
        self.chrome.rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);

        self.draw_waveform(target, buf_w, ctx.damage);

        let span = ctx.viewport.effective_span();
        let font_size = (span / 60.0).max(8.0);

        // Column headers.
        for j in 0..N_PARAMS {
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            ctx.text.draw_text_center_u32(
                &mut canvas, PARAM_LABELS[j], self.col_cx[j], self.header_y,
                font_size, 400, LABEL_COLOUR, "Open Sans", None, None, None,
            );
        }

        // Voice grid.
        for k in 0..N_IDENTITIES {
            for p in 0..N_PITCHES {
                let row = k * N_PITCHES + p;
                let cy = self.grid_top + (row as Coord + 0.5) * self.row_pitch;

                let colour = if p == 0 { LABEL_COLOUR } else { DIM_LABEL };
                let label = if p == 0 {
                    format!("{}·{}", k + 1, PITCH_NAMES[p])
                } else {
                    format!("  {}", PITCH_NAMES[p])
                };
                let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
                ctx.text.draw_text_right_u32(
                    &mut canvas, &label, self.label_right, cy,
                    font_size, 400, colour, "Open Sans", None, None, None,
                );

                for j in 0..N_PARAMS {
                    let idx = slider_idx(k, p, j);
                    let s = &mut self.sliders[idx];
                    let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
                    let id = s.hit_id();
                    s.render_content_into(&mut canvas, Some(&mut self.chrome.hit_test_map), id);
                }
            }
        }

        // Global row: 4 sliders with labels above.
        let glob_label_y = self.glob_y - self.row_pitch * 0.6;
        for i in 0..4 {
            {
                let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
                ctx.text.draw_text_center_u32(
                    &mut canvas, GLOB_LABELS[i], self.glob_cx[i], glob_label_y,
                    font_size, 400, LABEL_COLOUR, "Open Sans", None, None, None,
                );
            }
            let s = &mut self.sliders[N_VOICE_SLIDERS + i];
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            let id = s.hit_id();
            s.render_content_into(&mut canvas, Some(&mut self.chrome.hit_test_map), id);
        }

        // Buttons.
        for (btn, offset_x, offset_y) in [
            (&mut self.play_button as *mut Button, 0.0f32, 0.0f32),
            (&mut self.random_button as *mut Button, 0.0f32, 0.0f32),
        ] {
            let btn = unsafe { &mut *btn };
            let mut canvas = Canvas::new(target, buf_w, buf_h, ctx.damage);
            let id = btn.hit_id();
            btn.render_content_into(
                &mut canvas, offset_x, offset_y, ctx.text, None,
                Some(&mut self.chrome.hit_test_map), id,
            );
        }

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
