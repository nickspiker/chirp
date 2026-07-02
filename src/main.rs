mod sound;

use rodio::{DeviceSinkBuilder, Player};
use std::fs;
use std::num::NonZero;

fn main() {
    let samples = sound::samples();

    // Play it.
    let handle = DeviceSinkBuilder::open_default_sink().expect("open default audio device");
    let player = Player::connect_new(&handle.mixer());
    let buf = rodio::buffer::SamplesBuffer::new(
        NonZero::new(1).unwrap(),
        NonZero::new(sound::SAMPLE_RATE).unwrap(),
        samples.clone(),
    );
    player.append(buf);

    // Write SVG + HTML viewer.
    let svg = chirp::samples_to_svg(&samples);
    fs::write("chirp.svg", &svg).expect("write chirp.svg");

    let html = r#"<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>chirp</title>
<style>
  * { margin: 0; padding: 0; }
  body { background: black; display: flex; align-items: center; justify-content: center; height: 100vh; }
  img { width: 100%; max-height: 100vh; object-fit: contain; }
</style>
</head>
<body>
<img id="w" src="chirp.svg">
<script>setInterval(() => { document.getElementById("w").src = "chirp.svg?" + Date.now(); }, 500);</script>
</body>
</html>"#;
    fs::write("chirp.html", html).expect("write chirp.html");

    let duration_ms = samples.len() as f32 / sound::SAMPLE_RATE as f32 * 1000.0;
    println!(
        "wrote chirp.html — {:.0}ms, {} samples",
        duration_ms,
        samples.len(),
    );

    player.sleep_until_end();
}
