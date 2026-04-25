//! Write an SVG waveform for each command-line argument.
//!
//! Run with `cargo run --example plot -- hello world build-finished`.
//! Produces files like `hello.svg`, `world.svg`, etc.

use std::env;
use std::fs;

use chirp::Chirp;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let labels: Vec<&str> = if args.is_empty() {
        vec!["chirp"]
    } else {
        args.iter().map(String::as_str).collect()
    };

    for label in labels {
        let svg = Chirp::from_bytes(label.as_bytes()).to_svg();
        let path = format!("{label}.svg");
        fs::write(&path, &svg).expect("write svg");
        println!("wrote {path}");
    }
}
