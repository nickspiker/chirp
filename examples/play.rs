//! Play a chirp for each command-line argument.
//!
//! Run with `cargo run --example play -- hello world build-finished`.

use std::env;

use chirp::Chirp;
use rodio::{DeviceSinkBuilder, Player};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let labels: Vec<&str> = if args.is_empty() {
        vec!["chirp"]
    } else {
        args.iter().map(String::as_str).collect()
    };

    let handle = DeviceSinkBuilder::open_default_sink().expect("open default audio device");
    let player = Player::connect_new(&handle.mixer());

    for label in labels {
        println!("playing chirp for {label:?}");
        player.append(Chirp::from_bytes(label.as_bytes()));
    }

    player.sleep_until_end();
}
