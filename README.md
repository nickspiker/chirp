# chirp

Deterministic procedural notification sounds for Rust. A `Chirp` is a short,
sine-swept tone synthesized from a 256-bit hash — the same hash always
produces the same sound, and different hashes produce audibly distinct ones.
Useful for per-event UI cues where you want the *kind* of event to be
recognizable from the sound itself, without curating a sound bank.

## Usage

```rust
use chirp::Chirp;

// From a 256-bit hash:
let source = Chirp::from_hash([0xde, 0xad, 0xbe, 0xef, /* ...28 more bytes */]);

// Or from arbitrary bytes (e.g. an event id):
let source = Chirp::from_bytes(b"build-finished");

// Render the waveform as an SVG:
let svg = Chirp::from_bytes(b"build-finished").to_svg();
```

`Chirp` implements [`rodio::Source`], so you can pipe it straight into a
rodio player or mixer. See `examples/play.rs`:

```bash
cargo run --example play -- hello world build-finished
cargo run --example plot -- hello world build-finished
```

## Status

Early. The hash → sound mapping is intentionally simple (frequency sweep
with exponential decay) and may evolve before 1.0.

## License

MIT OR Apache-2.0
