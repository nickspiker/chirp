//! A wave rings twice and a beam three times in the same span (Nick 2026-10-31): the cadences differ in count, never in length.

#[test]
fn a_beam_ring_spans_the_wave_ring() {
    let h = [7u8; 32];
    let wave = chirp::Chirp::ring_from_hash(h).samples().len() as i64;
    let beam = chirp::Chirp::beam_ring_from_hash(h).samples().len() as i64;
    // Rounding each burst and gap to whole samples can differ by a few samples, never by a burst.
    assert!((wave - beam).abs() <= 1 << 3, "wave ring {wave} samples, beam ring {beam}");
}
