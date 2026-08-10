//! Is decoding a frame bound by the disk or by the CPU?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test decode_cost -- --ignored --nocapture
//! ```
//!
//! Load-bearing for the optimisation plan. "605 ms for a 300 MB frame must be CPU, because
//! the SSD is faster than that" is an *inference* from a spec sheet, and if it is wrong then
//! parallelising the decode buys nothing and the registration restructuring loses most of its
//! payoff. So it is measured two ways rather than argued:
//!
//! 1. **The split.** Raw bytes off disk versus the full decode of the same file, warm, so
//!    both see identical cache conditions. The difference is u16 -> f32 conversion and strip
//!    handling.
//! 2. **The scaling**, which is what actually decides the plan. Eight frames decoded
//!    sequentially against eight *different* frames decoded concurrently, both cold. Cache
//!    state is the obvious confound in a test like this, so the two groups are disjoint and
//!    neither is touched beforehand — a warm-up would measure memory bandwidth and quietly
//!    flatter the parallel case.

use std::path::Path;
use std::time::{Duration, Instant};

use stackaroni_core::discovery::discover_stack;
use stackaroni_core::pipeline::Image;

fn decode_fully(path: &Path) -> usize {
    let image = Image::open(path).unwrap();
    let info = image.info();
    let mut band = vec![0f32; info.row_len() * 64];
    let (mut y, mut touched) = (0, 0usize);
    while y < info.height {
        let count = 64.min(info.height - y);
        image
            .read_rows(y, count, &mut band[..info.row_len() * count as usize])
            .unwrap();
        touched += count as usize;
        y += count;
    }
    touched
}

fn rate(bytes: u64, elapsed: Duration) -> String {
    format!(
        "{:>7.0?}  ({:.0} MB/s)",
        elapsed,
        bytes as f64 / 1e6 / elapsed.as_secs_f64()
    )
}

#[test]
#[ignore = "requires test-data/blossom, run with --release"]
fn decode_is_cpu_bound_and_scales() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/blossom");
    let Ok(stack) = discover_stack(&dir) else {
        eprintln!("skipping: test-data/blossom not present");
        return;
    };
    assert!(
        stack.frames.len() >= 17,
        "need enough frames for two groups"
    );

    // --- 1. the split, on one frame, warm -------------------------------------------
    let subject = &stack.frames[0];
    let size = std::fs::metadata(subject).unwrap().len();

    let started = Instant::now();
    let _ = std::fs::read(subject).unwrap();
    let cold_read = started.elapsed();

    // Second read of the same file: whatever the page cache can serve.
    let started = Instant::now();
    let bytes = std::fs::read(subject).unwrap();
    let warm_read = started.elapsed();
    assert_eq!(bytes.len() as u64, size);

    let _ = decode_fully(subject); // warm the decoder's own path too
    let started = Instant::now();
    decode_fully(subject);
    let warm_decode = started.elapsed();

    println!("\n=== one frame, {:.0} MB ===", size as f64 / 1e6);
    println!("raw read, cold:   {}", rate(size, cold_read));
    println!("raw read, warm:   {}", rate(size, warm_read));
    println!("full decode, warm:{}", rate(size, warm_decode));
    println!(
        "conversion share: {:.0}% of the decode is not I/O",
        100.0 * (warm_decode.saturating_sub(warm_read)).as_secs_f64() / warm_decode.as_secs_f64()
    );

    // --- 2. the scaling, on two disjoint cold groups ---------------------------------
    // Frames far from index 0 and from each other, none read above.
    let sequential_group: Vec<_> = (1..9).map(|i| stack.frames[i].clone()).collect();
    let parallel_group: Vec<_> = (9..17).map(|i| stack.frames[i].clone()).collect();
    let group_bytes = size * 8;

    let started = Instant::now();
    for path in &sequential_group {
        decode_fully(path);
    }
    let sequential = started.elapsed();

    let started = Instant::now();
    std::thread::scope(|scope| {
        for path in &parallel_group {
            scope.spawn(move || decode_fully(path));
        }
    });
    let parallel = started.elapsed();

    println!("\n=== 8 frames each, disjoint and cold ===");
    println!("sequential: {}", rate(group_bytes, sequential));
    println!("8 threads:  {}", rate(group_bytes, parallel));
    println!(
        "speedup:    {:.1}x",
        sequential.as_secs_f64() / parallel.as_secs_f64()
    );
    println!(
        "\nA speedup near 1x would mean the decode is disk-bound and the parallel plan is\nworth much less than estimated.\n"
    );
}
