//! A minimal sRGB ICC profile, generated rather than shipped.
//!
//! Output TIFFs carry no colour space of their own otherwise. Most viewers guess sRGB and
//! land on the right answer by luck, but a colour-managed application has nothing to read
//! and either asks the user or assumes its own working space — which is how a correct file
//! ends up displayed wrong, and how the rating loop this project depends on would be
//! judging the viewer rather than the pipeline.
//!
//! **Built here rather than embedded as a binary asset.** The profiles that ship with an OS
//! are not ours to redistribute, and a downloaded one is a blob in the tree that nobody can
//! review in a diff. This is ~2 KB of code with every number visible: the sRGB primaries
//! and white point as published, and a tone curve sampled from the same
//! [`srgb_to_linear`](crate::image::srgb_to_linear) the decoder uses, so the profile
//! describes the transfer function the pipeline actually applies rather than an
//! approximation of it.
//!
//! ICC.1:2001-04 (v2.4) — a display-class matrix/TRC profile, the simplest form that a
//! colour-managed reader will accept for RGB.

use crate::image::srgb_to_linear;

/// Samples in each tone-reproduction curve.
///
/// 1024 keeps the profile small while resolving the curve's toe, where the sRGB function is
/// linear and a coarse table would visibly kink.
const TRC_ENTRIES: usize = 1024;

/// sRGB primaries, chromatically adapted to the D50 profile connection space, and the D50
/// white point itself — the values published for sRGB IEC 61966-2.1.
const PRIMARIES: [[f64; 3]; 3] = [
    [0.436_07, 0.222_49, 0.013_92],
    [0.385_15, 0.716_87, 0.097_08],
    [0.143_07, 0.060_61, 0.714_10],
];
const WHITE_POINT: [f64; 3] = [0.964_20, 1.000_00, 0.824_91];

/// Creation date written into the header: 2026-08-18, midnight UTC.
///
/// Fixed rather than "now", and that is load-bearing. Two runs of the same pipeline on the
/// same frames must produce the same file, byte for byte — it is what `output_is_stable`
/// checks and what every "output did not change" claim in `docs/eval-log.md` rests on. A
/// timestamp here would make every output differ from every other.
const CREATED: [u16; 6] = [2026, 8, 18, 0, 0, 0];

/// The complete profile.
pub fn srgb_profile() -> Vec<u8> {
    let description = text_description("sRGB IEC61966-2.1");
    let copyright = text("Public domain");
    let curve = tone_curve();
    let white = xyz(WHITE_POINT);

    // The three TRC tags point at one curve: identical data, and the ICC spec allows tags
    // to share it. Written three times this profile would be 6 KB rather than 2.
    let elements: [(&[u8; 4], &[u8]); 6] = [
        (b"desc", &description),
        (b"wtpt", &white),
        (b"rXYZ", &xyz(PRIMARIES[0])),
        (b"gXYZ", &xyz(PRIMARIES[1])),
        (b"bXYZ", &xyz(PRIMARIES[2])),
        (b"curv", &curve),
    ];

    // Tag table entries, in the order a reader will see them. The last three name the
    // shared curve.
    let names: [(&[u8; 4], usize); 9] = [
        (b"desc", 0),
        (b"wtpt", 1),
        (b"rXYZ", 2),
        (b"gXYZ", 3),
        (b"bXYZ", 4),
        (b"rTRC", 5),
        (b"gTRC", 5),
        (b"bTRC", 5),
        (b"cprt", 6),
    ];

    let table_len = 4 + names.len() * 12;
    let mut offsets = Vec::with_capacity(elements.len() + 1);
    let mut cursor = 128 + table_len;
    for (_, data) in elements {
        cursor = cursor.next_multiple_of(4);
        offsets.push((cursor, data.len()));
        cursor += data.len();
    }
    // The copyright, appended last so it can share the same layout pass.
    cursor = cursor.next_multiple_of(4);
    offsets.push((cursor, copyright.len()));
    cursor += copyright.len();
    let total = cursor.next_multiple_of(4);

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&header(total));

    out.extend_from_slice(&(names.len() as u32).to_be_bytes());
    for (signature, element) in names {
        let (offset, size) = offsets[element];
        out.extend_from_slice(signature);
        out.extend_from_slice(&(offset as u32).to_be_bytes());
        out.extend_from_slice(&(size as u32).to_be_bytes());
    }

    for (index, (_, data)) in elements.iter().enumerate() {
        pad_to(&mut out, offsets[index].0);
        out.extend_from_slice(data);
    }
    pad_to(&mut out, offsets[elements.len()].0);
    out.extend_from_slice(&copyright);
    pad_to(&mut out, total);
    out
}

/// The 128-byte profile header.
fn header(size: usize) -> [u8; 128] {
    let mut h = [0u8; 128];
    h[0..4].copy_from_slice(&(size as u32).to_be_bytes());
    // Version 2.4.0, then: display device, RGB data, XYZ connection space.
    h[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    h[12..16].copy_from_slice(b"mntr");
    h[16..20].copy_from_slice(b"RGB ");
    h[20..24].copy_from_slice(b"XYZ ");
    for (i, part) in CREATED.iter().enumerate() {
        h[24 + i * 2..26 + i * 2].copy_from_slice(&part.to_be_bytes());
    }
    h[36..40].copy_from_slice(b"acsp");
    // Rendering intent 0, perceptual — the default for a display profile.
    h[64..68].copy_from_slice(&0u32.to_be_bytes());
    // The PCS illuminant is fixed at D50 by the specification; it is not a free parameter.
    h[68..80].copy_from_slice(&s15_fixed_16_triple(WHITE_POINT));
    h
}

/// `XYZType`: three s15Fixed16 values behind an 8-byte type header.
fn xyz(value: [f64; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(b"XYZ ");
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&s15_fixed_16_triple(value));
    out
}

/// `curveType`, sampled from the pipeline's own transfer function.
///
/// A TRC maps an encoded device value to linear light, which is exactly what
/// `srgb_to_linear` does, so the profile cannot drift from the decoder.
fn tone_curve() -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + TRC_ENTRIES * 2);
    out.extend_from_slice(b"curv");
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&(TRC_ENTRIES as u32).to_be_bytes());
    for i in 0..TRC_ENTRIES {
        let encoded = i as f32 / (TRC_ENTRIES - 1) as f32;
        let linear = srgb_to_linear(encoded).clamp(0.0, 1.0);
        out.extend_from_slice(&((linear * 65535.0).round() as u16).to_be_bytes());
    }
    out
}

/// `textDescriptionType`, the v2 form: ASCII, then empty Unicode and ScriptCode blocks that
/// are required to be present even when unused.
fn text_description(value: &str) -> Vec<u8> {
    let ascii = value.as_bytes();
    let mut out = Vec::with_capacity(90 + ascii.len());
    out.extend_from_slice(b"desc");
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&(ascii.len() as u32 + 1).to_be_bytes());
    out.extend_from_slice(ascii);
    out.push(0);
    out.extend_from_slice(&[0; 8]); // Unicode language code and count.
    out.extend_from_slice(&[0; 3]); // ScriptCode code and count.
    out.extend_from_slice(&[0; 67]); // ScriptCode description, fixed width.
    out
}

/// `textType`: a null-terminated ASCII string.
fn text(value: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(9 + value.len());
    out.extend_from_slice(b"text");
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(value.as_bytes());
    out.push(0);
    out
}

fn s15_fixed_16_triple(value: [f64; 3]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for (i, v) in value.iter().enumerate() {
        let fixed = (v * 65536.0).round() as i32;
        out[i * 4..i * 4 + 4].copy_from_slice(&fixed.to_be_bytes());
    }
    out
}

fn pad_to(out: &mut Vec<u8>, offset: usize) {
    while out.len() < offset {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header's own size field is what a reader trusts to find the end of the profile.
    #[test]
    fn the_header_describes_the_profile_that_follows() {
        let profile = srgb_profile();
        assert_eq!(
            u32::from_be_bytes(profile[0..4].try_into().unwrap()) as usize,
            profile.len(),
            "size field must match the actual length"
        );
        assert_eq!(&profile[36..40], b"acsp", "profile signature");
        assert_eq!(&profile[16..20], b"RGB ");
        assert_eq!(profile.len() % 4, 0, "profiles are 4-byte aligned");
    }

    /// Every tag must name data that lies inside the profile and starts on a word boundary.
    /// A tag table pointing past the end is the failure mode a reader cannot recover from.
    #[test]
    fn every_tag_points_inside_the_profile() {
        let profile = srgb_profile();
        let count = u32::from_be_bytes(profile[128..132].try_into().unwrap()) as usize;
        assert_eq!(
            count, 9,
            "nine tags: desc, wtpt, three primaries, three TRCs, cprt"
        );

        let mut seen = Vec::new();
        for i in 0..count {
            let entry = 132 + i * 12;
            let signature = &profile[entry..entry + 4];
            let offset =
                u32::from_be_bytes(profile[entry + 4..entry + 8].try_into().unwrap()) as usize;
            let size =
                u32::from_be_bytes(profile[entry + 8..entry + 12].try_into().unwrap()) as usize;

            assert_eq!(offset % 4, 0, "tag {i} is not word-aligned");
            assert!(offset + size <= profile.len(), "tag {i} runs past the end");
            seen.push(String::from_utf8_lossy(signature).into_owned());
        }
        assert_eq!(
            seen,
            [
                "desc", "wtpt", "rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC", "cprt"
            ]
        );
    }

    /// The three TRC tags share one curve, and the curve is the transfer function the
    /// decoder applies — checked at the two ends and at the toe/shoulder join, where a
    /// wrong curve would differ most.
    #[test]
    fn the_tone_curve_is_the_pipelines_own() {
        let profile = srgb_profile();
        let tag = |name: &[u8]| {
            let count = u32::from_be_bytes(profile[128..132].try_into().unwrap()) as usize;
            (0..count)
                .map(|i| 132 + i * 12)
                .find(|&e| &profile[e..e + 4] == name)
                .map(|e| u32::from_be_bytes(profile[e + 4..e + 8].try_into().unwrap()) as usize)
                .unwrap()
        };
        let (r, g, b) = (tag(b"rTRC"), tag(b"gTRC"), tag(b"bTRC"));
        assert_eq!((r, g), (r, b), "the three curves share one element");

        let entries = u32::from_be_bytes(profile[r + 8..r + 12].try_into().unwrap()) as usize;
        assert_eq!(entries, TRC_ENTRIES);
        let at = |i: usize| {
            let o = r + 12 + i * 2;
            u16::from_be_bytes(profile[o..o + 2].try_into().unwrap()) as f32 / 65535.0
        };
        assert_eq!(at(0), 0.0, "black stays black");
        assert!((at(entries - 1) - 1.0).abs() < 1e-4, "white stays white");
        for i in [1, 8, 64, 512, 900] {
            let want = srgb_to_linear(i as f32 / (TRC_ENTRIES - 1) as f32);
            assert!(
                (at(i) - want).abs() < 1e-4,
                "entry {i}: {} != {want}",
                at(i)
            );
        }
    }
}
