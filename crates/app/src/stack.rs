//! Loading a folder of frames and turning each into a filmstrip thumbnail.
//!
//! # Why this is threaded
//!
//! A real stack is 24-60 MP x 30-100+ frames. Decoding every frame to build thumbnails
//! is tens of gigabytes of I/O, so doing it inline would freeze the window for minutes
//! with no indication of progress. Frames are decoded on a worker thread and sent back
//! one at a time, so the filmstrip fills in progressively and the UI stays responsive
//! throughout.
//!
//! Only the *header* read happens synchronously: [`load`] discovers the frames and
//! probes their geometry, which is cheap and is what the status bar needs immediately.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use eframe::egui;
use stackaroni_core::discovery::discover_stack;
use stackaroni_core::error::Result;
use stackaroni_core::image::{FrameInfo, linear_to_srgb};
use stackaroni_core::pipeline::Image;

/// Thumbnail width in pixels. Matches the filmstrip's inner width so the common case
/// needs no rescaling at draw time.
const THUMBNAIL_WIDTH: usize = 128;

/// Source rows averaged per output row, at most.
///
/// A box filter over the full `factor`-row band would be the correct downsample, but on
/// an 8664x5784 frame that is a 65-row band and means reading every row of every frame —
/// ~30 GB across a 100-frame stack, just to draw a filmstrip. Averaging the first few
/// rows of each band instead keeps the thumbnail free of the worst aliasing while
/// reading a small fraction of the file. Thumbnails are for recognizing a frame, not for
/// judging it; the preview pane is where fidelity will matter.
const MAX_ROWS_PER_BAND: u32 = 4;

/// A frame's thumbnail, or the reason there isn't one.
pub enum Thumbnail {
    Pending,
    Ready(egui::TextureHandle),
    Failed(String),
}

pub struct Frame {
    pub path: PathBuf,
    pub thumbnail: Thumbnail,
}

/// A loaded stack: frame list and geometry, with thumbnails arriving over time.
pub struct Stack {
    pub name: String,
    pub info: FrameInfo,
    pub frames: Vec<Frame>,
    /// Thumbnails decoded so far, ready or failed.
    pub decoded: usize,
    receiver: Receiver<Message>,
    /// Bumped on every load so a superseded worker's results are discarded rather than
    /// landing in the wrong stack.
    generation: Arc<AtomicU64>,
    id: u64,
}

enum Message {
    Ready {
        id: u64,
        index: usize,
        image: egui::ColorImage,
    },
    Failed {
        id: u64,
        index: usize,
        error: String,
    },
}

impl Stack {
    /// Discover and probe a folder, then start decoding thumbnails in the background.
    ///
    /// Returns once the geometry is known — the frames themselves are still loading.
    pub fn load(dir: &Path, generation: Arc<AtomicU64>) -> Result<Self> {
        let discovered = discover_stack(dir)?;
        let probe = discovered.probe()?;
        let id = generation.fetch_add(1, Ordering::SeqCst) + 1;

        let (sender, receiver) = channel();
        let paths: Vec<PathBuf> = discovered.frames.clone();
        let worker_generation = Arc::clone(&generation);
        std::thread::spawn(move || decode_all(&paths, id, &worker_generation, &sender));

        Ok(Self {
            name: discovered.name,
            info: probe.info,
            frames: discovered
                .frames
                .into_iter()
                .map(|path| Frame {
                    path,
                    thumbnail: Thumbnail::Pending,
                })
                .collect(),
            decoded: 0,
            receiver,
            generation,
            id,
        })
    }

    /// Take whatever the worker has finished since the last call and upload it.
    ///
    /// Called once per UI pass. Uploading textures needs the egui context, which the
    /// worker thread has no business touching, so the handoff is raw pixels.
    pub fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.receiver.try_recv() {
            match message {
                Message::Ready { id, index, image } => {
                    if id != self.id {
                        continue;
                    }
                    let handle = ctx.load_texture(
                        format!("thumb-{id}-{index}"),
                        image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.frames[index].thumbnail = Thumbnail::Ready(handle);
                    self.decoded += 1;
                }
                Message::Failed { id, index, error } => {
                    if id != self.id {
                        continue;
                    }
                    self.frames[index].thumbnail = Thumbnail::Failed(error);
                    self.decoded += 1;
                }
            }
        }
    }

    pub fn is_loading(&self) -> bool {
        self.decoded < self.frames.len()
    }
}

impl Drop for Stack {
    /// Retire this stack's id so its worker stops at the next frame rather than
    /// decoding a folder nobody is looking at any more.
    fn drop(&mut self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

fn decode_all(paths: &[PathBuf], id: u64, generation: &AtomicU64, sender: &Sender<Message>) {
    for (index, path) in paths.iter().enumerate() {
        // Checked per frame: a folder opened while this one is still decoding should
        // not keep a stale worker reading gigabytes in the background.
        if generation.load(Ordering::SeqCst) != id {
            return;
        }
        let message = match thumbnail(path) {
            Ok(image) => Message::Ready { id, index, image },
            Err(error) => Message::Failed {
                id,
                index,
                error: error.to_string(),
            },
        };
        // A closed receiver means the stack is gone; stop rather than decode into the void.
        if sender.send(message).is_err() {
            return;
        }
    }
}

/// Decode one frame down to a thumbnail.
fn thumbnail(path: &Path) -> Result<egui::ColorImage> {
    let image = Image::open(path)?;
    let info = image.info();

    let factor = (info.width as usize / THUMBNAIL_WIDTH).max(1) as u32;
    let (width, height) = (
        (info.width / factor).max(1) as usize,
        (info.height / factor).max(1) as usize,
    );

    let rows_per_band = factor.min(MAX_ROWS_PER_BAND);
    let mut band = vec![0f32; info.row_len() * rows_per_band as usize];
    let mut pixels = Vec::with_capacity(width * height);

    for ty in 0..height {
        let y0 = ty as u32 * factor;
        let count = rows_per_band.min(info.height - y0);
        image.read_rows(y0, count, &mut band)?;

        for tx in 0..width {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for row in 0..count as usize {
                let row = &band[row * info.row_len()..][..info.row_len()];
                for sx in 0..factor as usize {
                    let x = (tx * factor as usize + sx).min(info.width as usize - 1);
                    r += row[x * 3];
                    g += row[x * 3 + 1];
                    b += row[x * 3 + 2];
                }
            }
            let inv = 1.0 / (count as f32 * factor as f32);
            pixels.push(egui::Color32::from_rgb(
                encode(r * inv),
                encode(g * inv),
                encode(b * inv),
            ));
        }
    }

    Ok(egui::ColorImage::new([width, height], pixels))
}

/// Linear light to an 8-bit sRGB sample.
///
/// The same re-encode the TIFF writer does. Skipping it would make every thumbnail far
/// too dark, since the decoder hands back linear light and the screen expects sRGB.
fn encode(linear: f32) -> u8 {
    (linear_to_srgb(linear).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use stackaroni_core::tiff_io::write_rgb16_srgb;

    /// Write a frame of one flat colour, given in linear light.
    fn write_flat(path: &Path, width: u32, height: u32, rgb: [f32; 3]) {
        let info = FrameInfo {
            width,
            height,
            samples: 3,
            bits_per_sample: 16,
        };
        write_rgb16_srgb(path, info, |_, row| {
            for pixel in row.chunks_exact_mut(3) {
                pixel.copy_from_slice(&rgb);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn thumbnail_downsamples_and_keeps_the_aspect_ratio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        write_flat(&path, 512, 256, [0.5, 0.5, 0.5]);

        // 512 / 128 = 4, so both edges shrink by the same factor. A thumbnail that
        // scaled the axes independently would still look plausible in the filmstrip
        // while quietly distorting every frame.
        let thumb = thumbnail(&path).unwrap();
        assert_eq!(thumb.size, [128, 64]);
    }

    #[test]
    fn thumbnail_re_encodes_to_srgb_and_keeps_channels_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        // Deliberately asymmetric: a red/blue swap is invisible on any grey fixture.
        write_flat(&path, 256, 128, [0.2, 0.5, 0.8]);

        let thumb = thumbnail(&path).unwrap();
        let pixel = thumb.pixels[thumb.pixels.len() / 2];

        // The decoder yields linear light; writing it to screen without re-encoding
        // would render every thumbnail far too dark — 0.5 linear is 188, not 128.
        for (got, linear) in [(pixel.r(), 0.2), (pixel.g(), 0.5), (pixel.b(), 0.8)] {
            let want = encode(linear);
            assert!(
                got.abs_diff(want) <= 2,
                "got {got}, want ~{want} for linear {linear}"
            );
        }
    }

    #[test]
    fn thumbnail_handles_a_frame_smaller_than_one_thumbnail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        write_flat(&path, 64, 32, [0.5, 0.5, 0.5]);

        // factor clamps to 1 rather than 0 — dividing by it otherwise panics, and a
        // small frame is exactly what a hand-made test folder contains.
        let thumb = thumbnail(&path).unwrap();
        assert_eq!(thumb.size, [64, 32]);
    }
}
