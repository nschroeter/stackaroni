//! Typed errors at the `core` library boundary.
//!
//! `cli` and `app` are free to use `anyhow` internally, but the pipeline stages
//! return these so callers can tell failure kinds apart — the debug/diagnostic view
//! needs "frame 47 failed to decode" and "ran out of scratch disk" to be
//! distinguishable, which an opaque error string collapses.

use std::path::PathBuf;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    /// Reading or writing a file failed.
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A TIFF was structurally unreadable.
    #[error("{path}: decoding failed: {source}")]
    Decode {
        path: PathBuf,
        #[source]
        source: tiff::TiffError,
    },

    /// Writing the output TIFF failed.
    #[error("{path}: encoding failed: {source}")]
    Encode {
        path: PathBuf,
        #[source]
        source: tiff::TiffError,
    },

    /// A readable TIFF that isn't what the pipeline accepts.
    #[error("{path}: expected 16-bit RGB TIFF, found {found} — develop to 16-bit TIFF first")]
    UnsupportedFormat { path: PathBuf, found: String },

    /// A stack directory with nothing to stack.
    #[error("{dir}: no TIFF frames found")]
    NoFrames { dir: PathBuf },

    /// A test-set root with no stack directories under it.
    #[error("{root}: contains no stack directories")]
    NoStacks { root: PathBuf },

    /// Frames in one stack disagree on geometry.
    #[error(
        "{path} is {width}x{height} {bits}-bit, but {reference} is \
         {ref_width}x{ref_height} {ref_bits}-bit — frames in a stack must match"
    )]
    Geometry {
        path: PathBuf,
        width: u32,
        height: u32,
        bits: u8,
        reference: PathBuf,
        ref_width: u32,
        ref_height: u32,
        ref_bits: u8,
    },

    /// Creating or mapping a scratch plane failed. Disk exhaustion lands here.
    #[error("scratch plane {path}: {source}")]
    Scratch {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The caller asked the run to stop.
    ///
    /// Deliberately an error rather than an `Ok` variant: it has to unwind out of a
    /// loop nested several stages deep, and every call site already handles `?`. It is
    /// the one variant that is not a fault — callers should treat it as "nothing to
    /// report and nothing to inspect", and in particular should *not* keep the scratch
    /// directory the way they do for a genuine failure.
    #[error("cancelled")]
    Cancelled,

    /// A band request fell outside the image.
    #[error("rows {start}..{end} out of bounds for height {height}")]
    Bounds { start: u64, end: u64, height: u32 },

    /// A caller-supplied buffer was the wrong size.
    #[error("buffer is {got} samples, expected {want}")]
    BufferSize { got: usize, want: usize },

    /// The result would land in the directory it was stacked from.
    ///
    /// Its own variant because the remedy is specific and the user must see it *before*
    /// the run, not as a generic write failure afterwards.
    #[error(
        "{output} is inside the stack directory {dir}, so the result would be read back \
         as an extra frame on the next run — write it somewhere else"
    )]
    OutputInsideStack { output: PathBuf, dir: PathBuf },
}

impl Error {
    /// Attach a path to an [`std::io::Error`].
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
