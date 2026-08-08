//! Focus-stacking pipeline: registration, focus measurement, weight estimation, fusion.
//!
//! See `docs/algorithms.md` for the algorithms each stage implements and why.

pub mod discovery;
pub mod error;
pub mod image;
pub mod pipeline;
pub mod tiff_io;
