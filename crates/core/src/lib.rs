//! Focus-stacking pipeline: registration, focus measurement, weight estimation, fusion.
//!
//! See `docs/algorithms.md` for the algorithms each stage implements and why.

pub mod budget;
pub mod debug;
pub mod defaults;
pub mod discovery;
pub mod error;
pub mod filter;
pub mod focus;
pub mod fusion;
pub mod grid;
pub mod image;
pub mod pipeline;
pub mod registration;
pub mod tiff_io;
pub mod weights;
