//! Main stacking view.
//!
//! Follows the reference layout named in `CLAUDE.md`: filmstrip of frame thumbnails on
//! the left, large preview in the centre, parameter panel on the right. That is the
//! established shape for this category of tool (Lightroom's Develop module, darktable,
//! Zerene Stacker, Helicon Focus), so it is copied rather than designed.
//!
//! Fixed panels, not `egui_dock`. A dockable layout is one more unknown and buys nothing
//! until a fixed one is shown to be limiting in practice.
//!
//! Folder loading, the filmstrip, the preview, running and export are all real, and
//! every control in the parameter panel reaches the pipeline — nothing here is
//! cosmetic any more.
//!
//! # A `CLAUDE.md` spec clause intentionally replaced, not dropped
//!
//! The spec for this view asks to "preview registration/focus-map output **on a crop**
//! before running the full stack". That was built, tried, and removed on 2026-08-10.
//!
//! **Why it was removed.** It showed the measured transform for one pair — a scale
//! factor and an offset — alongside an alignment overlay and a focus heatmap. None of
//! those answer a question anyone actually has, because there is no baseline to judge
//! them against: `scale 1.00312` is neither good nor bad on its own, and a focus heatmap
//! of a single frame says nothing about which frame is sharpest here. It was information
//! rather than an answer.
//!
//! **What replaces it.** Pan and zoom on the preview pane, which serves the underlying
//! need directly: zoom into an antenna and step through frames to see which resolves it,
//! which is a comparison a person can actually make. The [`View`] state deliberately
//! survives frame changes for exactly this.
//!
//! **What is therefore still not covered.** Nothing previews the *guide radius*,
//! *epsilon* or *fusion rule* — those act on weights and fusion, which need every frame,
//! not one crop, so no crop-shaped feature could have covered them either. Judging those
//! still costs a full run. If that becomes the bottleneck, the shape to reach for is a
//! run over a spatial subset of every frame, not a richer single-frame preview.

mod reap;
mod run;
mod stack;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use eframe::egui;
use run::{Export, Outcome, Run, Settings};
use stack::{Preview, Stack, Thumbnail};
use stackaroni_core::discovery::ensure_output_outside_stack;
use stackaroni_core::image::FrameInfo;

fn main() -> eframe::Result {
    eframe::run_native(
        "Stackaroni",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
            ..Default::default()
        },
        Box::new(|_cc| {
            // Before the window is doing anything else. Stale scratch from runs that
            // died or were killed is exactly what eats the headroom the free-space
            // check protects, so the two belong together.
            let reaped = reap::stale(&reap::temp_root());
            Ok(Box::new(App {
                reaped: (reaped.entries > 0).then_some(reaped),
                ..App::default()
            }))
        }),
    )
}

/// Empty-state plates, so the filmstrip has its real shape before a folder is opened.
const PLACEHOLDER_FRAMES: usize = 8;

/// Height of a filmstrip entry. Thumbnails are fitted inside this, letterboxed.
const THUMBNAIL_HEIGHT: f32 = 88.0;

// The app offers every entry in `Method::ALL`. The fusion *rule* is no longer a choice
// here: `Method::Local` carries `defaults::FUSION`, which is `Select`.
//
// `Blend` remains reachable from the CLI (`--fusion blend`) and not from here. It is a
// real alternative with a real trade-off, but on both real test stacks it measured worse
// (blossom 1/5 against selection's 5/5), and the eval log has no case where anyone should
// choose it for a photograph. It stays available for reproducing older eval-log rows,
// which is a job for the headless runner, not a dropdown a photographer reads.
//
// The pipeline still supports it end to end — `the_app_and_the_cli_agree_byte_for_byte`
// covers both rules — so offering it here would mean giving `Local` its own rule combo,
// not editing a list.

/// Preview key for the fused result, kept clear of every real frame index.
const RESULT_KEY: usize = usize::MAX;

/// Furthest zoom in, as screen pixels per source pixel. Past 1:1 there is no more
/// detail in the file, so this is only headroom for looking closely at what is there.
const MAX_ZOOM: f32 = 4.0;

/// How the source image is mapped onto the pane: `scale` screen pixels per source pixel,
/// with the source origin at `origin`.
///
/// Deliberately survives changing frames. Zooming into an antenna and then stepping
/// through the stack to see which frame renders it sharpest is the reason this exists,
/// and resetting the view on every selection would destroy exactly that.
#[derive(Clone, Copy)]
struct View {
    scale: f32,
    origin: egui::Pos2,
}

impl View {
    fn fit(rect: egui::Rect, info: FrameInfo) -> Self {
        let scale = (rect.width() / info.width as f32).min(rect.height() / info.height as f32);
        Self {
            scale,
            origin: rect.center() - egui::vec2(info.width as f32, info.height as f32) * scale / 2.0,
        }
    }

    fn source_to_screen(&self, region: stack::Region) -> egui::Rect {
        egui::Rect::from_min_size(
            self.origin + egui::vec2(region.x as f32, region.y as f32) * self.scale,
            egui::vec2(region.w as f32, region.h as f32) * self.scale,
        )
    }

    /// Scroll to zoom about the cursor, left-drag to pan.
    fn interact(&mut self, ui: &mut egui::Ui, rect: egui::Rect, info: FrameInfo) {
        let response = ui.interact(
            rect,
            ui.id().with("preview-view"),
            egui::Sense::click_and_drag(),
        );

        // Panning only exists where there is something off-screen to reach. Fully
        // visible, dragging would just slide the image around inside the pane for no
        // reason, so the axis clamps below undo it and the cursor never offers it.
        let size = egui::vec2(info.width as f32, info.height as f32) * self.scale;
        let pannable = size.x > rect.width() + 0.5 || size.y > rect.height() + 0.5;

        if response.dragged() && pannable {
            self.origin += response.drag_delta();
        }
        if pannable && (response.dragged() || response.hovered()) {
            ui.ctx().set_cursor_icon(if response.dragged() {
                egui::CursorIcon::Grabbing
            } else {
                egui::CursorIcon::Grab
            });
        }

        // Anchored on the pointer, not the pane centre: zooming should keep whatever is
        // under the cursor under the cursor, which is what makes it feel like moving a
        // loupe rather than resizing a picture.
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if response.hovered()
            && scroll != 0.0
            && let Some(pointer) = response.hover_pos()
        {
            let fit = Self::fit(rect, info).scale;
            let factor = (scroll * 0.004).exp();
            let scale = (self.scale * factor).clamp(fit.min(MAX_ZOOM), MAX_ZOOM);
            let applied = scale / self.scale;
            self.origin = pointer + (self.origin - pointer) * applied;
            self.scale = scale;
        }

        // Double-click returns to fit, the standard escape hatch from being lost while
        // zoomed in.
        if response.double_clicked() {
            *self = Self::fit(rect, info);
        }

        // Per axis, because zoom crosses the two thresholds at different moments: a 3:2
        // frame in a wider pane overflows vertically well before it overflows
        // horizontally, and that axis should already pan while the other stays put.
        //
        // Overflowing, the image is clamped to cover the pane — no panning past its own
        // edge into empty space. Fitting, it is pinned centred, which is also what
        // silently cancels any drag on that axis.
        let size = egui::vec2(info.width as f32, info.height as f32) * self.scale;
        self.origin.x = if size.x > rect.width() {
            self.origin.x.clamp(rect.max.x - size.x, rect.min.x)
        } else {
            rect.center().x - size.x / 2.0
        };
        self.origin.y = if size.y > rect.height() {
            self.origin.y.clamp(rect.max.y - size.y, rect.min.y)
        } else {
            rect.center().y - size.y / 2.0
        };
    }

    /// The source rectangle now visible, plus the width to decode it at.
    ///
    /// Padded by half a pane and snapped to a grid so small pans land on the same
    /// request and reuse the texture, instead of starting a decode per frame of drag.
    fn wanted_region(&self, rect: egui::Rect, info: FrameInfo) -> (stack::Region, usize) {
        let to_source = |p: egui::Pos2| (p - self.origin) / self.scale;
        let min = to_source(rect.min);
        let max = to_source(rect.max);
        let pad = (max - min) * 0.25;

        let grid = ((64.0 / self.scale).max(1.0)) as u32;
        let snap_down = |v: f32| ((v.max(0.0) as u32) / grid) * grid;
        let snap_up =
            |v: f32, limit: u32| (((v.max(0.0) as u32).div_ceil(grid) + 1) * grid).min(limit);

        let x = snap_down(min.x - pad.x);
        let y = snap_down(min.y - pad.y);
        let region = stack::Region {
            x,
            y,
            w: snap_up(max.x + pad.x, info.width).saturating_sub(x).max(1),
            h: snap_up(max.y + pad.y, info.height).saturating_sub(y).max(1),
        };
        // One output pixel per screen pixel across the region, capped so an extreme zoom
        // cannot ask for a texture larger than the source it came from.
        let target = ((region.w as f32 * self.scale) as usize).clamp(256, 4096);
        (region, target)
    }
}

/// A finished run, retained after the worker is gone.
///
/// The modal stays up rather than vanishing at completion: the per-stage timings only
/// exist while it is on screen, and a dialog that disappears the instant the work ends
/// takes the answer with it just as the user looks up.
struct RunSummary {
    headline: &'static str,
    detail: Option<String>,
    elapsed: std::time::Duration,
    stages: [Option<std::time::Duration>; 4],
}

/// Names shown in the run modal, in the order the pipeline executes them.
const RUN_STAGES: [&str; 4] = ["Register", "Focus", "Weights", "Fuse"];

/// A small state marker: filled and ticked when done, ringed when running, hollow when
/// still to come. Drawn rather than typed, for the same reason the filmstrip badge is —
/// no glyph in the default font stack is guaranteed on every platform.
fn phase_marker(ui: &mut egui::Ui, state: &str, colour: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(14.0), egui::Sense::hover());
    let centre = rect.center();
    let painter = ui.painter();
    match state {
        "done" => {
            painter.circle_filled(centre, 7.0, colour);
            let stroke = egui::Stroke::new(2.0, ui.visuals().strong_text_color());
            let b = rect.shrink(4.0);
            let elbow = egui::pos2(b.left() + b.width() * 0.36, b.bottom());
            painter.line_segment([egui::pos2(b.left(), b.center().y), elbow], stroke);
            painter.line_segment([elbow, egui::pos2(b.right(), b.top())], stroke);
        }
        "now" => {
            painter.circle_stroke(centre, 6.0, egui::Stroke::new(2.0, colour));
            painter.circle_filled(centre, 2.5, colour);
        }
        _ => {
            painter.circle_stroke(centre, 6.0, egui::Stroke::new(1.0, colour));
        }
    }
}

/// `m:ss`, or `h:mm:ss` once a run is long enough to need it.
fn clock(d: std::time::Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// Side of the include/exclude checkbox drawn in a plate's corner.
const BADGE: f32 = 16.0;

/// Opacity of an excluded thumbnail.
///
/// The first attempt used 23%, which did not read as "excluded" — it read as a grey
/// wash, indistinguishable from a rendering fault. Dimming is reinforcement here, not
/// the signal: the unchecked badge carries the meaning, so this only has to be visibly
/// different while leaving the frame recognizable enough to change your mind about.
const EXCLUDED_OPACITY: u8 = 150;

// Bound to core's own types now that a run consumes them; they were local placeholders
// only while nothing read them.
use stackaroni_core::defaults;
use stackaroni_core::fusion::FusionKind;
use stackaroni_core::pipeline::Method;
use stackaroni_core::weights::GuideSpace;

/// The handful of parameters the CLI exposes, mirrored here.
#[derive(Clone, PartialEq)]
struct Params {
    registration_level: u32,
    focus_radius: u32,
    guide_radius: u32,
    guide_epsilon: f32,
    guide_space: GuideSpace,
    /// The whole pipeline shape, with each method's own parameters nested inside it.
    /// One field rather than a method plus a loose copy of every method's knobs, so
    /// the panel cannot offer a setting the selected method does not read.
    method: Method,
    pyramid_floor: u32,
}

impl Default for Params {
    /// Taken from `core::defaults`, so the panel opens on exactly the configuration the
    /// CLI runs and `docs/eval-log.md` scored.
    fn default() -> Self {
        Self {
            registration_level: defaults::REGISTRATION_LEVEL,
            focus_radius: defaults::FOCUS_RADIUS,
            guide_radius: defaults::GUIDE_RADIUS,
            guide_epsilon: defaults::GUIDE_EPSILON,
            guide_space: defaults::GUIDE_SPACE,
            method: defaults::METHOD,
            pyramid_floor: defaults::PYRAMID_FLOOR,
        }
    }
}

#[derive(Default)]
struct App {
    selected: usize,
    params: Params,
    stack: Option<Stack>,
    preview: Preview,
    run: Option<Run>,
    /// Names each run's scratch and output uniquely.
    run_sequence: u64,
    /// The fused result of the last successful run, shown in place of a frame.
    result: Option<std::path::PathBuf>,
    export: Option<Export>,
    /// How the preview pane is currently zoomed and panned. `None` until first shown,
    /// then kept across frame changes on purpose.
    view: Option<View>,
    /// A finished run, kept on screen until dismissed so the timings can be read.
    run_summary: Option<RunSummary>,
    /// What a startup sweep reclaimed, if anything. Reported rather than silent, because
    /// tens of GB disappearing without explanation is worse than a line of text.
    reaped: Option<reap::Reaped>,
    /// Where the last export landed, so the status bar can say so.
    exported: Option<std::path::PathBuf>,
    /// Shared with worker threads so a superseded load can be abandoned mid-decode.
    generation: Arc<AtomicU64>,
    error: Option<String>,
}

impl App {
    fn open_folder(&mut self) {
        let Some(dir) = rfd::FileDialog::new()
            .set_title("Choose a folder of 16-bit TIFF frames")
            .pick_folder()
        else {
            return;
        };

        // Dropping the previous stack retires its worker, so opening a second folder
        // mid-decode does not leave the first one reading gigabytes.
        self.stack = None;
        self.preview = Preview::default();
        self.selected = 0;

        match Stack::load(&dir, Arc::clone(&self.generation)) {
            Ok(stack) => {
                self.error = None;
                self.stack = Some(stack);
            }
            // Reported rather than swallowed: "no frames here" and "this folder mixes
            // geometries" are the two things most likely to go wrong on a real folder,
            // and core distinguishes them.
            Err(e) => self.error = Some(e.to_string()),
        }
    }
}

impl App {
    fn running(&self) -> bool {
        self.run.is_some()
    }

    fn start_run(&mut self, ctx: &egui::Context) {
        let Some(stack) = &self.stack else { return };
        let stack_info = stack.info;
        let frames = run::included_paths(stack);
        if frames.len() < 2 {
            self.error = Some("a run needs at least two included frames".into());
            return;
        }
        self.run_sequence += 1;
        let settings = Settings {
            registration_level: self.params.registration_level,
            focus_radius: self.params.focus_radius,
            guide_radius: self.params.guide_radius,
            guide_epsilon: self.params.guide_epsilon,
            guide_space: self.params.guide_space,
            method: self.params.method,
            pyramid_floor: self.params.pyramid_floor,
        };
        match Run::start(frames, stack_info, settings, ctx.clone(), self.run_sequence) {
            Ok(run) => {
                self.error = None;
                self.result = None;
                self.exported = None;
                self.run_summary = None;
                self.run = Some(run);
            }
            // Loud and immediate: a run that cannot fit must say so on the button
            // press, not by quietly failing to start.
            Err(e) => self.error = Some(e),
        }
    }

    /// Errors as a modal that has to be dismissed, not a line in the status bar.
    ///
    /// The status line was there and was not enough: the export refusal drew into it and
    /// read as passive text next to the stack name, which is a poor way to say "the thing
    /// you just asked for did not happen". Every error here reports a *user action that
    /// did not take effect* — a folder that would not load, a run that failed, an export
    /// that was refused — so acknowledging it is the point.
    ///
    /// One presentation for all of them rather than a special case for the refusal:
    /// a second error channel would be a second thing to keep in sync, and none of these
    /// is more dismissible than the others.
    fn error_modal(&mut self, ctx: &egui::Context) {
        let Some(message) = self.error.clone() else {
            return;
        };
        let response = egui::Modal::new(egui::Id::new("error-modal")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.heading("That did not work");
            ui.add_space(8.0);
            ui.label(message);
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                // Right-aligned, where the confirming button belongs on every platform
                // this ships to.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.button("OK").clicked()
                })
                .inner
            })
            .inner
        });

        // Escape and a click outside dismiss it too, which `should_close` folds in.
        if response.inner || response.should_close() {
            self.error = None;
        }
    }

    /// Why this export must not happen, if it must not.
    ///
    /// Refused rather than warned about: saving a result beside the frames it came from
    /// corrupts the stack for every later run, invisibly. The result has the frames'
    /// geometry, so it is simply stacked as an extra frame, and the only symptom is a
    /// frame count. Not hypothetical — the suggested filename in [`Self::export_result`]
    /// is exactly what landed in `test-data/blossom` and `test-data/ruler`, and went
    /// unnoticed through several runs and a full round of ratings.
    ///
    /// The directory comes from a frame rather than a stored field, because the frames
    /// are what is being protected: their own location is the authoritative answer and
    /// cannot drift from it.
    ///
    /// Split out from the dialog so it can be tested. The dialog itself cannot be driven
    /// here, so without this the app's half of the guard would rest on inspection alone.
    fn refusal_for(stack: Option<&stack::Stack>, destination: &Path) -> Option<String> {
        let dir = stack?.frames.first()?.path.parent()?;
        ensure_output_outside_stack(destination, dir)
            .err()
            .map(|e| e.to_string())
    }

    fn export_result(&mut self, ctx: &egui::Context) {
        let Some(source) = self.result.clone() else {
            return;
        };
        // Named after the stack, since "stackaroni-run-48612-3.tif" is not a filename
        // anyone wants in their photo library.
        let suggested = match &self.stack {
            Some(stack) => format!("{}_stacked.tif", stack.name),
            None => "stacked.tif".to_string(),
        };
        let Some(destination) = rfd::FileDialog::new()
            .set_title("Save the fused image")
            .set_file_name(suggested)
            .add_filter("16-bit TIFF", &["tif", "tiff"])
            .save_file()
        else {
            return;
        };

        if let Some(refusal) = Self::refusal_for(self.stack.as_ref(), &destination) {
            self.error = Some(refusal);
            return;
        }

        self.error = None;
        self.exported = None;
        self.export = Some(Export::start(source, destination, ctx.clone()));
    }

    fn poll_export(&mut self, ctx: &egui::Context) {
        let Some(export) = &mut self.export else {
            return;
        };
        match export.poll() {
            None => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            Some(result) => {
                self.export = None;
                match result {
                    Ok(path) => {
                        // The result now lives where the user put it; the temp copy is
                        // gone, so point at the new location or a later re-decode would
                        // read a file that no longer exists.
                        self.result = Some(path.clone());
                        self.exported = Some(path);
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
    }

    /// Collect a finished run. Only clears `self.run` once the worker has exited, which
    /// is what re-enables every control — see `run.rs`.
    fn poll_run(&mut self, ctx: &egui::Context) {
        let Some(run) = &mut self.run else { return };
        match run.poll() {
            None => {
                // Heartbeat, so the spinner animates and liveness does not depend on a
                // progress call landing between passes. Not a full-rate repaint loop:
                // this runs for twenty minutes.
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
            }
            Some(outcome) => {
                let shared = std::sync::Arc::clone(&run.shared);
                shared.finish();
                let (headline, detail) = match &outcome {
                    Outcome::Done(_) => ("Finished", None),
                    Outcome::Cancelled => (
                        "Cancelled",
                        Some("stopped before the output was written".to_string()),
                    ),
                    Outcome::Failed(error) => ("Failed", Some(error.clone())),
                };
                self.run_summary = Some(RunSummary {
                    headline,
                    detail,
                    elapsed: shared.elapsed(),
                    stages: shared.stage_times(),
                });
                self.run = None;

                match outcome {
                    Outcome::Done(path) => {
                        self.preview = Preview::default();
                        self.result = Some(path);
                    }
                    // Nothing to preserve: the write is never interrupted, so a cancelled
                    // run has produced no output and leaves the view as it was.
                    Outcome::Cancelled => {}
                    Outcome::Failed(error) => self.error = Some(error),
                }
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_run(ui.ctx());
        self.poll_export(ui.ctx());
        self.error_modal(ui.ctx());
        // Collect whatever the decoder finished since the last pass, and keep painting
        // while it works so thumbnails appear as they land rather than all at the end.
        if let Some(stack) = &mut self.stack {
            stack.poll(ui.ctx());
            if stack.is_loading() {
                ui.ctx().request_repaint();
            }
        }
        self.preview.poll(ui.ctx());
        if self.preview.is_loading() {
            ui.ctx().request_repaint();
        }

        // X toggles the selected frame, the keyboard half of the badge. Guarded on
        // egui not wanting the key itself, so typing into a future text field cannot
        // silently drop a frame from the run.
        if !ui.ctx().egui_wants_keyboard_input()
            && ui.input(|i| i.key_pressed(egui::Key::X))
            && let Some(stack) = &mut self.stack
            && let Some(frame) = stack.frames.get_mut(self.selected)
        {
            frame.included = !frame.included;
        }

        // Order matters: top and bottom claim full width, then the sides, then whatever
        // is left becomes the preview.
        //
        // Two egui 0.36 details worth stating, because both are recent changes and the
        // older spelling is what most examples still show. `egui::Panel` replaces
        // `SidePanel`/`TopBottomPanel`, which were unified into one type keyed by side and
        // removed rather than deprecated. And `show` takes a `Ui` — eframe hands us one
        // instead of a `Context` — so the nested-panel call is `show`, not the now
        // deprecated `show_inside`.
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

        egui::Panel::left("filmstrip")
            .resizable(false)
            .exact_size(150.0)
            .show(ui, |ui| self.filmstrip(ui));

        egui::Panel::right("parameters")
            .resizable(false)
            .exact_size(260.0)
            .show(ui, |ui| self.parameters(ui));

        egui::CentralPanel::default().show(ui, |ui| self.preview(ui));

        // Last, so it sits over the panels rather than under them.
        let ctx = ui.ctx().clone();
        self.run_modal(&ctx);
    }
}

impl App {
    /// The four phases of a run, all visible, with only the current one moving.
    ///
    /// A single bar that fills and restarts four times reads as a bug — the run looks
    /// finished, then apparently starts over. Showing every phase at once makes the shape of
    /// the work legible instead: what is done, what is running, what is still to come.
    /// Standard multi-step progress, the same pattern an installer uses.
    ///
    /// Modal because it matches what is already true: during a run the only available action
    /// is Cancel. The panels behind it are disabled anyway, and the backdrop makes that
    /// obvious rather than leaving the user to discover it by clicking.
    fn run_modal(&mut self, ctx: &egui::Context) {
        // One dialog, two states: a run in flight, and the same run once it has ended.
        // Sharing the layout is what makes the transition read as "this finished" rather
        // than "something else appeared".
        enum Shown {
            Running(std::sync::Arc<run::Shared>),
            Finished,
        }
        let shown = match (&self.run, &self.run_summary) {
            (Some(run), _) => Shown::Running(std::sync::Arc::clone(&run.shared)),
            (None, Some(_)) => Shown::Finished,
            (None, None) => return,
        };

        let (current, times, elapsed, headline, detail, progress) = match &shown {
            Shown::Running(shared) => {
                let (_, done, total) = shared.snapshot();
                (
                    shared.stage_index(),
                    shared.stage_times(),
                    shared.elapsed(),
                    format!(
                        "Stacking {}",
                        self.stack.as_ref().map(|s| s.name.as_str()).unwrap_or("")
                    ),
                    None,
                    Some((done, total)),
                )
            }
            Shown::Finished => {
                let summary = self.run_summary.as_ref().expect("checked above");
                (
                    // Past the last stage, so every phase draws as complete.
                    RUN_STAGES.len(),
                    summary.stages,
                    summary.elapsed,
                    summary.headline.to_string(),
                    summary.detail.clone(),
                    None,
                )
            }
        };

        let mut close = false;
        egui::Modal::new(egui::Id::new("run-progress")).show(ctx, |ui| {
            ui.set_width(400.0);
            ui.heading(headline.trim());
            if let Some(detail) = &detail {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(detail).weak());
            }
            ui.add_space(10.0);

            for (index, stage) in RUN_STAGES.iter().enumerate() {
                ui.horizontal(|ui| {
                    let (state, colour) = match index.cmp(&current) {
                        std::cmp::Ordering::Less => ("done", ui.visuals().selection.bg_fill),
                        std::cmp::Ordering::Equal => ("now", ui.visuals().strong_text_color()),
                        std::cmp::Ordering::Greater => ("todo", ui.visuals().weak_text_color()),
                    };
                    phase_marker(ui, state, colour);
                    ui.add_space(6.0);

                    let label = egui::RichText::new(*stage);
                    ui.label(match index.cmp(&current) {
                        std::cmp::Ordering::Equal => label.strong(),
                        std::cmp::Ordering::Greater => label.weak(),
                        std::cmp::Ordering::Less => label,
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some((done, total)) = progress.filter(|_| index == current) {
                            let fraction = if total == 0 {
                                0.0
                            } else {
                                done as f32 / total as f32
                            };
                            ui.add(
                                egui::ProgressBar::new(fraction)
                                    .desired_width(170.0)
                                    .text(format!("{done}/{total}")),
                            );
                        } else if let Some(took) = times[index] {
                            ui.label(
                                egui::RichText::new(format!("{:.0}s", took.as_secs_f32())).weak(),
                            );
                        }
                    });
                });
                ui.add_space(6.0);
            }

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("elapsed {}", clock(elapsed))).strong());
                if let Shown::Running(shared) = &shown
                    && shared.cancel_requested()
                {
                    ui.label(egui::RichText::new("· stopping after this frame").weak());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    match &shown {
                        Shown::Running(shared) => {
                            let cancelling = shared.cancel_requested();
                            let label = if cancelling {
                                "Cancelling…"
                            } else {
                                "Cancel"
                            };
                            if ui
                                .add_enabled(!cancelling, egui::Button::new(label))
                                .clicked()
                            {
                                shared.cancel();
                            }
                        }
                        // Dismissed by hand, never automatically: the timings are only
                        // here, and a dialog that closes itself takes them away exactly
                        // when someone looks up to read them.
                        Shown::Finished => {
                            if ui.button("Close").clicked() {
                                close = true;
                            }
                        }
                    }
                });
            });
        });

        if close {
            self.run_summary = None;
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let running = self.running();

            // Everything except Cancel is disabled while a run is live: the parameters
            // it was started with are fixed, and a second run would collide with the
            // first over scratch and output.
            if ui
                .add_enabled(!running, egui::Button::new("Open folder…"))
                .clicked()
            {
                self.open_folder();
            }
            ui.separator();

            match &self.run {
                None => {
                    let ready = self.stack.is_some();
                    if ui
                        .add_enabled(ready, egui::Button::new("Run stack"))
                        .clicked()
                    {
                        let ctx = ui.ctx().clone();
                        self.start_run(&ctx);
                    }
                }
                Some(run) => {
                    // Flips the instant it is pressed, before the pipeline notices.
                    // Perceived responsiveness is the acknowledgement, not the stop
                    // latency — the worst case is one frame of fusion behind it.
                    let requested = run.shared.cancel_requested();
                    let label = if requested { "Cancelling…" } else { "Cancel" };
                    if ui
                        .add_enabled(!requested, egui::Button::new(label))
                        .clicked()
                    {
                        run.shared.cancel();
                    }
                }
            }

            let exporting = self.export.is_some();
            let label = if exporting {
                "Exporting…"
            } else {
                "Export…"
            };
            if ui
                .add_enabled(
                    !running && !exporting && self.result.is_some(),
                    egui::Button::new(label),
                )
                .clicked()
            {
                let ctx = ui.ctx().clone();
                self.export_result(&ctx);
            }
            ui.separator();

            match (&self.error, &self.stack) {
                (Some(error), _) => {
                    ui.label(egui::RichText::new(error).color(ui.visuals().error_fg_color));
                }
                (None, Some(stack)) => {
                    let included = stack::included_count(&stack.frames);
                    let total = stack.frames.len();
                    ui.label(egui::RichText::new(&stack.name).strong());

                    // Highlighted once anything is excluded: the count is the difference
                    // between what is on screen and what a run would actually use, and
                    // that is worth noticing before pressing Run.
                    let count = egui::RichText::new(format!("{included}/{total} included"));
                    ui.label(if included == total {
                        count.weak()
                    } else {
                        count.strong()
                    });

                    ui.label(
                        egui::RichText::new(format!(
                            "· {}x{} · {}-bit",
                            stack.info.width, stack.info.height, stack.info.bits_per_sample,
                        ))
                        .weak(),
                    );
                }
                (None, None) => {
                    ui.label(egui::RichText::new("no stack loaded").weak());
                }
            }
        });
        ui.add_space(4.0);
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(2.0);
        if let Some(run) = &self.run {
            // No progress, no clock: the modal is on screen and owns both. Repeating
            // the elapsed time behind it just gives the eye two places to read the same
            // number.
            let _ = run;
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(egui::RichText::new("stacking…").strong());
            });
            ui.add_space(2.0);
            return;
        }
        if self.export.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(egui::RichText::new("exporting…").strong());
                ui.label(egui::RichText::new("copying the result to its destination").weak());
            });
            ui.add_space(2.0);
            return;
        }
        if let Some(path) = &self.exported {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("saved").strong());
                ui.label(egui::RichText::new(path.display().to_string()).weak());
            });
            ui.add_space(2.0);
            return;
        }
        ui.horizontal(|ui| {
            match &self.stack {
                Some(stack) if stack.is_loading() => {
                    ui.spinner();
                    ui.label(
                        egui::RichText::new(format!(
                            "decoding thumbnails {}/{}",
                            stack.decoded,
                            stack.frames.len()
                        ))
                        .weak(),
                    );
                }
                Some(stack) => {
                    ui.label(
                        egui::RichText::new(format!("{} thumbnails ready", stack.decoded)).weak(),
                    );
                }
                None => match &self.reaped {
                    Some(reaped) => {
                        ui.label(egui::RichText::new("Ready").weak());
                        ui.label(
                            egui::RichText::new(format!(
                                "· reclaimed {:.1} GB from {} abandoned run{}",
                                reaped.bytes as f64 / 1e9,
                                reaped.entries,
                                if reaped.entries == 1 { "" } else { "s" }
                            ))
                            .weak(),
                        );
                    }
                    None => {
                        ui.label(egui::RichText::new("Ready").weak());
                    }
                },
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new("no pipeline wired").weak());
            });
        });
        ui.add_space(2.0);
    }

    fn filmstrip(&mut self, ui: &mut egui::Ui) {
        // Selection and exclusion both freeze during a run: the frame list was captured
        // when it started, so changing it would misrepresent what is being stacked.
        let running = self.running();
        ui.add_enabled_ui(!running, |ui| self.filmstrip_inner(ui));
        let _ = running;
    }

    fn filmstrip_inner(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.label(egui::RichText::new("Frames").strong());
        ui.add_space(4.0);

        let Some(stack) = &self.stack else {
            // Empty state: the same plates as before a folder is chosen, so the panel
            // does not collapse to nothing and its width stays legible.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for _ in 0..PLACEHOLDER_FRAMES {
                        plate(ui, false, |_, _| {});
                        ui.add_space(4.0);
                    }
                });
            return;
        };

        // `show_rows` rather than a plain loop: a 100+ frame stack means most entries are
        // scrolled out of view, and only the visible ones are worth laying out.
        let row_height = THUMBNAIL_HEIGHT + 4.0;
        // Selecting and excluding are separate actions on separate affordances: clicking
        // the plate selects, the corner badge (or X on the selected frame) includes or
        // excludes. They were one click before, which meant you could not look at a frame
        // without dropping it from the run.
        let mut clicked = None;
        let mut toggled = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, row_height, stack.frames.len(), |ui, range| {
                for index in range {
                    let frame = &stack.frames[index];
                    let included = frame.included;
                    let response = plate(ui, index == self.selected, |ui, rect| {
                        match &frame.thumbnail {
                            Thumbnail::Ready(texture) => {
                                // Letterboxed: frames are 3:2 and the plate is not, so
                                // filling it would crop or distort.
                                let size = fit(texture.size_vec2(), rect.size());
                                let image = egui::Image::new(texture).corner_radius(2.0);
                                // Dimmed, not hidden: an excluded frame stays
                                // recognizable so the decision can be reversed by
                                // looking rather than by remembering.
                                let image = if included {
                                    image
                                } else {
                                    image.tint(egui::Color32::from_white_alpha(EXCLUDED_OPACITY))
                                };
                                image.paint_at(
                                    ui,
                                    egui::Rect::from_center_size(rect.center(), size),
                                );
                            }
                            Thumbnail::Pending => {
                                label_in(ui, rect, "…", ui.visuals().weak_text_color());
                            }
                            Thumbnail::Failed(_) => {
                                label_in(ui, rect, "failed", ui.visuals().error_fg_color);
                            }
                        }
                    });

                    // Registered after the plate and overlapping it, so egui gives the
                    // click to the badge rather than to the plate underneath.
                    let badge_rect = egui::Rect::from_min_size(
                        egui::pos2(
                            response.rect.right() - BADGE - 5.0,
                            response.rect.top() + 5.0,
                        ),
                        egui::Vec2::splat(BADGE),
                    );
                    let badge = ui.interact(
                        badge_rect,
                        ui.id().with(("include", index)),
                        egui::Sense::click(),
                    );
                    draw_badge(ui, badge_rect, included, &badge);
                    if badge.clicked() {
                        toggled = Some(index);
                    }

                    let name = frame
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    badge.on_hover_text(if included {
                        "included — click to exclude from the run"
                    } else {
                        "excluded — click to include in the run"
                    });

                    // A failed frame's reason belongs on the frame itself; the status bar
                    // only counts them.
                    let hint = match &frame.thumbnail {
                        Thumbnail::Failed(error) => format!("{name}\n{error}"),
                        _ => format!("{name}\nX to exclude"),
                    };
                    if response.on_hover_text(hint).clicked() {
                        clicked = Some(index);
                    }
                    ui.add_space(4.0);
                }
            });

        // Applied after the loop because the closure borrows the stack immutably.
        //
        // The plates above were already drawn from the pre-click state, so this pass
        // shows stale visuals and the change only appears on the next one. egui repaints
        // reactively, and nothing else here asks for a repaint once thumbnails have
        // finished loading — so without this the change appears not to happen until some
        // unrelated input (a mouse move, or a second click) triggers a pass.
        if let Some(index) = clicked {
            self.selected = index;
            // Picking a frame leaves the fused-result view; the run is still available
            // through Export, it is just no longer what the pane is showing.
            self.result = None;
            ui.ctx().request_repaint();
        }
        if let Some(index) = toggled {
            if let Some(stack) = &mut self.stack {
                stack.frames[index].included = !stack.frames[index].included;
            }
            ui.ctx().request_repaint();
        }
    }

    fn preview(&mut self, ui: &mut egui::Ui) {
        let heading = match (&self.result, &self.stack) {
            (Some(_), _) => "fused result".to_string(),
            (None, Some(stack)) => stack
                .frames
                .get(self.selected)
                .and_then(|f| f.path.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            (None, None) => String::new(),
        };
        ui.vertical_centered(|ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(heading).strong());
        });
        ui.add_space(6.0);

        // Fills whatever the panels leave, which is the point of the centre pane: it is
        // where a 50 MP frame has to be legible.
        let rect = ui.available_rect_before_wrap();
        let visuals = ui.visuals().clone();
        ui.painter().rect(
            rect,
            visuals.window_corner_radius,
            visuals.extreme_bg_color,
            visuals.window_stroke,
            egui::StrokeKind::Inside,
        );

        // A finished run takes over the pane until another frame is selected. Keyed by
        // `RESULT_KEY` so it cannot be confused with a frame index.
        let (path, key, included) = match &self.result {
            Some(result) => (result.clone(), RESULT_KEY, true),
            None => {
                let Some(stack) = &self.stack else {
                    label_in(
                        ui,
                        rect,
                        "open a folder to begin",
                        visuals.weak_text_color(),
                    );
                    return;
                };
                let Some(frame) = stack.frames.get(self.selected) else {
                    return;
                };
                (frame.path.clone(), self.selected, frame.included)
            }
        };

        let Some(info) = self.stack.as_ref().map(|s| s.info) else {
            return;
        };
        let view = self.view.get_or_insert_with(|| View::fit(rect, info));
        view.interact(ui, rect, info);
        let view = *view;

        // Decode what is actually on screen, at screen resolution. Zoomed in, that is a
        // small region near 1:1; zoomed out to fit, it is the whole frame downsampled,
        // exactly as before. Padded and snapped so a few pixels of pan reuse the texture
        // instead of starting a decode per frame of drag.
        let (region, target) = view.wanted_region(rect, info);
        self.preview.request(&path, key, region, target);

        match (&self.preview.texture, &self.preview.error) {
            (_, Some(error)) => label_in(ui, rect, error, visuals.error_fg_color),
            (Some(texture), None) => {
                // Drawn at wherever *its own* region maps to now, not stretched to the
                // pane. That is what lets a stale texture stay correct mid-pan: it slides
                // with the image rather than sitting still while the view moves under it.
                let placed = match self.preview.region {
                    Some(r) => view.source_to_screen(r),
                    None => rect,
                };
                let image = egui::Image::new(texture);
                // Dimmed to match the filmstrip, so the preview cannot contradict what
                // the badge says about whether this frame is in the run.
                let image = if included {
                    image
                } else {
                    image.tint(egui::Color32::from_white_alpha(EXCLUDED_OPACITY))
                };
                // Clipped to the pane: zoomed in, the drawn rect is far larger than the
                // pane and would otherwise paint over the filmstrip and parameter panel.
                let mut clipped = ui.new_child(egui::UiBuilder::new().max_rect(rect));
                clipped.set_clip_rect(rect);
                image.paint_at(&clipped, placed);

                if self.preview.is_loading() {
                    // The old view stays up while the new region decodes — blanking on
                    // every pan would flicker harder than it helps. Which makes the
                    // indicator the only thing distinguishing "this is what you asked
                    // for" from "this is still the last thing", so it gets its own
                    // backing rather than sitting as faint text over an arbitrary image.
                    loading_overlay(ui, rect, &visuals);
                }
            }
            (None, None) => loading_overlay(ui, rect, &visuals),
        }
    }

    fn parameters(&mut self, ui: &mut egui::Ui) {
        // Locked during a run: these are the settings it was started with, and letting
        // them move would show a configuration that does not match what is executing.
        let running = self.running();
        ui.add_enabled_ui(!running, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Method").strong());
                        egui::ComboBox::from_id_salt("method")
                            .selected_text(self.params.method.label())
                            .show_ui(ui, |ui| {
                                // Every method the app offers. The fusion *rule* is no
                                // longer chosen here — under `Local` the app still ships
                                // only `Select`, and `Blend` stays CLI-only for
                                // reproducing older eval-log rows.
                                for choice in Method::ALL {
                                    // Compared by method, not by value: each variant
                                    // carries its own parameters, so `selectable_value`'s
                                    // equality would call a tuned radius "not this entry"
                                    // and reset it on click.
                                    let selected = choice.token() == self.params.method.token();
                                    if ui
                                        .selectable_label(selected, choice.label())
                                        .on_hover_text(choice.summary())
                                        .clicked()
                                        && !selected
                                    {
                                        self.params.method = choice;
                                    }
                                }
                            });
                    });
                    // On screen, not only in a tooltip. Wavelet is a trade rather than a
                    // worse Local, and someone who picks it and gets a seamy macro shot
                    // needs to be able to read why without hunting for it.
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(self.params.method.summary())
                            .small()
                            .weak(),
                    );
                    ui.add_space(6.0);
                    ui.separator();

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Parameters").strong());
                        // Right-aligned, and dead once nothing has moved: the point is
                        // getting back to the rated configuration without having to
                        // remember eight numbers.
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let modified = self.params != Params::default();
                            if ui
                                .add_enabled(modified, egui::Button::new("Reset"))
                                .on_hover_text("Restore the defaults the pipeline is tuned to")
                                .clicked()
                            {
                                self.params = Params::default();
                            }
                        });
                    });
                    ui.add_space(6.0);

                    let p = &mut self.params;

                    ui.label("Registration");
                    ui.add(egui::Slider::new(&mut p.registration_level, 0..=5).text("level"));
                    ui.add_space(8.0);

                    // Everything below registration belongs to one method or the other,
                    // and each slider binds *into* the selected variant. A method that
                    // does not read a parameter therefore has no slider to draw, rather
                    // than a greyed one to remember to disable — the panel cannot
                    // disagree with the type.
                    match &mut p.method {
                        Method::Local { fusion } => {
                            ui.label("Focus measure");
                            ui.add(egui::Slider::new(&mut p.focus_radius, 1..=16).text("radius"));
                            ui.add_space(8.0);

                            ui.label("Weight refinement");
                            ui.add(
                                egui::Slider::new(&mut p.guide_radius, 1..=16).text("guide radius"),
                            );
                            ui.add(
                                egui::Slider::new(&mut p.guide_epsilon, 1e-5..=1e-1)
                                    .logarithmic(true)
                                    .text("epsilon"),
                            );
                            ui.horizontal(|ui| {
                                ui.label("guide:");
                                ui.selectable_value(
                                    &mut p.guide_space,
                                    GuideSpace::Linear,
                                    "linear",
                                );
                                ui.selectable_value(
                                    &mut p.guide_space,
                                    GuideSpace::Perceptual,
                                    "perceptual",
                                );
                            });
                            ui.add_space(8.0);

                            ui.label("Fusion");
                            if let FusionKind::Select { salience_radius } = &mut *fusion {
                                ui.add(
                                    egui::Slider::new(salience_radius, 0..=4)
                                        .text("salience radius"),
                                );
                            }
                        }
                        Method::Wavelet {
                            consistency_threshold,
                        } => {
                            ui.label("Coefficient selection");
                            ui.add(
                                egui::Slider::new(consistency_threshold, 0..=8)
                                    .text("consistency threshold"),
                            )
                            .on_hover_text(
                                "How many of a coefficient's 8 neighbours must agree \
                                 before its selected frame is overridden. 0 disables \
                                 consistency verification.",
                            );
                        }
                    }
                    // Outside the match: both methods build a multi-scale decomposition
                    // and both read this as its depth, which is what keeps them
                    // comparable at matched depth rather than at accidentally different
                    // ones. Confirmed from the constructors, not assumed.
                    ui.add_space(8.0);
                    ui.add(
                        egui::Slider::new(&mut p.pyramid_floor, 8..=128)
                            .text("decomposition floor"),
                    );
                });
        });
    }
}

/// One filmstrip entry: a selectable plate, with `contents` drawn inside it.
fn plate(
    ui: &mut egui::Ui,
    selected: bool,
    contents: impl FnOnce(&mut egui::Ui, egui::Rect),
) -> egui::Response {
    let size = egui::vec2(ui.available_width(), THUMBNAIL_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());

    if ui.is_rect_visible(rect) {
        // `interact`, not `interact_selectable`. The selectable variant expresses
        // selection by swapping the *fill* to the selection colour, and that is wrong
        // for a plate that holds an image: an excluded frame is drawn translucent, so
        // the fill behind it shows through and a selected+excluded plate reads as a
        // solid blue rectangle rather than as a dimmed thumbnail. Keeping the fill
        // neutral means "dimmed" always means dimmed, whatever else is true of the frame.
        let visuals = *ui.style().interact(&response);
        ui.painter().rect(
            rect,
            visuals.corner_radius,
            visuals.weak_bg_fill,
            visuals.bg_stroke,
            egui::StrokeKind::Inside,
        );
        contents(ui, rect.shrink(3.0));

        // Selection is a border drawn over the contents, for two reasons: egui's own
        // `interact_selectable` deliberately leaves `bg_stroke` alone (the line is
        // commented out in its source), and a thumbnail covers all but a ~3 px margin of
        // the plate, so any fill-based cue is invisible once real frames load. A border
        // survives the image being painted over it, and stays readable on a dimmed one.
        if selected {
            let stroke = ui.visuals().selection.stroke;
            ui.painter().rect_stroke(
                rect,
                visuals.corner_radius,
                egui::Stroke::new(stroke.width.max(2.0), stroke.color),
                egui::StrokeKind::Inside,
            );
        }
    }
    response
}

/// The include/exclude checkbox in a plate's corner.
///
/// Drawn from primitives rather than a glyph: the default font stack has no guaranteed
/// checkmark, and a missing glyph would render as tofu on whichever platform lacks it —
/// found by the user, not by us.
fn draw_badge(ui: &egui::Ui, rect: egui::Rect, included: bool, response: &egui::Response) {
    let visuals = ui.style().interact(response);
    let painter = ui.painter();

    painter.rect(
        rect,
        3.0,
        if included {
            ui.visuals().selection.bg_fill
        } else {
            // Opaque, not the plate's own fill: the badge sits on top of the thumbnail,
            // so it needs its own background or the image shows through and the state
            // becomes unreadable on a busy frame.
            ui.visuals().extreme_bg_color
        },
        visuals.bg_stroke,
        egui::StrokeKind::Inside,
    );

    if included {
        // A checkmark, two strokes, inset from the box.
        let stroke = egui::Stroke::new(2.0, ui.visuals().strong_text_color());
        let b = rect.shrink(4.0);
        let elbow = egui::pos2(b.left() + b.width() * 0.36, b.bottom());
        painter.line_segment([egui::pos2(b.left(), b.center().y), elbow], stroke);
        painter.line_segment([elbow, egui::pos2(b.right(), b.top())], stroke);
    }
}

/// A spinner and "Loading…" on an opaque pill, centred in `area`.
///
/// Backed rather than plain text because it is drawn over a photograph: any fixed text
/// colour is illegible against some frame, and this one has to be readable against all
/// of them.
fn loading_overlay(ui: &mut egui::Ui, area: egui::Rect, visuals: &egui::Visuals) {
    let pill = egui::Rect::from_center_size(area.center(), egui::vec2(148.0, 40.0));
    ui.painter().rect(
        pill,
        8.0,
        visuals.extreme_bg_color.gamma_multiply(0.92),
        visuals.window_stroke,
        egui::StrokeKind::Inside,
    );

    let spinner = egui::Rect::from_center_size(
        egui::pos2(pill.left() + 26.0, pill.center().y),
        egui::Vec2::splat(18.0),
    );
    ui.put(
        spinner,
        egui::Spinner::new()
            .size(18.0)
            .color(visuals.strong_text_color()),
    );
    ui.painter().text(
        egui::pos2(pill.left() + 46.0, pill.center().y),
        egui::Align2::LEFT_CENTER,
        "Loading…",
        egui::FontId::proportional(15.0),
        visuals.strong_text_color(),
    );
}

fn label_in(ui: &egui::Ui, rect: egui::Rect, text: &str, color: egui::Color32) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(12.0),
        color,
    );
}

/// Largest size with `content`'s aspect ratio that fits inside `bounds`.
fn fit(content: egui::Vec2, bounds: egui::Vec2) -> egui::Vec2 {
    let scale = (bounds.x / content.x).min(bounds.y / content.y).min(1.0);
    content * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use stackaroni_core::fusion::FusionKind;

    /// Every string the parameter panel actually draws, for one method.
    ///
    /// Renders the real `parameters` method into a headless `egui::Context` and reads the
    /// text back out of the emitted shapes. Nothing is stubbed: what this sees is what a
    /// window would show.
    fn panel_text(method: Method) -> Vec<String> {
        let mut app = App {
            params: Params {
                method,
                ..Params::default()
            },
            ..App::default()
        };

        fn collect(shape: &egui::epaint::Shape, into: &mut Vec<String>) {
            match shape {
                egui::epaint::Shape::Text(text) => into.push(text.galley.text().to_owned()),
                egui::epaint::Shape::Vec(shapes) => {
                    for s in shapes {
                        collect(s, into);
                    }
                }
                _ => {}
            }
        }

        let ctx = egui::Context::default();
        let mut found = Vec::new();
        // Two passes: egui lays widgets out against sizes it learned on the previous
        // frame, so the first is not representative.
        for _ in 0..2 {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ctx| {
                egui::Area::new("test".into()).show(ctx, |ui| app.parameters(ui));
            });
            found.clear();
            for clipped in &output.shapes {
                collect(&clipped.shape, &mut found);
            }
            // Font atlas updates are handed to a renderer in a real frame; there is none
            // here, and dropping them unapplied is a panic rather than a leak.
            output.textures_delta.clear();
        }
        found
    }

    /// Blend stays out of the app until it earns a place.
    ///
    /// The assertion that fails, with a reason, if someone routes the rule back into the
    /// dropdown without the measurement that would justify it. The pipeline supports it
    /// and the CLI exposes it — this is only about what a photographer is offered.
    ///
    /// Asserted over `Method::ALL`, which is what the combo box iterates, so it covers
    /// the rule actually reachable from the UI rather than a list kept beside it.
    #[test]
    fn the_app_does_not_offer_blend() {
        for method in Method::ALL {
            if let Method::Local { fusion } = method {
                assert!(
                    fusion.is_select(),
                    "blend measured worse on both real stacks (blossom 1/5 against 5/5); \
                     if that changed, add an eval-log row first"
                );
            }
        }
    }

    /// Every method the combo offers has to be one the run path can actually execute.
    ///
    /// `Method::ALL` drives the dropdown, and a variant added there without a matching
    /// arm in `run::pipeline` would be a menu entry that panics on Run.
    #[test]
    fn every_offered_method_is_runnable() {
        for method in Method::ALL {
            assert!(
                Method::from_token(method.token()) == Some(method),
                "{} does not round-trip through its CLI token",
                method.label()
            );
            assert!(!method.summary().is_empty(), "{}", method.label());
        }
        assert_eq!(Method::ALL.len(), Method::TOKENS.len());
    }

    /// Text drawn by the error modal, or empty if it does not appear.
    ///
    /// Same two-pass trick as [`panel_text`], and the same reason.
    fn modal_text(error: Option<String>) -> Vec<String> {
        let mut app = App {
            error,
            ..App::default()
        };

        fn collect(shape: &egui::epaint::Shape, into: &mut Vec<String>) {
            match shape {
                egui::epaint::Shape::Text(text) => into.push(text.galley.text().to_owned()),
                egui::epaint::Shape::Vec(shapes) => {
                    for s in shapes {
                        collect(s, into);
                    }
                }
                _ => {}
            }
        }

        let ctx = egui::Context::default();
        let mut found = Vec::new();
        for _ in 0..2 {
            let mut output = ctx.run_ui(egui::RawInput::default(), |ctx| app.error_modal(ctx));
            found.clear();
            for clipped in &output.shapes {
                collect(&clipped.shape, &mut found);
            }
            output.textures_delta.clear();
        }
        found
    }

    /// An error has to be acknowledged, not just printed somewhere.
    ///
    /// The export refusal originally drew into the status bar, where it read as passive
    /// text beside the stack name — easy to miss for a message whose whole job is to say
    /// that the thing you just asked for did not happen.
    #[test]
    fn an_error_appears_in_a_modal_with_a_dismiss_button() {
        let drawn = modal_text(Some("the sky is falling".into()));
        let has = |needle: &str| drawn.iter().any(|t| t.contains(needle));

        assert!(has("the sky is falling"), "message missing: {drawn:?}");
        assert!(has("OK"), "no way to dismiss it: {drawn:?}");

        // And it stays out of the way when there is nothing wrong.
        assert!(
            modal_text(None).is_empty(),
            "the modal must not draw without an error"
        );
    }

    /// The app's half of the export guard.
    ///
    /// `core` proves `ensure_output_outside_stack` decides correctly; this proves the app
    /// *asks* it, and asks it about the right directory — the frames' own, not a stored
    /// field that could drift. Worth having separately because the two halves failed
    /// independently in the incident that motivated them: the check existed in `core` as
    /// a name allowlist, and the export path never consulted anything at all.
    ///
    /// The save dialog cannot be driven from a test, and on this machine cannot be driven
    /// at all — `osascript` is denied assistive access, so the window cannot even be
    /// raised. Splitting the decision out of the dialog is what makes the guard testable
    /// rather than merely inspected.
    #[test]
    fn export_is_refused_into_the_stacks_own_directory() {
        let dir = tempfile::tempdir().unwrap();
        let frames = dir.path().join("blossom");
        std::fs::create_dir(&frames).unwrap();
        for name in ["a.tif", "b.tif"] {
            let info = FrameInfo {
                width: 32,
                height: 16,
                samples: 3,
                bits_per_sample: 16,
            };
            stackaroni_core::tiff_io::write_rgb16_srgb(&frames.join(name), info, |_, row| {
                row.fill(0.5);
                Ok(())
            })
            .unwrap();
        }
        let stack = Stack::load(&frames, Arc::new(AtomicU64::new(0))).unwrap();

        // The filename the app itself suggests, which is what made this easy to do.
        let refused = frames.join("blossom_stacked.tif");
        let message = App::refusal_for(Some(&stack), &refused)
            .expect("saving into the frames' own directory must be refused");
        assert!(
            message.contains("extra frame"),
            "the refusal must say why, not just refuse: {message}"
        );

        // Everywhere else is still allowed, including a path that does not exist yet.
        for allowed in [
            dir.path().join("blossom_stacked.tif"),
            dir.path().join("exports/blossom_stacked.tif"),
        ] {
            assert!(
                App::refusal_for(Some(&stack), &allowed).is_none(),
                "should have been allowed: {}",
                allowed.display()
            );
        }

        // No stack loaded means nothing to protect, and export must not be blocked.
        assert!(App::refusal_for(None, &refused).is_none());
    }

    /// Does the panel actually swap, on screen, when the method changes?
    ///
    /// The byte-identical gates compare pipeline *output* and would pass whatever this
    /// panel drew — a focus-radius slider left visible under Wavelet is invisible to
    /// them. This is the only automated check that looks at what is on screen, which is
    /// why it asserts absence as well as presence: a knob that drives a stage the
    /// selected method never runs is the specific failure the swap exists to prevent,
    /// and a test that only checked Local would never see it.
    #[test]
    fn the_panel_swaps_with_the_method() {
        let local = panel_text(defaults::METHOD);
        let wavelet = panel_text(Method::Wavelet {
            consistency_threshold: defaults::CONSISTENCY_THRESHOLD,
        });

        let has = |texts: &[String], needle: &str| texts.iter().any(|t| t.contains(needle));

        // Wavelet runs no focus or weights stage, so none of their knobs may appear.
        for dead in ["radius", "epsilon", "guide", "salience"] {
            assert!(
                !has(&wavelet, dead),
                "Wavelet runs no stage that reads {dead:?}, so it must not be on screen; \
                 panel drew {wavelet:?}"
            );
        }
        assert!(
            has(&wavelet, "consistency threshold"),
            "Wavelet reads the consistency threshold and must offer it; drew {wavelet:?}"
        );

        // ...and the converse, so this cannot pass by drawing an empty panel.
        for live in ["radius", "epsilon", "guide", "salience radius"] {
            assert!(
                has(&local, live),
                "Local reads {live:?} and must offer it; panel drew {local:?}"
            );
        }
        assert!(
            !has(&local, "consistency threshold"),
            "Local has no consistency verification; panel drew {local:?}"
        );

        // Read by both methods as the depth of their decomposition, so it must survive
        // the swap in both directions.
        for (name, texts) in [("Local", &local), ("Wavelet", &wavelet)] {
            assert!(
                has(texts, "decomposition floor"),
                "{name} builds a multi-scale decomposition and reads the floor; \
                 panel drew {texts:?}"
            );
        }

        // The label and the trade-off sentence, so a silent copy regression is caught too.
        assert!(has(&local, defaults::METHOD.label()));
        assert!(has(&wavelet, "Wavelet"));
        // The rating and the mechanism, not a particular phrase: the two methods are not
        // peers, and a chooser that offers them as equals is the thing this guards
        // against. Was "seam" until 2026-08-15, when wavelet was re-rated 2/4/2 and the
        // trade-off became the defocus spread effect rather than seaming.
        assert!(
            has(&wavelet, "2/4/2"),
            "wavelet's rating must be on screen; panel drew {wavelet:?}"
        );
        assert!(
            has(&wavelet, "defocus spread"),
            "wavelet's trade-off must be on screen; panel drew {wavelet:?}"
        );
        assert!(
            has(&local, "5/5/5"),
            "local's rating must be on screen; panel drew {local:?}"
        );
    }

    /// The fusion rule still swaps the salience slider *within* `Local`.
    ///
    /// The app does not offer `Blend` in the dropdown, but the panel branch that hides
    /// its slider is still live code, and it is what a restored rule would depend on.
    #[test]
    fn the_salience_slider_follows_the_fusion_rule() {
        let select = panel_text(defaults::METHOD);
        let blend = panel_text(Method::Local {
            fusion: FusionKind::Blend,
        });

        let has = |texts: &[String], needle: &str| texts.iter().any(|t| t.contains(needle));
        assert!(has(&select, "salience radius"));
        assert!(
            !has(&blend, "salience radius"),
            "Blend ignores the salience radius entirely, so it must not be on screen; \
             panel drew {blend:?}"
        );
    }
}
