//! Main stacking view — static layout skeleton.
//!
//! Follows the reference layout named in `CLAUDE.md`: filmstrip of frame thumbnails on
//! the left, large preview in the centre, parameter panel on the right. That is the
//! established shape for this category of tool (Lightroom's Develop module, darktable,
//! Zerene Stacker, Helicon Focus), so it is copied rather than designed.
//!
//! Fixed panels, not `egui_dock`. A dockable layout is one more unknown and buys nothing
//! until a fixed one is shown to be limiting in practice.
//!
//! Folder loading, the filmstrip and the preview are real. The parameter widgets are
//! not: they edit local state that nothing reads, and exist so the panel has real
//! controls at real sizes. The enums they use stay local placeholders rather than being
//! bound to the core types they mirror, because nothing consumes them yet — they get
//! bound when a run does.

mod stack;

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use eframe::egui;
use stack::{Preview, Stack, Thumbnail};

fn main() -> eframe::Result {
    eframe::run_native(
        "Stackaroni",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

/// Empty-state plates, so the filmstrip has its real shape before a folder is opened.
const PLACEHOLDER_FRAMES: usize = 8;

/// Height of a filmstrip entry. Thumbnails are fitted inside this, letterboxed.
const THUMBNAIL_HEIGHT: f32 = 88.0;

/// Side of the include/exclude checkbox drawn in a plate's corner.
const BADGE: f32 = 16.0;

/// Opacity of an excluded thumbnail.
///
/// The first attempt used 23%, which did not read as "excluded" — it read as a grey
/// wash, indistinguishable from a rendering fault. Dimming is reinforcement here, not
/// the signal: the unchecked badge carries the meaning, so this only has to be visibly
/// different while leaving the frame recognizable enough to change your mind about.
const EXCLUDED_OPACITY: u8 = 150;

#[derive(PartialEq, Eq, Clone, Copy)]
enum GuideSpace {
    Linear,
    Perceptual,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum FusionRule {
    Blend,
    Select,
}

/// The handful of parameters the CLI exposes, mirrored here at their current defaults.
struct Params {
    registration_level: u32,
    focus_radius: u32,
    guide_radius: u32,
    guide_epsilon: f32,
    guide_space: GuideSpace,
    fusion: FusionRule,
    salience_radius: u32,
    pyramid_floor: u32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            registration_level: 3,
            focus_radius: 4,
            guide_radius: 4,
            guide_epsilon: 1e-4,
            guide_space: GuideSpace::Perceptual,
            fusion: FusionRule::Select,
            salience_radius: 2,
            pyramid_floor: 32,
        }
    }
}

#[derive(Default)]
struct App {
    selected: usize,
    params: Params,
    stack: Option<Stack>,
    preview: Preview,
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

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
    }
}

impl App {
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Open folder…").clicked() {
                self.open_folder();
            }
            ui.separator();
            // Still disabled: nothing behind them yet, and a button that does nothing is
            // worse than one that is visibly not ready.
            ui.add_enabled(false, egui::Button::new("Run stack"));
            ui.add_enabled(false, egui::Button::new("Export…"));
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
                None => {
                    ui.label(egui::RichText::new("Ready").weak());
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new("no pipeline wired").weak());
            });
        });
        ui.add_space(2.0);
    }

    fn filmstrip(&mut self, ui: &mut egui::Ui) {
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
        let heading = match &self.stack {
            Some(stack) => stack
                .frames
                .get(self.selected)
                .and_then(|f| f.path.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            None => String::new(),
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

        // Sized from the pane rather than a constant, so a large window gets a sharper
        // preview and a small one does not decode pixels it will immediately throw away.
        // Doubled for a little headroom against resizing and HiDPI.
        let target = (rect.width() * 2.0).clamp(512.0, 3000.0) as usize;
        self.preview.request(&frame.path, self.selected, target);

        match (&self.preview.texture, &self.preview.error) {
            (_, Some(error)) => label_in(ui, rect, error, visuals.error_fg_color),
            (Some(texture), None) => {
                let size = fit(texture.size_vec2(), rect.shrink(8.0).size());
                let image = egui::Image::new(texture);
                // Dimmed to match the filmstrip, so the preview cannot contradict what
                // the badge says about whether this frame is in the run.
                let image = if frame.included {
                    image
                } else {
                    image.tint(egui::Color32::from_white_alpha(EXCLUDED_OPACITY))
                };
                image.paint_at(ui, egui::Rect::from_center_size(rect.center(), size));

                if self.preview.is_loading() {
                    // The old frame stays up while the new one decodes — blanking the
                    // pane on every selection change would flicker harder than it helps.
                    // Which makes the indicator the only thing distinguishing "this is
                    // the frame you picked" from "this is still the previous one", so it
                    // gets its own backing and the centre of the pane rather than sitting
                    // as faint text over an arbitrary image.
                    loading_overlay(ui, rect, &visuals);
                }
            }
            (None, None) => loading_overlay(ui, rect, &visuals),
        }
    }

    fn parameters(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Parameters").strong());
                ui.add_space(6.0);

                let p = &mut self.params;

                ui.label("Registration");
                ui.add(egui::Slider::new(&mut p.registration_level, 0..=5).text("level"));
                ui.add_space(8.0);

                ui.label("Focus measure");
                ui.add(egui::Slider::new(&mut p.focus_radius, 1..=16).text("radius"));
                ui.add_space(8.0);

                ui.label("Weight refinement");
                ui.add(egui::Slider::new(&mut p.guide_radius, 1..=16).text("guide radius"));
                ui.add(
                    egui::Slider::new(&mut p.guide_epsilon, 1e-5..=1e-1)
                        .logarithmic(true)
                        .text("epsilon"),
                );
                ui.horizontal(|ui| {
                    ui.label("guide:");
                    ui.selectable_value(&mut p.guide_space, GuideSpace::Linear, "linear");
                    ui.selectable_value(&mut p.guide_space, GuideSpace::Perceptual, "perceptual");
                });
                ui.add_space(8.0);

                ui.label("Fusion");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut p.fusion, FusionRule::Blend, "blend");
                    ui.selectable_value(&mut p.fusion, FusionRule::Select, "select");
                });
                // Only the selection rule reads this, matching the CLI, where it is
                // documented as ignored by `blend`.
                ui.add_enabled(
                    p.fusion == FusionRule::Select,
                    egui::Slider::new(&mut p.salience_radius, 0..=4).text("salience radius"),
                );
                ui.add(egui::Slider::new(&mut p.pyramid_floor, 8..=128).text("pyramid floor"));
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
