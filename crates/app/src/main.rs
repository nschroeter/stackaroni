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
//! Folder loading and the filmstrip are real; everything else is still placeholder. The
//! preview pane and the parameter widgets are not wired to anything — the parameters
//! exist so the panel has real controls at real sizes, and nothing reads them yet.
//!
//! The enums below stay local placeholders rather than being bound to the core types
//! they mirror, because nothing consumes them yet. They get bound when a run does.

mod stack;

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use eframe::egui;
use stack::{Stack, Thumbnail};

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
        let mut clicked = None;
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
                                    image.tint(egui::Color32::from_white_alpha(60))
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

                    let name = frame
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let hint = match &frame.thumbnail {
                        Thumbnail::Failed(error) => error.clone(),
                        _ if included => format!("{name}\nclick to exclude"),
                        _ => format!("{name}\nexcluded — click to include"),
                    };
                    if response.on_hover_text(hint).clicked() {
                        clicked = Some(index);
                    }
                    ui.add_space(4.0);
                }
            });

        // Applied after the loop because the closure borrows the stack immutably.
        if let Some(index) = clicked {
            if let Some(stack) = &mut self.stack {
                stack.frames[index].included = !stack.frames[index].included;
            }
            self.selected = index;
        }
    }

    fn preview(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(format!("Frame {:03}", self.selected)).strong());
        });
        ui.add_space(6.0);

        // Fills whatever the panels leave, which is the point of the centre pane: it is
        // where a 50 MP frame has to be legible.
        let rect = ui.available_rect_before_wrap();
        let painter = ui.painter();
        let visuals = ui.visuals();
        painter.rect(
            rect,
            visuals.window_corner_radius,
            visuals.extreme_bg_color,
            visuals.window_stroke,
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "preview",
            egui::FontId::proportional(16.0),
            visuals.weak_text_color(),
        );
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
        let visuals = ui.style().interact_selectable(&response, selected);
        ui.painter().rect(
            rect,
            visuals.corner_radius,
            visuals.weak_bg_fill,
            visuals.bg_stroke,
            egui::StrokeKind::Inside,
        );
        contents(ui, rect.shrink(3.0));
    }
    response
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
