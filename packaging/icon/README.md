# Icon

A mantis over three stacked planes — the subject and the process.

| File | What it is |
|---|---|
| `master.png` | The original artwork, 1254², RGB, **white background, no alpha** |
| `stackaroni.png` | The shippable icon: 1024², alpha, plate on the macOS grid |
| `stackaroni-256.png` | The same, at the size window and taskbar icons are drawn |
| `stackaroni.ico` | The Windows executable's own icon, 16 through 256 |
| `cut-out.py` | Derives the PNGs from the master |

## Why the master cannot be shipped as-is

It has no alpha channel and the rounded square is painted *on* white rather than cut out
of it. Used directly, macOS would draw a white square in the Dock with the icon inside
it.

`cut-out.py` finds the plate, replaces the white ground with transparency, and centres
the result as an 824² plate on a 1024² canvas — the macOS Big Sur grid, which is what
makes it sit at the same visual size as every other Dock icon.

**The outline is flood-filled from the artwork, not drawn.** The first attempt masked
with a rounded rectangle at the measured corner radius (~20.5% of the plate) and left a
white fringe along the middle of every corner arc: the drawn corners are a *squircle*,
continuous curvature, as Apple's are, and no circular radius follows that curve. The
artwork's own outline is the only accurate one.

The trade is that the silhouette inherits the artwork's slight irregularities — the
straight edges wander by a pixel or two. Invisible at Dock size, faintly visible in a
Finder preview at 512. Snapping to a true superellipse would fix it and clip a little of
what was drawn.

## Regenerating

```sh
python3 packaging/icon/cut-out.py packaging/icon/master.png /tmp/icon
```

It writes the assets plus proof sheets: the cut-out over white, black and magenta, and
the corners at 1:1. A fringe or a clipped corner is invisible on one background and
obvious on another, which is the point of checking three.

## Where each one is used

**macOS** — `.github/workflows/release.yml` builds `AppIcon.icns` from
`stackaroni.png` with `sips` and `iconutil` at release time, into
`Stackaroni.app/Contents/Resources/`. Ten sizes, 16 through 512 at 1× and 2×. Nothing
binary is committed for this; one PNG in the repository, every size derived.

**Windows and Linux** — `stackaroni-256.png` is embedded in the binary with
`include_bytes!` and attached via `ViewportBuilder::with_icon`. That covers the window
and the taskbar. macOS ignores it and uses the bundle icon.

**Windows, additionally** — the icon Explorer draws for `stackaroni-app.exe` itself is a
resource linked into the binary, which nothing at runtime can set. `crates/app/build.rs`
compiles `stackaroni.ico` into it with `winresource`. Only on a Windows host, because it
drives `rc.exe` from the Windows SDK; the release workflow builds each platform on its
own runner, so that always holds where it matters.

Regenerate the `.ico` after changing the artwork:

```sh
magick packaging/icon/stackaroni.png \
  -define icon:auto-resize=256,128,64,48,32,16 packaging/icon/stackaroni.ico
```

Committed rather than built, so neither the build nor CI needs ImageMagick. It is the
one derived binary in here that is not generated at release time, and the reason is that
`build.rs` runs before any tooling could produce it.
