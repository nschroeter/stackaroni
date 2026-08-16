"""Turn the hand-made stackaroni.png into shippable icon assets.

The source is a 1254x1254 RGB PNG with a *white* background: the rounded square is
painted on white rather than cut out, and the file has no alpha channel at all. Shipped
as-is, macOS would draw a white square in the Dock with the icon inside it.

So the plate is measured, cropped, and given an alpha mask matching the corner radius
the artwork already has (~20.5% of its width — close enough to Apple's 22.45% that
re-cutting to Apple's would clip the drawn corner rather than match it).

Output follows the macOS Big Sur grid: an 824x824 plate centred on a 1024x1024
transparent canvas, which is what makes it sit at the same visual size as every other
Dock icon.
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFilter

SRC, OUT = sys.argv[1], sys.argv[2]

CANVAS = 1024
PLATE = 824
SS = 4
# Measured off the source: the straight edge begins 228 px down a 1123 px-wide plate.
RADIUS_FRACTION = 0.205
# Eats the antialiased white fringe left where the drawn plate met the white ground.
TRIM = 2.0

WHITE = 235


def plate_bbox(img):
    """The drawn rounded square, found as everything that is not the white ground."""
    px = img.load()
    w, h = img.size

    def fg(x, y):
        r, g, b = px[x, y]
        return not (r > WHITE and g > WHITE and b > WHITE)

    xs = [x for x in range(w) if any(fg(x, y) for y in range(0, h, 3))]
    ys = [y for y in range(h) if any(fg(x, y) for x in range(0, w, 3))]
    return xs[0], ys[0], xs[-1] + 1, ys[-1] + 1


def exterior_mask(img):
    """White ground reachable from the border, as a mask of what to cut away.

    Flood-filled rather than drawn: the plate's corners are a squircle (continuous
    curvature, as Apple's are), so a circular-radius rounded rectangle cannot follow
    them — it leaves a white fringe along the middle of each arc, which is exactly what
    the first attempt produced. The artwork's own outline is the only accurate one.
    """
    w, h = img.size
    flood = img.copy()
    ImageDraw.floodfill(flood, (0, 0), (255, 0, 255), thresh=60)

    px = flood.load()
    mask = Image.new("L", (w, h), 255)
    mpx = mask.load()
    for y in range(h):
        for x in range(w):
            if px[x, y] == (255, 0, 255):
                mpx[x, y] = 0
    return mask


def cut_out(img):
    """Crop to the plate and replace the white ground with transparency."""
    box = plate_bbox(img)

    mask = exterior_mask(img).crop(box)
    # Erode by a pixel, then soften: the flood fill stops where antialiasing has already
    # mixed white into the edge, so the outermost ring of kept pixels is part ground.
    mask = mask.filter(ImageFilter.MinFilter(5))
    mask = mask.filter(ImageFilter.GaussianBlur(1.0))

    plate = img.crop(box).convert("RGBA")
    plate.putalpha(mask)
    plate = plate.resize((PLATE, PLATE), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    canvas.paste(plate, ((CANVAS - PLATE) // 2, (CANVAS - PLATE) // 2), plate)
    return canvas, box


src = Image.open(SRC).convert("RGB")
icon, box = cut_out(src)
os.makedirs(OUT, exist_ok=True)
icon.save(f"{OUT}/icon-1024.png")
print(f"source plate bbox {box}")

# A proof sheet: the cut-out over white, black and magenta. A leftover fringe or a
# clipped corner is invisible on one background and obvious on another.
proof = Image.new("RGB", (CANVAS // 2 * 3, CANVAS // 2), (0, 0, 0))
for i, bg in enumerate([(255, 255, 255), (0, 0, 0), (255, 0, 255)]):
    tile = Image.new("RGB", (CANVAS, CANVAS), bg)
    tile.paste(icon, (0, 0), icon)
    proof.paste(tile.resize((CANVAS // 2, CANVAS // 2), Image.LANCZOS), (i * CANVAS // 2, 0))
proof.save(f"{OUT}/proof.png")

# Corners at 1:1, where a fringe actually shows.
corner = Image.new("RGB", (420, 210), (255, 0, 255))
tile = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
tile.paste(icon, (0, 0), icon)
corner.paste(tile.crop((90, 90, 300, 300)), (0, 0))
tile2 = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
tile2.paste(icon, (0, 0), icon)
corner.paste(tile2.crop((90, 90, 300, 300)), (210, 0))
corner.save(f"{OUT}/corner.png")

# The sizes that decide legibility.
strip = Image.new("RGBA", (16 + 32 + 64 + 128 + 60, 128), (0, 0, 0, 0))
x = 0
for s in (16, 32, 64, 128):
    strip.paste(icon.resize((s, s), Image.LANCZOS), (x, (128 - s) // 2))
    x += s + 20
strip.save(f"{OUT}/sizes.png")

# .iconset for iconutil: every size macOS asks for, at 1x and 2x.
iconset = f"{OUT}/Stackaroni.iconset"
os.makedirs(iconset, exist_ok=True)
for size in (16, 32, 128, 256, 512):
    icon.resize((size, size), Image.LANCZOS).save(f"{iconset}/icon_{size}x{size}.png")
    icon.resize((size * 2, size * 2), Image.LANCZOS).save(f"{iconset}/icon_{size}x{size}@2x.png")
print(f"wrote {iconset}")
