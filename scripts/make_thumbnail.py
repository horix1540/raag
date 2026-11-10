"""Generate the project thumbnail required by the college portal.

Drawn programmatically with Pillow rather than in a design tool so the asset is
reproducible and version-controlled alongside the code.

    python scripts/make_thumbnail.py
    -> assets/thumbnail.png  (1200x630)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "thumbnail.png"

W, H = 1200, 630

# Deep navy so the accent colours carry; the portal renders cards on dark chrome.
BG_TOP = (11, 18, 32)
BG_BOTTOM = (17, 29, 51)
WHITE = (245, 248, 252)
MUTED = (146, 163, 189)
DIM = (99, 116, 143)
TEAL = (45, 212, 191)
AMBER = (251, 191, 36)
RED = (248, 113, 113)
GREEN = (74, 222, 128)
PANEL = (23, 35, 58)

FONT_DIR = Path("C:/Windows/Fonts")
FONTS = {
    "black": "seguibl.ttf",
    "semibold": "seguisb.ttf",
    "regular": "segoeui.ttf",
    "mono": "consola.ttf",
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / FONTS[kind]
    if path.exists():
        return ImageFont.truetype(str(path), size)
    for fallback in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def gradient(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), BG_TOP)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        draw.line(
            [(0, y), (width, y)],
            fill=tuple(round(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM, strict=True)),
        )
    return img


def text_w(draw: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont) -> int:
    return int(draw.textbbox((0, 0), s, font=f)[2])


def pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    *,
    accent: tuple[int, int, int] = DIM,
    fill: tuple[int, int, int] = PANEL,
    text_colour: tuple[int, int, int] = WHITE,
    size: int = 21,
    kind: str = "semibold",
    border: int = 2,
) -> None:
    x0, y0, x1, y1 = box
    radius = (y1 - y0) // 2
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=accent, width=border)
    f = font(kind, size)
    tw = text_w(draw, label, f)
    bbox = draw.textbbox((0, 0), label, font=f)
    th = bbox[3] - bbox[1]
    draw.text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - bbox[1]),
        label,
        font=f,
        fill=text_colour,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    width: int = 3,
    head: int = 9,
) -> None:
    draw.line([start, end], fill=colour, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):  # horizontal
        d = 1 if x1 >= x0 else -1
        draw.polygon(
            [(x1, y1), (x1 - d * head, y1 - head // 2 - 1), (x1 - d * head, y1 + head // 2 + 1)],
            fill=colour,
        )
    else:  # vertical
        d = 1 if y1 >= y0 else -1
        draw.polygon(
            [(x1, y1), (x1 - head // 2 - 1, y1 - d * head), (x1 + head // 2 + 1, y1 - d * head)],
            fill=colour,
        )


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    *,
    dash: int = 7,
    gap: int = 6,
    width: int = 3,
) -> None:
    """Dashed straight segment, horizontal or vertical."""
    x0, y0 = start
    x1, y1 = end
    step = dash + gap
    if x0 == x1:
        lo, hi = min(y0, y1), max(y0, y1)
        for y in range(lo, hi, step):
            draw.line([(x0, y), (x0, min(y + dash, hi))], fill=colour, width=width)
    else:
        lo, hi = min(x0, x1), max(x0, x1)
        for x in range(lo, hi, step):
            draw.line([(x, y0), (min(x + dash, hi), y0)], fill=colour, width=width)


def fail_branch(
    draw: ImageDraw.ImageDraw,
    gate_x: int,
    gate_bottom: int,
    elbow_y: int,
    target_x: int,
    target_top: int,
    colour: tuple[int, int, int],
) -> None:
    """Elbow connector from a gate's underside into the shared ABSTAIN pill.

    Both gates fail into the same outcome, so the two branches converge rather
    than trailing off into empty space.
    """
    dashed_line(draw, (gate_x, gate_bottom), (gate_x, elbow_y), colour)
    dashed_line(draw, (gate_x, elbow_y), (target_x, elbow_y), colour)
    draw.ellipse((gate_x - 4, elbow_y - 4, gate_x + 4, elbow_y + 4), fill=colour)
    arrow(draw, (target_x, elbow_y), (target_x, target_top - 2), colour, width=3, head=10)


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    f_kicker = font("mono", 19)
    draw.text((64, 56), "ARTIFICIAL INTELLIGENCE & ML SYSTEMS", font=f_kicker, fill=TEAL)

    f_title = font("black", 62)
    f_sub = font("semibold", 40)
    draw.text((62, 96), "Confidence-Aware RAG", font=f_title, fill=WHITE)
    draw.text((64, 172), "with Hallucination Detection", font=f_sub, fill=MUTED)

    # Accent rule under the title block.
    draw.rounded_rectangle((64, 232, 200, 237), radius=3, fill=TEAL)


def draw_pipeline(draw: ImageDraw.ImageDraw) -> None:
    """The two-gate flow: the one idea the thumbnail has to communicate."""
    y0, y1 = 288, 344
    nodes = [
        ("Question", 64, 196, DIM),
        ("Retrieve", 222, 354, DIM),
        ("GATE A", 380, 536, AMBER),
        ("Generate", 562, 706, DIM),
        ("GATE B", 732, 888, AMBER),
        ("Answer", 914, 1058, GREEN),
    ]
    for label, x0, x1, accent in nodes:
        is_gate = label.startswith("GATE")
        pill(
            draw,
            (x0, y0, x1, y1),
            label,
            accent=accent,
            fill=(38, 32, 20) if is_gate else PANEL,
            text_colour=accent if is_gate else WHITE,
            size=22 if is_gate else 21,
            kind="black" if is_gate else "semibold",
            border=3 if is_gate else 2,
        )

    mid = (y0 + y1) // 2
    for current, following in zip(nodes, nodes[1:], strict=False):
        arrow(draw, (current[2] + 6, mid), (following[1] - 8, mid), DIM, width=3)

    # Both gates fail downward into the same abstain outcome.
    abstain = (520, 462, 748, 520)
    pill(
        draw,
        abstain,
        "ABSTAIN",
        accent=RED,
        fill=(45, 22, 26),
        text_colour=RED,
        size=23,
        kind="black",
        border=3,
    )
    elbow_y = 420
    fail_branch(draw, 458, y1 + 6, elbow_y, 634, abstain[1], RED)
    fail_branch(draw, 810, y1 + 6, elbow_y, 634, abstain[1], RED)
    f_fail = font("mono", 16)
    draw.text((468, 372), "fail", font=f_fail, fill=RED)
    draw.text((820, 372), "fail", font=f_fail, fill=RED)


def draw_footer(draw: ImageDraw.ImageDraw) -> None:
    f_line = font("semibold", 27)
    draw.text(
        (64, 542),
        "Retrieval checked before generation. Groundedness checked after.",
        font=f_line,
        fill=WHITE,
    )
    f_small = font("regular", 20)
    draw.text(
        (64, 578),
        "LangChain · ChromaDB · sentence-transformers · DeBERTa NLI · DeepEval · Streamlit",
        font=f_small,
        fill=DIM,
    )

    # Right-hand chip, aligned with the kicker row so it clears the title band.
    f_chip = font("mono", 17)
    label = "composite = weighted harmonic mean(A, B)"
    cw = text_w(draw, label, f_chip)
    x1 = W - 64
    x0 = x1 - cw - 34
    draw.rounded_rectangle((x0, 44, x1, 86), radius=21, fill=PANEL, outline=TEAL, width=2)
    draw.text((x0 + 17, 56), label, font=f_chip, fill=TEAL)

    f_note = font("regular", 18)
    note = "either gate fails  ->  no answer is shown"
    nw = text_w(draw, note, f_note)
    draw.text((x1 - nw, 100), note, font=f_note, fill=DIM)


def main() -> int:
    img = gradient(W, H)
    draw = ImageDraw.Draw(img)

    draw_header(draw)
    draw_pipeline(draw)
    draw_footer(draw)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)

    # A square crop as well: some portals render avatars rather than banners.
    square = img.crop((0, 0, H, H)).resize((600, 600), Image.LANCZOS)
    square.save(OUT.with_name("thumbnail_square.png"), "PNG", optimize=True)

    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {W}x{H})")
    print(f"wrote {OUT.with_name('thumbnail_square.png')} (600x600)")
    return 0


if __name__ == "__main__":
    sys.exit(main())



