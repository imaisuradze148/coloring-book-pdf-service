"""
Coloring Book PDF Service
Endpoints:
  GET  /                — health check
  POST /build-pdf       — full coloring book interior PDF
  POST /build-cover     — 2000x2000 PNG Etsy listing thumbnail (uses page 1 only)
  POST /build-preview   — 2000x2000 PNG showing 4 sample pages
  POST /build-readme    — customer-facing thank you + terms PDF (uses title + page_count)

Cover layout: title at top in narrow band, cover art fills the rest of the canvas,
cream background to blend with Gemini-generated cover art (matches Etsy reference style).
"""

import base64
import os
import re
from contextlib import asynccontextmanager
from io import BytesIO
from urllib.request import urlopen

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageChops, ImageDraw, ImageFont
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

SERVICE_KEY = os.getenv("SERVICE_KEY", "")
FONT_DIR = "/tmp/fonts"
INTER_BOLD = f"{FONT_DIR}/Inter-Bold.ttf"
INTER_REGULAR = f"{FONT_DIR}/Inter-Regular.ttf"

# Print sizing
PRINT_DPI = int(os.getenv("PRINT_DPI", "300"))
LETTER_MAX_PX = int(11 * PRINT_DPI)
BINARIZE_THRESHOLD = int(os.getenv("BINARIZE_THRESHOLD", "200"))

# Cover canvas color — matches the cream backgrounds Gemini produces for cover art.
# If you want pure white, change to "white". If you want a different cream, e.g.,
# warmer or cooler, swap the hex.
COVER_BG = os.getenv("COVER_BG", "#FFF6E8")


def ensure_fonts() -> bool:
    if os.path.exists(INTER_BOLD) and os.path.exists(INTER_REGULAR):
        return True
    try:
        os.makedirs(FONT_DIR, exist_ok=True)
        urls = {
            INTER_BOLD: "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Bold.ttf",
            INTER_REGULAR: "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.ttf",
        }
        for path, url in urls.items():
            if not os.path.exists(path):
                with urlopen(url, timeout=20) as resp:
                    with open(path, "wb") as f:
                        f.write(resp.read())
        return True
    except Exception:
        return False


def get_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    primary = INTER_BOLD if bold else INTER_REGULAR
    if os.path.exists(primary):
        try:
            return ImageFont.truetype(primary, size)
        except Exception:
            pass
    fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in fallbacks:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_fonts()
    yield


app = FastAPI(title="Coloring Book PDF Service", lifespan=lifespan)


class PageImage(BaseModel):
    filename: str
    data: str


class BuildRequest(BaseModel):
    images: list[PageImage] = []
    title: str = ""
    page_count: int | None = None


def auth_check(key: str) -> None:
    if SERVICE_KEY and key != SERVICE_KEY:
        raise HTTPException(401, "Invalid or missing X-Service-Key header")


def extract_page_num(filename: str) -> int:
    m = re.search(r"page-(\d+)", filename)
    return int(m.group(1)) if m else 9999


def decode_grayscale(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("L")


def decode_color(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def crop_to_content(img: Image.Image, padding_pct: float = 0.02, threshold: int = 240) -> Image.Image:
    gray = img.convert("L") if img.mode != "L" else img
    inverted = ImageChops.invert(gray.point(lambda p: 255 if p > threshold else 0))
    bbox = inverted.getbbox()
    if not bbox:
        return img
    pad = int(max(img.size) * padding_pct)
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.size[0], bbox[2] + pad)
    bottom = min(img.size[1], bbox[3] + pad)
    return img.crop((left, top, right, bottom))


def shrink_to_print_size(img: Image.Image, max_dim: int = LETTER_MAX_PX) -> Image.Image:
    longest = max(img.size)
    if longest <= max_dim:
        return img
    ratio = max_dim / longest
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    return img.resize(new_size, Image.LANCZOS)


def binarize(img: Image.Image, threshold: int = BINARIZE_THRESHOLD) -> Image.Image:
    gray = img.convert("L") if img.mode != "L" else img
    return gray.point(lambda p: 255 if p > threshold else 0).convert("1")


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-") or "coloring-book"


def shorten_title_for_cover(title: str) -> str:
    """Strip common Etsy SEO phrases that clutter the cover but aren't needed visually.

    Your Etsy listing title is optimized for search (long, keyword-stuffed).
    The cover image needs a short, punchy display title that reads at thumbnail scale.
    This function extracts the core niche from a typical SEO title pattern.

    Examples:
        'Gothic Coloring Book for Adults, Dark Engraving Art' -> 'GOTHIC'
        'Teen Girls Positive Affirmations Coloring Book' -> 'TEEN GIRLS POSITIVE AFFIRMATIONS'
        'Dragon Coloring Book for Kids Ages 4-7' -> 'DRAGON'
        'Animal Detectives Coloring Book' -> 'ANIMAL DETECTIVES'
        'Gothic Architecture Coloring Book for Adults' -> 'GOTHIC ARCHITECTURE'
    """
    if not title:
        return "COLORING BOOK"

    s = title.upper().strip()

    # Sort by length descending so longer patterns are stripped before shorter
    # overlapping ones (e.g., "COLORING BOOK FOR ADULTS" before "COLORING BOOK").
    strips = sorted(
        [
            " COLORING BOOK FOR TEEN GIRLS",
            " COLORING BOOK FOR ADULTS",
            " COLORING BOOK FOR TEENS",
            " COLORING BOOK FOR KIDS",
            " COLORING BOOK",
            " FOR TEEN GIRLS",
            " FOR ADULTS",
            " FOR TEENS",
            " FOR KIDS",
            " AGES 4-7",
            " AGES 8-12",
            ", PRINTABLE PDF",
            ", INSTANT DOWNLOAD",
            ", SELF-CARE GIFT",
            ", DIGITAL DOWNLOAD",
            ", PRINTABLE",
        ],
        key=len,
        reverse=True,
    )

    for phrase in strips:
        s = s.replace(phrase, "")

    # If the title still has a comma, keep only the part before it — typical
    # Etsy pattern is "[CORE NICHE] Coloring Book for X, [SECONDARY KEYWORDS]",
    # and after stripping the middle, "[CORE NICHE], [SECONDARY]" remains.
    if "," in s:
        s = s.split(",")[0]

    # Collapse whitespace, strip stray punctuation
    s = " ".join(s.split()).strip(" ,.-")

    return s or "COLORING BOOK"


def fit_title(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_size: int = 200,
    min_size: int = 80,
    max_lines: int = 2,
):
    """Find the largest font size where text fits, wrapping to multiple lines if needed.

    Returns (lines_list, font, font_size).

    Strategy:
      1. Try single-line at decreasing font sizes (200 -> 80 in steps of 8)
      2. If single-line at min_size still doesn't fit, try 2-line wrap.
         Find the split point that maximizes the resulting font size.
      3. Fallback to min_size single line (may overflow; rare case).
    """
    text = text.upper()

    # Single-line attempt
    for size in range(max_size, min_size - 1, -8):
        font = get_font(size, bold=True)
        w, _ = measure_text(draw, text, font)
        if w <= max_width:
            return [text], font, size

    # Multi-line wrap
    if max_lines >= 2:
        words = text.split()
        if len(words) >= 2:
            best_size = 0
            best_lines = None
            for split_idx in range(1, len(words)):
                line1 = " ".join(words[:split_idx])
                line2 = " ".join(words[split_idx:])
                for size in range(max_size, min_size - 1, -8):
                    font_try = get_font(size, bold=True)
                    w1, _ = measure_text(draw, line1, font_try)
                    w2, _ = measure_text(draw, line2, font_try)
                    if w1 <= max_width and w2 <= max_width:
                        if size > best_size:
                            best_size = size
                            best_lines = [line1, line2]
                        break
            if best_lines:
                return best_lines, get_font(best_size, bold=True), best_size

    # Couldn't fit even with wrapping
    return [text], get_font(min_size, bold=True), min_size


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


@app.post("/build-pdf")
def build_pdf(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)
    if not req.images:
        raise HTTPException(400, "No images provided")

    pages = sorted(req.images, key=lambda i: extract_page_num(i.filename))
    page_w, page_h = letter
    margin = 18

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    if req.title:
        c.setFont("Helvetica-Bold", 44)
        c.drawCentredString(page_w / 2, page_h / 2 + 30, req.title)
        c.setFont("Helvetica", 16)
        c.drawCentredString(page_w / 2, page_h / 2 - 30, "A Coloring Book")
        c.showPage()

    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin

    for img_meta in pages:
        try:
            img = decode_grayscale(img_meta.data)
            img = crop_to_content(img)
            img = shrink_to_print_size(img)
            img = binarize(img)
        except Exception as e:
            raise HTTPException(400, f"Failed to decode {img_meta.filename}: {e}")

        iw, ih = img.size
        ratio = min(max_w / iw, max_h / ih)
        nw, nh = iw * ratio, ih * ratio
        x = (page_w - nw) / 2
        y = (page_h - nh) / 2

        img_io = BytesIO()
        img.save(img_io, format="PNG", optimize=True)
        img_io.seek(0)
        c.drawImage(ImageReader(img_io), x, y, width=nw, height=nh)
        c.showPage()

    c.save()
    name = safe_filename(req.title or "coloring-book")
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
    )


@app.post("/build-cover")
def build_cover(req: BuildRequest, x_service_key: str = Header(default="")):
    """Layout (Variant A: bold spec line, no bottom text):
        - Cream background spans the entire canvas (blends with Gemini cover art)
        - Title at top
        - Bold spec line below title: "{N} UNIQUE PRINTABLE PAGES"
        - Cover art fills the remaining ~75% down to the bottom edge
    Two text lines total. Page count is now the prominent second line
    (large + bold + slightly darker) so it reads at Etsy thumbnail scale.
    """
    auth_check(x_service_key)
    if not req.images:
        raise HTTPException(400, "No images provided")

    pages = sorted(req.images, key=lambda i: extract_page_num(i.filename))
    total_pages = req.page_count if req.page_count is not None else len(pages)

    W = H = 2000
    canvas_img = Image.new("RGB", (W, H), COVER_BG)
    draw = ImageDraw.Draw(canvas_img)

    # === Title at top (Variant A: smart-shortened, wraps to 2 lines if needed) ===
    # We shorten the long Etsy SEO title to a punchy core phrase for the cover,
    # then size it as large as it will go. The full SEO title still lives in
    # the listing — this only affects the cover image.
    cover_title = shorten_title_for_cover(req.title)
    title_lines, title_font, title_size = fit_title(
        draw,
        cover_title,
        max_width=W - 160,
        max_size=240,
        min_size=80,
        max_lines=2,
    )

    title_y = 100
    line_height = int(title_size * 1.10)
    last_line_bottom_y = title_y
    for line in title_lines:
        tw, th = measure_text(draw, line, title_font)
        draw.text(((W - tw) // 2, title_y), line, font=title_font, fill="black")
        last_line_bottom_y = title_y + th
        title_y += line_height

    # === Spec line under title (Variant A) ===
    # Replaces the generic "A COLORING BOOK" subtitle with a concrete value
    # statement that buyers see at thumbnail scale. Bold + larger + darker
    # than the old subtitle so it actually carries weight on the cover.
    spec_text = f"{total_pages} UNIQUE PRINTABLE PAGES"
    spec_size = 72
    while spec_size > 44:
        spec_font = get_font(spec_size, bold=True)
        sw, _ = measure_text(draw, spec_text, spec_font)
        if sw <= W - 200:
            break
        spec_size -= 4

    spec_font = get_font(spec_size, bold=True)
    sw, sh = measure_text(draw, spec_text, spec_font)
    sub_y = last_line_bottom_y + 35
    draw.text(((W - sw) // 2, sub_y), spec_text, font=spec_font, fill="#3A3A3A")

    # === Cover art fills the rest, all the way down to near the bottom ===
    # Removed the bottom page-count line, so the image can extend further.
    img_top = sub_y + sh + 60
    img_bottom = H - 70  # was H - 130 when the bottom text occupied space
    img_area_h = img_bottom - img_top
    img_area_w = W - 80  # 40px side padding

    page1 = decode_color(pages[0].data)
    page1 = crop_to_content(page1)
    pw, ph = page1.size
    # Fit-within: scale so the art fills as much of the area as possible
    # while preserving aspect ratio. Cream background hides any leftover gaps.
    ratio = min(img_area_w / pw, img_area_h / ph)
    nw, nh = int(pw * ratio), int(ph * ratio)
    page1 = page1.resize((nw, nh), Image.LANCZOS)
    img_x = (W - nw) // 2
    img_y = img_top + (img_area_h - nh) // 2
    canvas_img.paste(page1, (img_x, img_y))

    # (Variant A: no bottom page-count text; spec line above carries that info.)

    out = BytesIO()
    canvas_img.save(out, format="PNG", optimize=True)
    name = safe_filename(req.title or "cover")
    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{name}-cover.png"'},
    )


@app.post("/build-preview")
def build_preview(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)
    if len(req.images) < 1:
        raise HTTPException(400, "Need at least 1 image to build a preview")

    pages = sorted(req.images, key=lambda i: extract_page_num(i.filename))
    n = len(pages)
    total_pages = req.page_count if req.page_count is not None else n

    if n >= 4:
        indices = [int(i * (n - 1) / 3) for i in range(4)]
        selected = [pages[i] for i in indices]
    else:
        selected = pages[:]
        while len(selected) < 4:
            selected.append(pages[-1])

    W = H = 2000
    canvas_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas_img)

    top_text = f"{total_pages} UNIQUE PAGES INSIDE"
    top_font = get_font(56, bold=True)
    tw, _ = measure_text(draw, top_text, top_font)
    draw.text(((W - tw) // 2, 80), top_text, font=top_font, fill="black")

    cell = 700
    gap = 100
    grid_size = 2 * cell + gap
    grid_left = (W - grid_size) // 2
    grid_top = 220

    for idx, page in enumerate(selected):
        row, col = divmod(idx, 2)
        cell_x = grid_left + col * (cell + gap)
        cell_y = grid_top + row * (cell + gap)

        page_img = decode_grayscale(page.data)
        page_img = crop_to_content(page_img)
        pw, ph = page_img.size
        ratio = min(cell / pw, cell / ph) * 0.96
        nw, nh = int(pw * ratio), int(ph * ratio)
        page_img = page_img.resize((nw, nh), Image.LANCZOS).convert("RGB")

        ox = cell_x + (cell - nw) // 2
        oy = cell_y + (cell - nh) // 2

        draw.rectangle([ox + 8, oy + 8, ox + nw + 8, oy + nh + 8], fill="#dddddd")
        canvas_img.paste(page_img, (ox, oy))
        draw.rectangle([ox, oy, ox + nw, oy + nh], outline="#999999", width=2)

    bottom_text = "PRINT AT HOME - PERSONAL USE"
    bot_font = get_font(36, bold=False)
    bw, _ = measure_text(draw, bottom_text, bot_font)
    draw.text(((W - bw) // 2, H - 110), bottom_text, font=bot_font, fill="#666666")

    out = BytesIO()
    canvas_img.save(out, format="PNG", optimize=True)
    name = safe_filename(req.title or "preview")
    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{name}-preview.png"'},
    )


@app.post("/build-readme")
def build_readme(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)

    title = req.title or "Coloring Book"
    page_count = req.page_count if req.page_count is not None else len(req.images)

    page_w, page_h = letter
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    margin = 72
    y = page_h - margin - 30

    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(page_w / 2, y, "Thank You!")
    y -= 50
    c.setFont("Helvetica-Oblique", 14)
    c.drawCentredString(page_w / 2, y, f"You've purchased {title}")
    y -= 60

    def section(heading: str, body: str) -> None:
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, heading)
        y -= 22
        c.setFont("Helvetica", 11)
        for line in body.split("\n"):
            c.drawString(margin, y, line)
            y -= 16
        y -= 12

    section(
        "What's Inside",
        f"This pack contains {page_count} unique coloring pages designed for your enjoyment.\n"
        "All pages are formatted as US Letter (8.5 x 11 inches) and ready to print at home.",
    )

    section(
        "Printing Tips",
        "- Use 80 lb (or heavier) paper for best results with markers and gel pens.\n"
        "- Set your printer to 'Fit to Page' or '100% scale' - do not enlarge.\n"
        "- Print one page at a time if your printer struggles with the full PDF.\n"
        "- Set print quality to High for the crispest line work.",
    )

    section(
        "Terms of Use",
        "- This file is for PERSONAL use only.\n"
        "- You may print as many copies as you'd like for yourself or your family.\n"
        "- You may NOT resell, redistribute, or share the digital file.\n"
        "- You may NOT use these pages for commercial coloring books or paid products.\n"
        "- You may NOT claim authorship of these designs.",
    )

    section(
        "About These Designs",
        "These coloring pages were created with the assistance of AI image generation,\n"
        "then curated and assembled by a human. Each page was reviewed for quality\n"
        "before being included in this pack.",
    )

    section(
        "Need Help?",
        "If you have any issues with the file, please reach out via Etsy messages.\n"
        "I'm happy to help with downloads, printing, or any other questions.",
    )

    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(page_w / 2, margin / 2, "Happy coloring!")
    c.save()

    name = safe_filename(req.title or "readme")
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}-readme.pdf"'},
    )


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "coloring-book-pdf",
        "endpoints": ["/build-pdf", "/build-cover", "/build-preview", "/build-readme"],
        "config": {
            "print_dpi": PRINT_DPI,
            "binarize_threshold": BINARIZE_THRESHOLD,
            "cover_bg": COVER_BG,
        },
    }
