"""
Coloring Book PDF Service
Endpoints:
  GET  /                — health check
  POST /build-pdf       — full coloring book interior PDF
  POST /build-cover     — 2000x2000 PNG Etsy listing thumbnail
  POST /build-preview   — 2000x2000 PNG showing 4 sample pages
  POST /build-readme    — customer-facing thank you + terms PDF

All POST endpoints require X-Service-Key header (matching SERVICE_KEY env var).
All POST endpoints accept the same JSON body:
  { "title": "Medieval Snails", "images": [{"filename": "page-1.png", "data": "<b64>"}, ...] }
"""

import base64
import os
import re
from contextlib import asynccontextmanager
from io import BytesIO
from urllib.request import urlopen

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

SERVICE_KEY = os.getenv("SERVICE_KEY", "")
FONT_DIR = "/tmp/fonts"
INTER_BOLD = f"{FONT_DIR}/Inter-Bold.ttf"
INTER_REGULAR = f"{FONT_DIR}/Inter-Regular.ttf"

# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------

def ensure_fonts() -> bool:
    """Download Inter on first call; cache in /tmp. Returns False on failure."""
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
    """Return Inter at the requested size, falling back to system fonts then default."""
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
    ensure_fonts()  # pre-warm fonts on startup
    yield


app = FastAPI(title="Coloring Book PDF Service", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Models & helpers
# ---------------------------------------------------------------------------

class PageImage(BaseModel):
    filename: str
    data: str  # base64-encoded image bytes


class BuildRequest(BaseModel):
    images: list[PageImage]
    title: str = ""


def auth_check(key: str) -> None:
    if SERVICE_KEY and key != SERVICE_KEY:
        raise HTTPException(401, "Invalid or missing X-Service-Key header")


def extract_page_num(filename: str) -> int:
    m = re.search(r"page-(\d+)", filename)
    return int(m.group(1)) if m else 9999


def decode_grayscale(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("L")


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-") or "coloring-book"


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ---------------------------------------------------------------------------
# /build-pdf — full coloring book interior
# ---------------------------------------------------------------------------

@app.post("/build-pdf")
def build_pdf(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)
    if not req.images:
        raise HTTPException(400, "No images provided")

    pages = sorted(req.images, key=lambda i: extract_page_num(i.filename))
    page_w, page_h = letter
    margin = 36

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


# ---------------------------------------------------------------------------
# /build-cover — 2000x2000 PNG Etsy listing thumbnail
# ---------------------------------------------------------------------------

@app.post("/build-cover")
def build_cover(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)
    if not req.images:
        raise HTTPException(400, "No images provided")

    pages = sorted(req.images, key=lambda i: extract_page_num(i.filename))
    W = H = 2000
    canvas_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas_img)

    pad = 80
    img_area_h = int(H * 0.62)
    target_w = W - 2 * pad
    target_h = img_area_h - pad

    page1 = decode_grayscale(pages[0].data)
    pw, ph = page1.size
    ratio = min(target_w / pw, target_h / ph)
    nw, nh = int(pw * ratio), int(ph * ratio)
    page1 = page1.resize((nw, nh), Image.LANCZOS).convert("RGB")
    canvas_img.paste(page1, ((W - nw) // 2, pad))

    line_y = img_area_h + 40
    draw.line([(W // 4, line_y), (W - W // 4, line_y)], fill="black", width=4)

    title_text = (req.title or "Coloring Book").upper()
    title_size = 130
    while title_size > 60:
        font = get_font(title_size, bold=True)
        tw, _ = measure_text(draw, title_text, font)
        if tw <= W - 200:
            break
        title_size -= 8

    font = get_font(title_size, bold=True)
    tw, th = measure_text(draw, title_text, font)
    text_y = line_y + 60
    draw.text(((W - tw) // 2, text_y), title_text, font=font, fill="black")

    subtitle = "A COLORING BOOK"
    sub_font = get_font(48, bold=False)
    sw, _ = measure_text(draw, subtitle, sub_font)
    sub_y = text_y + th + 60
    draw.text(((W - sw) // 2, sub_y), subtitle, font=sub_font, fill="#666666")

    pc_text = f"{len(req.images)} UNIQUE PAGES"
    pc_font = get_font(36, bold=True)
    pcw, _ = measure_text(draw, pc_text, pc_font)
    draw.text(((W - pcw) // 2, sub_y + 90), pc_text, font=pc_font, fill="#999999")

    out = BytesIO()
    canvas_img.save(out, format="PNG", optimize=True)
    name = safe_filename(req.title or "cover")
    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{name}-cover.png"'},
    )


# ---------------------------------------------------------------------------
# /build-preview — 2000x2000 PNG showing 4 sample pages
# ---------------------------------------------------------------------------

@app.post("/build-preview")
def build_preview(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)
    if len(req.images) < 4:
        raise HTTPException(400, "Need at least 4 images to build a preview")

    pages = sorted(req.images, key=lambda i: extract_page_num(i.filename))
    n = len(pages)
    indices = [int(i * (n - 1) / 3) for i in range(4)]
    selected = [pages[i] for i in indices]

    W = H = 2000
    canvas_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas_img)

    top_text = f"{n} UNIQUE PAGES INSIDE"
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
        pw, ph = page_img.size
        ratio = min(cell / pw, cell / ph) * 0.92
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


# ---------------------------------------------------------------------------
# /build-readme — customer-facing thank you + terms PDF
# ---------------------------------------------------------------------------

@app.post("/build-readme")
def build_readme(req: BuildRequest, x_service_key: str = Header(default="")):
    auth_check(x_service_key)

    title = req.title or "Coloring Book"
    page_count = len(req.images)

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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "coloring-book-pdf",
        "endpoints": ["/build-pdf", "/build-cover", "/build-preview", "/build-readme"],
    }
