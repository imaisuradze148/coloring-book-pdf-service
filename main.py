"""
Coloring Book PDF Service
Accepts a list of base64-encoded images and assembles them into a print-ready PDF.

POST /build-pdf
  Headers:
    X-Service-Key: <secret>      (required if SERVICE_KEY env var is set)
  Body (JSON):
    {
      "title": "Medieval Snails",
      "images": [
        {"filename": "page-1.png", "data": "<base64>"},
        {"filename": "page-2.png", "data": "<base64>"},
        ...
      ]
    }
  Response: application/pdf binary
"""

import base64
import os
import re
from io import BytesIO

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

app = FastAPI(title="Coloring Book PDF Service")

# Optional shared-secret auth. Set SERVICE_KEY in Railway env vars to enable.
SERVICE_KEY = os.getenv("SERVICE_KEY", "")


class PageImage(BaseModel):
    filename: str
    data: str  # base64-encoded image bytes


class BuildPdfRequest(BaseModel):
    images: list[PageImage]
    title: str = ""


@app.get("/")
def health():
    return {"status": "ok", "service": "coloring-book-pdf"}


def extract_page_num(filename: str) -> int:
    """Extract page number from filenames like 'medieval-snails-page-12.png'."""
    match = re.search(r"page-(\d+)", filename)
    return int(match.group(1)) if match else 9999


@app.post("/build-pdf")
def build_pdf(req: BuildPdfRequest, x_service_key: str = Header(default="")):
    # Auth
    if SERVICE_KEY and x_service_key != SERVICE_KEY:
        raise HTTPException(401, "Invalid or missing X-Service-Key header")

    if not req.images:
        raise HTTPException(400, "No images provided")

    # Sort pages numerically (page-2 must come before page-10)
    images_sorted = sorted(req.images, key=lambda i: extract_page_num(i.filename))

    # US Letter at default 72pt scale (reportlab handles DPI internally)
    page_w, page_h = letter  # 612 x 792 points
    margin = 36  # 0.5 inch margins

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    # Cover page (only if a title was supplied)
    if req.title:
        c.setFont("Helvetica-Bold", 44)
        c.drawCentredString(page_w / 2, page_h / 2 + 30, req.title)
        c.setFont("Helvetica", 16)
        c.drawCentredString(page_w / 2, page_h / 2 - 30, "A Coloring Book")
        c.showPage()

    # Layout area for images
    max_w = page_w - 2 * margin
    max_h = page_h - 2 * margin

    for img_meta in images_sorted:
        try:
            raw = base64.b64decode(img_meta.data)
            # Convert to grayscale — line art doesn't need color, and L mode
            # produces a much smaller PDF without visible quality loss.
            img = Image.open(BytesIO(raw)).convert("L")
        except Exception as e:
            raise HTTPException(400, f"Failed to decode {img_meta.filename}: {e}")

        # Scale to fit page within margins, preserving aspect ratio
        img_w, img_h = img.size
        ratio = min(max_w / img_w, max_h / img_h)
        new_w = img_w * ratio
        new_h = img_h * ratio
        x = (page_w - new_w) / 2
        y = (page_h - new_h) / 2

        # Re-encode as optimized PNG before embedding (smaller PDF)
        img_io = BytesIO()
        img.save(img_io, format="PNG", optimize=True)
        img_io.seek(0)
        c.drawImage(ImageReader(img_io), x, y, width=new_w, height=new_h)
        c.showPage()

    c.save()
    pdf_bytes = buf.getvalue()

    safe_name = re.sub(r"[^a-z0-9-]+", "-", (req.title or "coloring-book").lower()).strip("-")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'},
    )
