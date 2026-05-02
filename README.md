# Coloring Book PDF Service

Tiny FastAPI service that turns a list of base64-encoded coloring page images
into a single print-ready PDF.

## Endpoints

- `GET /` — health check
- `POST /build-pdf` — accepts JSON, returns a PDF binary

## Environment variables

- `PORT` — set automatically by Railway
- `SERVICE_KEY` (optional) — if set, requests must include matching `X-Service-Key` header

## Local test

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# then POST to http://localhost:8000/build-pdf
```
