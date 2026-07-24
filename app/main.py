"""FastAPI backend for CT Plant ID: photo in, top-5 CT species out.

A phone-first tool. The user uploads a plant photo to /api/identify; we run
the trained EfficientNetV2 checkpoint (2,360 Connecticut species), take the
top-5 candidates, and annotate each with common name + native/invasive/weed
status so the answer is actionable, not just a Latin binomial.

The model is heavy (~89MB, a few seconds to load) so it is loaded ONCE at
import time and reused across requests — never per-request.
"""
import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

# Hard cap on decoded pixels: a "decompression bomb" is a tiny file with enormous
# dimensions. We reject anything past MAX_PIXELS by reading the image HEADER before
# decoding (see identify()). MAX_IMAGE_PIXELS is a Pillow-level backstop, but on
# its own it only WARNS up to 2x the limit, so the explicit header check is the
# real guard.
MAX_PIXELS = 40_000_000  # 40MP, far above any real phone photo
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# The inference core lives in src/ctplantid; make it importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ctplantid.predict import PlantModel  # noqa: E402

# The attributes module (common name + status) is built concurrently by another
# agent. Guard the import so the API runs even before it exists — every
# candidate then falls back to an honest "unknown".
try:
    from app.attributes import annotate
except Exception:  # module missing or not yet importable
    def annotate(species):
        return {"common_name": None, "status": "unknown", "is_weed": False}

STATIC_DIR = ROOT / "app" / "static"
CKPT_PATH = ROOT / "runs" / "stage2" / "model.pt"

# The model's probabilities are now TEMPERATURE-CALIBRATED in predict.py
# (scripts/fit_temperature.py fit T=0.6, dropping ECE 0.31 -> 0.05). So the
# top-1 prob reaching here already means "P(this ID is correct)" — mean
# confidence 78% vs measured accuracy 77%. That lets us threshold and label
# directly on the honest number instead of the old underconfident raw softmax.
#
# Below this calibrated probability we flag "not sure": a leading guess the
# model itself rates under ~40% likely to be right deserves the caution banner.
NOT_SURE_THRESHOLD = 0.40


# Qualitative label for the calibrated top-1 probability. Because the number now
# IS the estimated chance of being correct, the bands read literally:
#   >=0.80 very likely right, 0.60-0.80 probably, 0.40-0.60 leading-but-check.
def confidence_label(prob: float) -> str:
    if prob >= 0.80:
        return "Strong match"
    if prob >= 0.60:
        return "Likely match"
    if prob >= NOT_SURE_THRESHOLD:
        return "Possible match"
    return "Uncertain"

# Reject oversized uploads before decoding to bound memory. Phone photos are a
# few MB; 15MB is generous headroom without inviting decompression-bomb abuse.
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="CT Plant ID")

# Load the model once at startup. This is intentionally at import time so the
# first request is fast and health checks reflect a truly-ready server.
# Force CPU: Apple's MPS backend is faster but can crash (segfault) the whole
# process on some inputs, which takes the server down mid-session. CPU inference
# of one 384px image is well under 2s here and rock-solid — the right trade for
# serving. (Set CTPLANT_DEVICE to override, e.g. "cuda" on a GPU host.)
import os  # noqa: E402
model = PlantModel(str(CKPT_PATH), device=os.environ.get("CTPLANT_DEVICE", "cpu"))

# The frontend is owned by another agent and may not have landed yet. Ensure
# the directory exists so the StaticFiles mount and GET / don't crash the
# server before index.html appears.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Candidate(BaseModel):
    species: str
    common_name: str | None
    prob: float
    status: str
    is_weed: bool


class IdentifyResponse(BaseModel):
    top_species: str
    not_sure: bool
    confidence_label: str  # qualitative, honest label for the top match
    candidates: list[Candidate]


@app.get("/api/health")
def health():
    """Confirm the model loaded and report how many species it can name."""
    return {"ok": True, "species": len(model.classes)}


@app.get("/")
def index():
    """Serve the single-page frontend."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend not built yet")
    return FileResponse(str(index_path))


@app.post("/api/identify", response_model=IdentifyResponse)
async def identify(image: UploadFile = File(...)):
    """Identify the plant in an uploaded photo.

    Returns the top-5 candidate CT species, each annotated with common name and
    native/invasive/weed status, plus a not_sure flag when the leading guess is
    below the confidence threshold. Corrupt or non-image uploads yield a clean
    400 rather than a 500.
    """
    # Read at most MAX_UPLOAD_BYTES+1 so an oversized upload can't balloon memory
    # before we reject it below.
    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)",
        )

    # Decode defensively: anything that isn't a valid image is a 400. verify()
    # catches truncated/corrupt files; we then reopen because verify() leaves
    # the image unusable for further reading.
    try:
        # Image.open reads only the header, so .size is known without decoding
        # pixels — reject an oversized-dimension bomb BEFORE the expensive decode.
        probe = Image.open(io.BytesIO(raw))
        if probe.width * probe.height > MAX_PIXELS:
            raise HTTPException(status_code=400, detail="image dimensions too large")
        Image.open(io.BytesIO(raw)).verify()   # catches truncated/corrupt files
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="not a valid image file")

    preds = model.predict(pil, k=5)

    candidates = []
    for p in preds:
        attrs = annotate(p["species"])
        candidates.append(
            Candidate(
                species=p["species"],
                common_name=attrs.get("common_name"),
                prob=p["prob"],
                status=attrs.get("status", "unknown"),
                is_weed=attrs.get("is_weed", False),
            )
        )

    if not candidates:  # defensive: predict() always returns 5, so this is never
        raise HTTPException(status_code=422, detail="could not identify the image")
    top = candidates[0]
    return IdentifyResponse(
        top_species=top.species,
        not_sure=top.prob < NOT_SURE_THRESHOLD,
        confidence_label=confidence_label(top.prob),
        candidates=candidates,
    )
