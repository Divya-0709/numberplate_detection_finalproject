"""
FastAPI wrapper for the ALPR pipeline.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload

Docs at: http://localhost:8000/docs
"""

import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO
import easyocr

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ALPR API",
    description="Automatic License Plate Recognition using YOLOv8 + EasyOCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading (once at startup) ──────────────────────────────────────────
_yolo_model = None
_ocr_reader = None

@app.on_event("startup")
def load_models():
    global _yolo_model, _ocr_reader
    _yolo_model = YOLO("NumberPlate_Dataset/best.pt")
    _ocr_reader = easyocr.Reader(["en"], gpu=False)

# ── Response schemas ──────────────────────────────────────────────────────────
class PlateDetection(BaseModel):
    plate_text: str
    yolo_confidence: float
    ocr_confidence: float
    bounding_box: List[int]   # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    status: str
    num_plates: int
    detections: List[PlateDetection]
    processing_time_ms: float

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "running", "model": "YOLOv8 + EasyOCR", "version": "1.0.0"}

@app.get("/health")
def health():
    """Detailed health — confirms models are loaded."""
    return {
        "status": "healthy",
        "yolo_loaded": _yolo_model is not None,
        "ocr_loaded": _ocr_reader is not None,
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_plates(file: UploadFile = File(...)):
    """
    Detect license plates in an uploaded image.

    - **file**: JPG or PNG image file
    - Returns: list of detected plates with text and confidence scores
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )

    start_time = time.time()

    # Read image bytes → numpy array
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # YOLO detection
    results = _yolo_model(image, conf=0.25)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            yolo_conf = float(box.conf[0])

            # Crop plate region
            plate_crop = image[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # Preprocess: grayscale → 2x upscale → Otsu threshold
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            upscaled = cv2.resize(gray, (w * 2, h * 2))
            _, thresh = cv2.threshold(
                upscaled, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # OCR
            ocr_results = _ocr_reader.readtext(thresh)
            if not ocr_results:
                plate_text = ""
                ocr_conf = 0.0
            else:
                plate_text = " ".join([r[1] for r in ocr_results])
                ocr_conf = float(ocr_results[0][2])

            detections.append(PlateDetection(
                plate_text=plate_text,
                yolo_confidence=round(yolo_conf, 3),
                ocr_confidence=round(ocr_conf, 3),
                bounding_box=[x1, y1, x2, y2],
            ))

    processing_time = round((time.time() - start_time) * 1000, 1)

    return DetectionResponse(
        status="success",
        num_plates=len(detections),
        detections=detections,
        processing_time_ms=processing_time,
    )
