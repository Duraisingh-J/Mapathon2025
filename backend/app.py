import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import datetime

# IMPORTANT: correct import
from backend import pipeline   # ← FIXED

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

from fastapi.staticfiles import StaticFiles

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount Output directory to serve images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Lake Analysis API is running"}

# -----------------------------
# Main API
# -----------------------------
@app.post("/analyze")
async def analyze(
    satellite: List[UploadFile] = File(...),
    dem: UploadFile = File(None),
    base_level: float = Form(None),
    dates: str = Form(None)
):
    run_id = str(uuid.uuid4())[:8]

    # -----------------------------
    # Save DEM (optional)
    # -----------------------------
    dem_path = None
    if dem:
        dem_ext = os.path.splitext(dem.filename)[1]
        dem_path = os.path.join(UPLOAD_DIR, f"{run_id}_dem{dem_ext}")
        with open(dem_path, "wb") as f:
            shutil.copyfileobj(dem.file, f)

    # -----------------------------
    # Save satellite images
    # -----------------------------
    sat_paths = []
    for idx, file in enumerate(satellite):
        ext = os.path.splitext(file.filename)[1]
        sat_path = os.path.join(UPLOAD_DIR, f"{run_id}_sat_{idx}{ext}")
        with open(sat_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        sat_paths.append(sat_path)

    # -----------------------------
    # Run pipeline
    # -----------------------------
    try:
        results = pipeline.analyze_lake(
            image_paths=sat_paths,
            dem_path=dem_path,
            lake_id=run_id,
            date_string=dates,
            output_dir=OUTPUT_DIR,
            base_level=base_level
        )
        return results

    except Exception as e:
        return {
            "error": str(e),
            "message": "Analysis failed"
        }


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
