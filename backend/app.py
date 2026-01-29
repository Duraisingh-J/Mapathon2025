from fastapi import FastAPI, UploadFile, File
import shutil, os
from pipeline import analyze_lake

app = FastAPI()

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(satellite: UploadFile = File(...), dem: UploadFile = File(...)):

    sat_path = f"{UPLOAD_DIR}/{satellite.filename}"
    dem_path = f"{UPLOAD_DIR}/{dem.filename}"

    with open(sat_path, "wb") as f:
        shutil.copyfileobj(satellite.file, f)

    with open(dem_path, "wb") as f:
        shutil.copyfileobj(dem.file, f)

    result = analyze_lake(sat_path, dem_path)

    return result
