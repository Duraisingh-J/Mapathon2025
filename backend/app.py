import shutil
import os
import uuid
from datetime import date
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .pipeline import process_water_from_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Lake Analysis API is running"}

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze(satellite: UploadFile = File(...), dem: UploadFile = File(None)):
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    today_str = str(date.today())
    
    # Save Uploaded File
    sat_ext = os.path.splitext(satellite.filename)[1]
    sat_filename = f"{run_id}_sat{sat_ext}"
    sat_path = os.path.join(UPLOAD_DIR, sat_filename)
    
    with open(sat_path, "wb") as f:
        shutil.copyfileobj(satellite.file, f)
        
    print(f"[DEBUG] Satellite file saved to {sat_path}")

    # Save DEM if provided (optional)
    if dem:
        dem_ext = os.path.splitext(dem.filename)[1]
        dem_filename = f"{run_id}_dem{dem_ext}"
        dem_path = os.path.join(UPLOAD_DIR, dem_filename)
        with open(dem_path, "wb") as f:
            shutil.copyfileobj(dem.file, f)
        print(f"DEM saved to {dem_path}")

    try:
        print(f"[DEBUG] Calling pipeline with sat={sat_path}, id={run_id}")
        # Call Pipeline
        area_ha = process_water_from_image(
            image_path=sat_path, 
            lake_id=run_id, 
            date_str=today_str, 
            output_dir=OUTPUT_DIR
        )
        print(f"[DEBUG] Pipeline returned area: {area_ha}")
        
        return {
            "area_ha": round(area_ha, 2),
            "volume_m3": 0,    # Placeholder until volume logic is added
            "volume_tmc": 0,   # Placeholder until volume logic is added
            "message": "Analysis successful"
        }
    except Exception as e:
        print(f"Pipeline Error: {e}")
        return {"error": str(e)}
