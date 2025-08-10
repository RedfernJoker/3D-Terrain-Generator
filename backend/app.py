from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import uuid
import terrain_generator
import tempfile

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/double")
def double(x: int):
    # example endpoint that doubles a query‐param
    return {"result": x * 2}

@app.post("/generate-terrain")
async def generate_terrain(file: UploadFile = File(...)):
    try:
        # setup the models directory
        models_dir = Path("../frontend/public/models")
        models_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # generate the terrain
        terrain_generator.process_image_to_terrain(temp_file_path, str(models_dir / 't1.glb'))
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        # return the model file name
        return JSONResponse(content={"modelFileName": 't1.glb'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to generate terrain: {str(e)}")
    

