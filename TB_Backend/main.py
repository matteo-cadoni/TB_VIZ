import os
import shutil
from pathlib import Path
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


from utils.run_pipeline import run_pipeline

app = FastAPI()

allowed_origin = os.getenv("TB_VIZARR_FRONTEND", "http://127.0.0.1:8080")  # Default if env var not set


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],  # Allow the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class MoveImageRequest(BaseModel):
    filename: str
    target_dir: str


@app.get("/")
async def root():
    return {"message": "Welcome to the tuberculosis visualizer"}


@app.post("/upload_czi/")
async def upload_czi(file: UploadFile = File(...)):
    upload_dir = Path("uploads")

    # Create the uploads directory if it doesn't exist
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save the file to the uploads directory
    file_path = upload_dir / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(file.file.read())
    return {"message": f"File {file.filename} uploaded successfully",
            "file_path": file_path,
            "file_name": file.filename}


@app.get("/get_bacilli_count/")
async def get_bacilli_count(file_path: str):

    # Load the file from the local filesystem
    file_path = Path(file_path)

    og_path, map_path = run_pipeline(file_path)

    return {"message": f"File {file_path} processed successfully",
            "map_path": map_path, "og_path": og_path}


@app.post("/move-image")
async def move_image(request: MoveImageRequest):
    """
    Moves an image from its current location to the specified dist directory.
    """
    try:
        print(os.getcwd())  # Debugging: Print current working directory

        # Define source and destination paths
        source_path = request.filename  # Assume original location
        destination_path = os.path.join(request.target_dir, os.path.basename(request.filename))

        # Check if the file exists
        if not os.path.exists(source_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Move the file to the target directory
        shutil.copytree(source_path, destination_path)

        return {"message": "File moved successfully", "imageUrl": f"/{destination_path}/{source_path}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
