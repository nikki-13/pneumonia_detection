from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.model import predict_xray
import os
from pathlib import Path
import glob
from PIL import Image
import io
import time
from typing import List, Dict, Union

app = FastAPI(title="X-Ray Insight API", description="API for X-Ray pneumonia prediction")

# Configure CORS to allow requests from the frontend
# Default to localhost:5173 for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],  # Allow both default Vite ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "X-Ray Insight API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an X-ray image and get pneumonia prediction
    """
    try:
        # Process the uploaded image and get prediction
        result = await predict_xray(file)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/test")
async def run_batch_test(category: str = None, limit: int = 20):
    """
    Run tests on the X-ray model using test images
    
    Parameters:
    - category: Filter tests by "NORMAL" or "PNEUMONIA" (optional)
    - limit: Maximum number of test images to process (default 20)
    
    Returns results for model performance evaluation
    """
    test_images_dir = Path("../test_images/test").resolve()
    if not test_images_dir.exists():
        test_images_dir = Path("../../server/test_images/test").resolve()
    
    if not test_images_dir.exists():
        return {"error": f"Test images directory not found: {test_images_dir}"}
    
    # Get paths for test images
    image_paths = []
    
    if category and category.upper() in ["NORMAL", "PNEUMONIA"]:
        image_dir = test_images_dir / category.upper()
        image_paths = list(image_dir.glob("*.jpeg"))[:limit]
    else:
        # Get both normal and pneumonia images
        normal_images = list((test_images_dir / "NORMAL").glob("*.jpeg"))[:limit//2]
        pneumonia_images = list((test_images_dir / "PNEUMONIA").glob("*.jpeg"))[:limit//2]
        image_paths = normal_images + pneumonia_images
    
    if not image_paths:
        return {"error": "No test images found"}
    
    # Process each image
    results = []
    metrics = {"correct": 0, "total": 0, "normal_correct": 0, "normal_total": 0, "pneumonia_correct": 0, "pneumonia_total": 0}
    processing_time = 0
    
    for image_path in image_paths:
        try:
            start_time = time.time()
            
            # Open and prepare the image
            with open(image_path, "rb") as img_file:
                contents = img_file.read()
            
            # Create a mock UploadFile object
            mock_file = type('MockFile', (), {
                'filename': image_path.name,
                'read': lambda: contents
            })()
            
            # Get prediction
            result = await predict_xray(mock_file)
            end_time = time.time()
            
            # Calculate metrics
            processing_time += (end_time - start_time)
            
            # Add ground truth based on directory name
            true_label = "Normal" if "NORMAL" in str(image_path) else "Pneumonia"
            result["true_label"] = true_label
            result["correct"] = result["prediction"] == true_label
            
            # Update metrics
            metrics["total"] += 1
            if result["correct"]:
                metrics["correct"] += 1
            
            if true_label == "Normal":
                metrics["normal_total"] += 1
                if result["correct"]:
                    metrics["normal_correct"] += 1
            else:
                metrics["pneumonia_total"] += 1
                if result["correct"]:
                    metrics["pneumonia_correct"] += 1
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": image_path.name,
                "error": str(e)
            })
    
    # Calculate accuracy metrics
    if metrics["total"] > 0:
        metrics["accuracy"] = metrics["correct"] / metrics["total"]
        metrics["normal_accuracy"] = metrics["normal_correct"] / metrics["normal_total"] if metrics["normal_total"] > 0 else 0
        metrics["pneumonia_accuracy"] = metrics["pneumonia_correct"] / metrics["pneumonia_total"] if metrics["pneumonia_total"] > 0 else 0
        metrics["avg_processing_time"] = processing_time / metrics["total"]
    
    return {
        "metrics": metrics,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
