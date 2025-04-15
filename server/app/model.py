import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path
import timm

# Define the EnsembleModel class
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        # Simple averaging ensemble
        out = (out1 + out2 + out3) / 3
        return out

# Define the wrapper for unified ensemble model
class EnsembleModelWrapper(nn.Module):
    """
    A wrapper for the ensemble model that includes all three sub-models internally.
    This allows us to save and load a single model file.
    """
    def __init__(self):
        super(EnsembleModelWrapper, self).__init__()
        
        # Initialize the three models
        self.efficientnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        self.swin = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2)
        self.resnet = timm.create_model("resnet50d", pretrained=False, num_classes=2)
        
        # Create the ensemble model
        self.ensemble = EnsembleModel(self.efficientnet, self.swin, self.resnet)
    
    def forward(self, x):
        return self.ensemble(x)

# Get the directory where the models are stored
BASE_MODEL_DIR = Path("../src/model").resolve()
if not BASE_MODEL_DIR.exists():
    BASE_MODEL_DIR = Path("../../src/model").resolve()
if not BASE_MODEL_DIR.exists():
    BASE_MODEL_DIR = Path(os.path.join(os.path.dirname(__file__), "../../src/model")).resolve()

print(f"Looking for models in: {BASE_MODEL_DIR}")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Look for the unified ensemble model
UNIFIED_MODEL_PATH = BASE_MODEL_DIR / "unified_ensemble_model.pth"
ensemble_model = None

# Check if the unified model exists
if UNIFIED_MODEL_PATH.exists():
    try:
        # Load the unified model
        unified_model = EnsembleModelWrapper()
        unified_model.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=device))
        unified_model.to(device)
        unified_model.eval()
        ensemble_model = unified_model
        print(f"Unified ensemble model loaded successfully from {UNIFIED_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading unified model: {e}")
        ensemble_model = None
else:
    print(f"Unified ensemble model not found at {UNIFIED_MODEL_PATH}")
    print("Falling back to loading individual models")
    
    # Define model file paths for fallback
    MODELS = {
        "efficientnet": {
            "path": None,
            "files": ["efficientnet_pneumonia.pth"],
            "architecture": "efficientnet_b0",
            "model_obj": None
        },
        "swin": {
            "path": None,
            "files": ["swin_best-2.pth"],
            "architecture": "swinv2_tiny_window8_256",
            "model_obj": None
        },
        "resnet": {
            "path": None,
            "files": ["resnet_best.pth"],
            "architecture": "resnet50d",
            "model_obj": None
        }
    }

    # Find model files
    for model_name, model_info in MODELS.items():
        for filename in model_info["files"]:
            model_path = BASE_MODEL_DIR / filename
            if model_path.exists():
                MODELS[model_name]["path"] = model_path
                print(f"Found {model_name} model at: {model_path}")
                break

    # Check if all models were found
    all_models_found = all(model_info["path"] is not None for model_info in MODELS.values())

    # Load the models
    if all_models_found:
        try:
            # Load EfficientNet model
            model1 = timm.create_model(MODELS["efficientnet"]["architecture"], pretrained=False, num_classes=2)
            model1.load_state_dict(torch.load(MODELS["efficientnet"]["path"], map_location=device))
            model1.to(device)
            model1.eval()
            MODELS["efficientnet"]["model_obj"] = model1
            print("EfficientNet model loaded successfully")

            # Load SwinV2 model
            model2 = timm.create_model(MODELS["swin"]["architecture"], pretrained=False, num_classes=2)
            model2.load_state_dict(torch.load(MODELS["swin"]["path"], map_location=device), strict=False)
            model2.to(device)
            model2.eval()
            MODELS["swin"]["model_obj"] = model2
            print("Swin model loaded successfully")

            # Load ResNet model
            model3 = timm.create_model(MODELS["resnet"]["architecture"], pretrained=False, num_classes=2)
            model3.load_state_dict(torch.load(MODELS["resnet"]["path"], map_location=device), strict=False)
            model3.to(device)
            model3.eval()
            MODELS["resnet"]["model_obj"] = model3
            print("ResNet model loaded successfully")

            # Create ensemble model
            ensemble_model = EnsembleModel(model1, model2, model3)
            ensemble_model.to(device)
            ensemble_model.eval()
            print("Ensemble model created successfully")
            
        except Exception as e:
            print(f"Error creating ensemble model: {e}")
            ensemble_model = None
            
            # Try to load at least one model for fallback
            for model_name, model_info in MODELS.items():
                if model_info["path"] is not None:
                    try:
                        model = timm.create_model(model_info["architecture"], pretrained=False, num_classes=2)
                        model.load_state_dict(torch.load(model_info["path"], map_location=device), 
                                             strict=(model_name != "swin" and model_name != "resnet"))
                        model.to(device)
                        model.eval()
                        ensemble_model = model  # Use a single model as fallback
                        print(f"Using {model_name} as fallback model")
                        break
                    except Exception as e:
                        print(f"Failed to load {model_name} model: {e}")
    else:
        print("Warning: Not all models were found. Attempting to use what's available.")
        # Try to load any available model
        for model_name, model_info in MODELS.items():
            if model_info["path"] is not None:
                try:
                    model = timm.create_model(model_info["architecture"], pretrained=False, num_classes=2)
                    model.load_state_dict(torch.load(model_info["path"], map_location=device),
                                         strict=(model_name != "swin" and model_name != "resnet"))
                    model.to(device)
                    model.eval()
                    ensemble_model = model  # Use a single model as fallback
                    print(f"Using {model_name} as fallback model")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name} model: {e}")

# Define image transformations for the model
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Classes for prediction
CLASSES = ["Normal", "Pneumonia"]

async def predict_xray(file):
    """
    Process the uploaded X-ray image and return prediction using ensemble model
    """
    if ensemble_model is None:
        return {"error": "Model not loaded properly"}
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = ensemble_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            
        # Get the prediction class and confidence
        predicted_class = CLASSES[predicted.item()]
        confidence_value = probabilities[predicted.item()].item() * 100
        
        # Return the prediction results
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence_value:.2f}%",
            "probabilities": {
                CLASSES[i]: f"{prob.item() * 100:.2f}%" 
                for i, prob in enumerate(probabilities)
            }
        }
    except Exception as e:
        return {"error": str(e)}