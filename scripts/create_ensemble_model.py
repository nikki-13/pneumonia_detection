#!/usr/bin/env python3
"""
Script to create a unified ensemble model from individual models.
This will load the three separate models, combine them into an ensemble,
and save the ensemble as a single file for easier use.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.app.model import EnsembleModel

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

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find model directory
    model_dir = Path("../src/model").resolve()
    if not model_dir.exists():
        model_dir = Path("./src/model").resolve()
    if not model_dir.exists():
        model_dir = Path(os.path.join(os.path.dirname(__file__), "../src/model")).resolve()
    
    print(f"Looking for models in: {model_dir}")
    
    # Model file paths
    model_files = {
        "efficientnet": model_dir / "efficientnet_pneumonia.pth",
        "swin": model_dir / "swin_best-2.pth",
        "resnet": model_dir / "resnet_best.pth"
    }
    
    # Verify all model files exist
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            print(f"Error: {model_name} model file not found at {model_path}")
            return
        print(f"Found {model_name} model at: {model_path}")
    
    try:
        # Create the wrapper model
        unified_model = EnsembleModelWrapper()
        unified_model.to(device)
        
        # Load parameters for each model
        # EfficientNet
        efficientnet_state = torch.load(model_files["efficientnet"], map_location=device)
        unified_model.efficientnet.load_state_dict(efficientnet_state)
        print("EfficientNet model loaded successfully")
        
        # Swin
        swin_state = torch.load(model_files["swin"], map_location=device)
        unified_model.swin.load_state_dict(swin_state, strict=False)
        print("Swin model loaded successfully")
        
        # ResNet
        resnet_state = torch.load(model_files["resnet"], map_location=device)
        unified_model.resnet.load_state_dict(resnet_state, strict=False)
        print("ResNet model loaded successfully")
        
        # Set to evaluation mode
        unified_model.eval()
        
        # Save the unified model
        output_path = model_dir / "unified_ensemble_model.pth"
        torch.save(unified_model.state_dict(), output_path)
        print(f"Unified ensemble model saved to: {output_path}")
        
        # Test loading the model
        test_model = EnsembleModelWrapper()
        test_model.load_state_dict(torch.load(output_path, map_location=device))
        test_model.eval()
        print("Successfully loaded the unified model for testing")
        
        # Create a sample input to verify the model works
        sample_input = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            output = test_model(sample_input)
        
        print(f"Model output shape: {output.shape}")
        print("Unified ensemble model creation and testing completed successfully!")
        
    except Exception as e:
        print(f"Error creating unified ensemble model: {e}")

if __name__ == "__main__":
    main()