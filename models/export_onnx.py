import os
import torch
import torch.nn as nn
from torchvision import models

class ResNet50MLP(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.net = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.net(features)

def main():
    model_path = "/home/minhchau/Documents/Vision-Based-Fault-Detection/models/best_model.pth"
    onnx_path = "/home/minhchau/Documents/Vision-Based-Fault-Detection/models/best_model.onnx"
    
    print("Loading PyTorch model...")
    model = ResNet50MLP(num_classes=4)
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Strip prefix 'net.'
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            clean_state_dict[k[4:]] = v
        else:
            clean_state_dict[k] = v
            
    model.net.load_state_dict(clean_state_dict)
    model.eval()
    
    # Dummy input representing batch of 1, 3 color channels, 224x224 image
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    
    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # standard opset version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    main()
