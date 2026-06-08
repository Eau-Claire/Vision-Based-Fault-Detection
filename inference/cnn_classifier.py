import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class CNNClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.image_size = 224
        
        # Check if ONNX model
        if model_path.endswith('.onnx'):
            if ort is None:
                raise ImportError("onnxruntime is required to run .onnx models. Please run: pip install onnxruntime")
            print(f"--- Loading ONNX model from {model_path} ---")
            
            # Select execution providers based on target device
            providers = ['CPUExecutionProvider']
            if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider'] + providers
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.class_names = ['damaged', 'disconnected', 'misroute', 'normal']
            self.model_type = "onnx"
        else:
            # Load state dict first to inspect keys
            state_dict = torch.load(model_path, map_location='cpu')
            is_resnet_mlp = any(k.startswith('net.') for k in state_dict.keys())
            
            if is_resnet_mlp:
                print(f"--- Detected ResNet50 + MLP Classifier model from {model_path} ---")
                self.class_names = ['damaged', 'disconnected', 'misroute', 'normal']
                
                # Build ResNet50 + MLP
                try:
                    self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                except Exception:
                    self.backbone = models.resnet50(pretrained=True)
                self.backbone.fc = nn.Identity()
                self.backbone.to(self.device)
                self.backbone.eval()
                
                self.net = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, len(self.class_names))
                )
                
                # Strip 'net.' prefix and load weights
                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('net.'):
                        clean_state_dict[k[4:]] = v
                    else:
                        clean_state_dict[k] = v
                self.net.load_state_dict(clean_state_dict)
                self.net.to(self.device)
                self.net.eval()
                
                self.model_type = "resnet_mlp"
            else:
                print(f"--- Detected MobileNetV3 Small Classifier model from {model_path} ---")
                self.class_names = ['Clean-Insulator', 'Dirt-Insulator', 'Broken-Disc', 'Broken-Glass', 'Pollution-Flashover']
                
                self.model = models.mobilenet_v3_small(weights=None)
                in_features = self.model.classifier[3].in_features
                self.model.classifier[3] = nn.Linear(in_features, len(self.class_names))
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                
                self.model_type = "mobilenet"
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image):
        input_tensor = self.transform(pil_image).unsqueeze(0)
        
        if self.model_type == "onnx":
            # Convert PyTorch tensor to numpy for ONNX runtime
            input_numpy = input_tensor.numpy()
            input_name = self.session.get_inputs()[0].name
            
            # Run inference using ONNX session
            outputs = self.session.run(None, {input_name: input_numpy})
            logits = outputs[0][0]
            
            # Apply Softmax using numpy
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            predicted = np.argmax(probabilities)
            confidence = probabilities[predicted]
            
            return self.class_names[predicted], float(confidence)
            
        else:
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                if self.model_type == "resnet_mlp":
                    features = self.backbone(input_tensor)
                    outputs = self.net(features)
                else:
                    outputs = self.model(input_tensor)
                    
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            return self.class_names[predicted.item()], confidence.item()


