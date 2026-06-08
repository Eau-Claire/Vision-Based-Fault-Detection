import os
import sys
import argparse
from PIL import Image
from cnn_classifier import CNNClassifier

def main():
    parser = argparse.ArgumentParser(description="Test the CNN Classifier (ResNet50 + MLP or MobileNetV3)")
    parser.add_argument("--model", type=str, default="../models/best_model.pth", help="Path to the model .pth file")
    parser.add_argument("--image", type=str, required=True, help="Path to a test image or directory of images")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    # Verify model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'")
        sys.exit(1)

    # Load classifier
    print(f"Loading classifier from {args.model} on {args.device}...")
    try:
        classifier = CNNClassifier(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        sys.exit(1)

    # Verify image path
    if not os.path.exists(args.image):
        print(f"Error: Target path '{args.image}' does not exist")
        sys.exit(1)

    # Process single image or directory of images
    if os.path.isdir(args.image):
        image_files = [
            os.path.join(args.image, f) 
            for f in os.listdir(args.image) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        if not image_files:
            print(f"No valid images found in directory '{args.image}'")
            sys.exit(0)

        print(f"Found {len(image_files)} images in '{args.image}'. Running classification...")
        print("-" * 60)
        print(f"{'Image Filename':<40} | {'Prediction':<15} | {'Confidence':<10}")
        print("-" * 60)
        
        for img_path in sorted(image_files):
            try:
                img = Image.open(img_path).convert("RGB")
                label, confidence = classifier.predict(img)
                filename = os.path.basename(img_path)
                print(f"{filename:<40} | {label:<15} | {confidence:.2%}")
            except Exception as e:
                print(f"{os.path.basename(img_path):<40} | Error: {str(e)}")
        print("-" * 60)
    else:
        try:
            img = Image.open(args.image).convert("RGB")
            label, confidence = classifier.predict(img)
            print("-" * 40)
            print(f"Image:      {args.image}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 40)
        except Exception as e:
            print(f"Error classifying image '{args.image}': {e}")

if __name__ == "__main__":
    main()
