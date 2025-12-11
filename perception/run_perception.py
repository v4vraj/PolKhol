# perception/run_perception.py
import argparse
import json
import os
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel

# choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# categories for zero-shot classification
labels = ["pothole", "garbage", "water leak", "tree fall", "blocked road", "other"]

def classify_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Cannot open image: {e}")

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # [1, num_labels]
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0].tolist()
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    category = labels[best_idx]
    confidence = float(probs[best_idx])
    embedding = outputs.image_embeds[0].cpu().numpy().tolist()

    return category, confidence, embedding, probs

def main(input_path, output_path):
    category, confidence, embedding, probs = classify_image(input_path)
    result = {
        "category": category,
        "confidence": confidence,
        "probs": probs,
        "embedding": embedding,
        "model": model_name,
        "device": device,
        "input": os.path.basename(input_path)
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"WROTE RESULT -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    args = parser.parse_args()
    main(args.input, args.output)
