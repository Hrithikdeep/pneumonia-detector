import torch
from .model import load_model
from .utils import preprocess_image

CLASS_NAMES = ['Normal', 'Pneumonia']
model = load_model()

def predict(image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = CLASS_NAMES[predicted.item()]
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
    return label, confidence
