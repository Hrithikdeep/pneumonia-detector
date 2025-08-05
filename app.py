import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

#  Set Streamlit page configuration
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩺")

#  Define class names
class_names = ['Normal', 'Pneumonia']

#  Load the trained model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)  # We’re not loading ImageNet weights
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes
    model.load_state_dict(torch.load("models/model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ️ Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#  Streamlit UI
st.title("🩺 Pneumonia Detector")
st.markdown("Upload a chest X-ray image, and this model will predict whether the patient has **Pneumonia** or is **Normal**.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    #  Predict button
    if st.button("🔍 Predict"):
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]
        
        #  Show result
        st.success(f" Prediction: **{prediction}**")

