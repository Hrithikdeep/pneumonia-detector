# 🩺 Pneumonia Detection from Chest X-rays using Deep Learning (ResNet18 + Streamlit)

This is a complete end-to-end medical imaging project that detects **Pneumonia** in chest X-ray images using a Convolutional Neural Network (CNN) with **ResNet18** architecture. The model is trained using PyTorch and deployed via a simple **Streamlit app**.

---

## 📁 Folder Structure

```
pneumonia-detector/
├── data/                     # Dataset (train/val/test folders inside)
├── models/                   # Trained model is saved here as model.pth
├── outputs/                  # (Optional) Store prediction outputs or logs
├── app.py                    # Main Streamlit app
├── train_model.py            # Training and evaluation script
├── requirements.txt          # Required Python packages
├── README.md                 # Project README
├── .gitignore                # Git ignore file
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

**requirements.txt should include:**
```
torch
torchvision
streamlit
Pillow
scikit-learn
matplotlib
```

---

## 🏋️‍♀️ Train the Model

```bash
python train_model.py
```

Example output:
```
✅ Starting training...
Epoch 1/5 | Loss: 14.4386 | Accuracy: 0.9657
...
Epoch 5/5 | Loss: 1.1520 | Accuracy: 0.9981
✅ Model saved to models/model.pth
```

---

## 🌐 Run the Streamlit App

```Demo [https:/(/pneumonia-detector-hgjojhhexcpeceajgy8pje.streamlit.app/)]
```

---

## 🖼️ Streamlit App Screenshot

Upload a chest X-ray and the app will predict Pneumonia or Normal.

![App Screenshot](<img width="250" height="150" alt="Screenshot 2025-08-06 at 11 54 37 AM" src="https://github.com/user-attachments/assets/478de1b9-b34a-4335-972f-e46c8e9014d3" />
)

---

## 🚀 Deployment Options

### ✅ Streamlit Cloud
- Push all files to GitHub (including `model.pth` using Git LFS)
- Go to [streamlit.io/cloud]https:/(/pneumonia-detector-hgjojhhexcpeceajgy8pje.streamlit.app/)
- Link your GitHub repo
- Deploy the `app.py` file
- 
### ✅ Hugging Face Spaces
- Create a new Space using `Streamlit` SDK
- Upload all files including the model
- App will be hosted automatically

---

## 💾 How to Push Large Model to GitHub (Git LFS)

```bash
# Install LFS
git lfs install

# Track your model file
git lfs track "*.pth"
git add .gitattributes

# Add model and push
git add models/model.pth
git commit -m "Add model"
git push origin main
```

---

## 📄 .gitignore Example

```
# Python
__pycache__/
*.pyc

# Model and data
models/
*.pth
*.pt
*.h5

# Environments
.env
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
```

---

## 🙌 Author

**Hrithik Deep**  
📧 hrithikdeep.ds@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/hrithikdeep)

