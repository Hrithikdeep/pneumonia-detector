# 🩺 Pneumonia Detection from Chest X-rays using Deep Learning (ResNet18 + Streamlit)

Detect Pneumonia from chest X-ray images using a ResNet18 CNN model trained with PyTorch and deployed with Streamlit. This end-to-end pipeline covers data preprocessing, model training, evaluation, and real-time predictions.
---


## Project Demo

Web App Demo:
[Click here to try the live app](https://pneumonia-detector-hgjojhhexcpeceajgy8pje.streamlit.app/)


GIF Preview:
a chest X-ray image and get instant Pneumonia prediction with probability scores.
 
 ## 🖼️ App Screenshot

Upload a chest X-ray and the app will predict Pneumonia or Normal.

![App Screenshot](<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/a39748d0-ebe2-4fc8-83d1-ba6b1d16bf15" />
)


## Features

✅ Detect Pneumonia vs Normal X-rays

✅ Built with PyTorch (ResNet18)

✅ Real-time Streamlit app for predictions

✅ Modular structure for training, evaluation, and deployment

✅ Ready for cloud deployment (Streamlit Cloud / Hugging Face Spaces

## 📁 Folder Structure

```
pneumonia-detector/
├── data/                     # Dataset (train/val/test)
├── models/                   # Trained model (model.pth)
├── outputs/                  # Optional prediction logs or visualizations
├── app.py                    # Streamlit web app
├── train_model.py            # Model training & evaluation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Files/folders to ignore


---

## 🔧 Installation

```git clone <YOUR_REPO_URL>
cd pneumonia-detector
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

``streamlit run app.py

```

---


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

