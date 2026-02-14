import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SAMPLES_DIR = os.path.join(OUTPUTS_DIR, "samples")
PREDS_DIR = os.path.join(OUTPUTS_DIR, "preds")

# ✅ Ensure folders exist (prevents FastAPI crash)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

app = FastAPI(title="Cats vs Dogs API")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Serve static folders
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
app.mount("/preds", StaticFiles(directory=PREDS_DIR), name="preds")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).resize((224,224)).convert("RGB")
    arr = np.array(img) / 255.0
    pred = model.predict(arr.reshape(1,224,224,3))[0][0]
    return {
        "cat_probability": float(1 - pred),
        "dog_probability": float(pred)
    }

@app.get("/gallery", response_class=HTMLResponse)
def gallery():
    sample_imgs = os.listdir(SAMPLES_DIR)
    pred_imgs = os.listdir(PREDS_DIR)

    html = "<h1>Training Samples</h1>"
    if not sample_imgs:
        html += "<p>No training sample images found.</p>"
    for img in sample_imgs:
        html += f'<img src="/samples/{img}" width="200" style="margin:10px"/>'

    html += "<h1>Prediction Samples</h1>"
    if not pred_imgs:
        html += "<p>No prediction images found.</p>"
    for img in pred_imgs:
        html += f'<img src="/preds/{img}" width="200" style="margin:10px"/>'

    return html
