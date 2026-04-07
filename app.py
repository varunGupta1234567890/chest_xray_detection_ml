
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import uvicorn
import cv2
import numpy as np
import os
import pickle
import time
from tensorflow.keras.models import load_model


app = FastAPI()


model = load_model('models/CNN_Covid19_Xray_Version.h5')
le = pickle.load(open("models/Label_encoder.pkl", 'rb'))


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")


templates = Jinja2Templates(directory="templates")


def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    idx = np.argmax(predictions)

    label = le.inverse_transform([idx])[0]
    confidence = float(predictions[0][idx])

    return label, confidence


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
    
        filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        
        label, confidence = process_image(file_path)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_path": f"/uploads/{filename}",   # IMPORTANT
            "predicted_label": label,
            "confidence_score": round(confidence * 100, 2)
        })

    except Exception as e:
        return HTMLResponse(f"<h2>Error: {str(e)}</h2>")


@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)