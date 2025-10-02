from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from PIL import Image
import io
import os

from retouch import RetouchModel  # class trong RetouchML

app = FastAPI()

# load model (chỉ load 1 lần khi server start)
model = RetouchModel(pretrained=True, device="cpu")

@app.post("/beautify")
async def beautify(file: UploadFile = File(...)):
    # đọc ảnh input
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # chạy qua model
    result = model.retouch(img)

    # lưu output
    out_path = "output.jpg"
    result.save(out_path)

    return FileResponse(out_path, media_type="image/jpeg")
