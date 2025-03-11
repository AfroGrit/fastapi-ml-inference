from fastapi import FastAPI
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Load a pre-trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True).to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict(file: bytes):
    image = Image.open(io.BytesIO(file))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return {"prediction": output.argmax().item()}

