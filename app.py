import os
from flask import Flask, request, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pneu.pth", map_location=device))
model = model.to(device)
model.eval()


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["NORMAL", "PNEUMONIA"]

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]


# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_img = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

       
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        
        prediction = predict(filepath)
        uploaded_img = filepath

    return render_template("index.html", prediction=prediction, uploaded_img=uploaded_img)


if __name__ == "__main__":
    app.run(debug=True)
