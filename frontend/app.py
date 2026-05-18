from flask import (
    Flask,
    jsonify,
    render_template,
    request
)

import torch
from torchvision import transforms
from PIL import Image

from model import PneumoniaModel


app = Flask(__name__)

# ---------------- MODEL ---------------- #

device = torch.device("cpu")

model = PneumoniaModel().to(device)

model.load_state_dict(
    torch.load(
        "global_model.pth",
        map_location=device
    )
)

model.eval()

# ---------------- CLASSES ---------------- #

classes = [
    "COVID",
    "Normal",
    "Viral Pneumonia"
]

# ---------------- TRANSFORM ---------------- #

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- DASHBOARD DATA ---------------- #

training_data = {
    "round": 1,
    "accuracy": 87,
    "privacy_budget": 1.1,
    "hospital_a": "Training",
    "hospital_b": "Training",
    "hospital_c": "Training"
}

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/metrics")
def metrics():

    return jsonify(training_data)


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    image = Image.open(file).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

        prediction = classes[predicted.item()]

    return render_template(
        "index.html",
        prediction=prediction
    )


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    app.run(debug=True)