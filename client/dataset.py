import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =========================
# DEVICE
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# =========================
# DATASET LOADING
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_data(path):

    dataset = datasets.ImageFolder(
        root=path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    return loader

# =========================
# CHANGE THIS FOR EACH HOSPITAL
# =========================

trainloader = load_data(
    r"C:\PBL-project\data\hospital_A"
)

# Hospital B
# trainloader = load_data(
#     r"C:\PBL-project\data\hospital_B"
# )

# Hospital C
# trainloader = load_data(
#     r"C:\PBL-project\data\hospital_C"
# )

# =========================
# MODEL
# =========================

class PneumoniaModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights="DEFAULT"
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):

        return self.model(x)

model = PneumoniaModel().to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# TRAIN FUNCTION
# =========================

def train(model, trainloader, epochs=1):

    model.train()

    for epoch in range(epochs):

        for images, labels in trainloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

# =========================
# TEST FUNCTION
# =========================

def test(model, testloader):

    model.eval()

    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss += criterion(
                outputs,
                labels
            ).item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = correct / total

    return loss, accuracy

# =========================
# FLOWER CLIENT
# =========================

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        model.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(self, parameters, config):

        print("Training Started...")

        self.set_parameters(parameters)

        train(model, trainloader)

        print("Training Finished")

        # Save trained model
        torch.save(
            model.state_dict(),
            "global_model.pth"
        )

        return (
            self.get_parameters(config),
            len(trainloader.dataset),
            {}
        )

    def evaluate(self, parameters, config):

        print("Evaluating...")

        self.set_parameters(parameters)

        loss, accuracy = test(
            model,
            trainloader
        )

        print("Accuracy:", accuracy)

        return (
            float(loss),
            len(trainloader.dataset),
            {
                "accuracy": float(accuracy)
            }
        )

# =========================
# START CLIENT
# =========================

if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )