import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from model import PneumoniaModel
from dataset import load_data

# Device
device = torch.device("cpu")

# Load model
model = PneumoniaModel().to(device)

# Load data
# Load data
trainloader = load_data(
    r"C:\PBL-project\data\hospital_C"
)

testloader = trainloader


# ---------------- TRAIN FUNCTION ---------------- #

def train(model, trainloader, epochs=10):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    for epoch in range(epochs):

        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()


# ---------------- TEST FUNCTION ---------------- #

def test(model, testloader):

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss = 0.0

    model.eval()

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return loss, accuracy


# ---------------- FLOWER CLIENT ---------------- #

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

        model.load_state_dict(state_dict, strict=True)

    
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

        loss, accuracy = test(model, testloader)

        print(f"Accuracy: {accuracy}")

        return (
            float(loss),
            len(testloader.dataset),
            {"accuracy": float(accuracy)}
        )


# ---------------- START CLIENT ---------------- #

if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )