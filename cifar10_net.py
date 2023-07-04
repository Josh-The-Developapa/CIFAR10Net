import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Train dataset
train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transform)

# Test dataset
test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Function to load our data for us
def LoadData(dataset, batch_size, shuffle):
    """Function to load our data into an
    instantiated dataloader when the function is called"""
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)
    return dataloader


train_loader = LoadData(train_set, 16, True)
test_loader = LoadData(test_set, 16, False)


# Function for inline image display
def matplotlib_imshow(batch, num_images):
    """Function for producing an inline display
    of a set number images in a batch"""
    batch = [(batch[0][0:num_images]), batch[1][0:num_images]]
    fig, axes = plt.subplots(1, len(batch[0]), figsize=(10, 5))
    for idx, img in enumerate(batch[0]):
        imgu = img * std[0] + mean[0]  # unnormalise
        ax = axes[idx]
        ax.imshow(imgu.permute(1, 2, 0))
        ax.set_title(classes[batch[1][idx]])

    plt.tight_layout()
    plt.show()


# Display the first batch only
for batch in train_loader:
    matplotlib_imshow(batch, 5)
    break


# Create our Model class
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = self.linear_layer_stack(x)
        return x


# Instantiate our model
model = CIFAR10Net()

# Define our criterion/ loss_fn
criterion = nn.CrossEntropyLoss()

# Define our optimiser
optimiser = optim.SGD(model.parameters(), lr=0.01)


# Train our Model
def train_model(
    model=model, criterion=criterion, optimiser=optimiser, dataloader=train_loader
):
    """A function to train our model.
    It passes the entire dataset from a loader through the model
    Must be executed per epoch
    """

    for idx, batch in enumerate(dataloader):
        imgs, labels = batch[0], batch[1]

        # Zero gradients
        optimiser.zero_grad()

        # Forward pass
        predictions = model(imgs)

        # Calculate loss
        loss = criterion(predictions, labels)

        # Back propagation and update parameters
        loss.backward()
        optimiser.step()

        if idx % 100 == 0:
            print(f"Loss: {loss} | Batch: {idx}/{len(dataloader)}")


# Evaluate our Model's performance after training
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    # Feel free to adjust the code below to train an untrained model
    model1 = CIFAR10Net()
    model1.load_state_dict(torch.load("cifar10_net.pt"))
    epochs = 50

    # Put our Model in training mode
    model1.train()
    for epoch in range(epochs):
        print(f"\n Epoch:{epoch}  \n ----------------")
        train_model(model1, criterion, optimiser, dataloader=train_loader)

    # Save our model parameters
    torch.save(model1.state_dict(), "cifar10_net.pt")

    model1.eval()
    with torch.no_grad():
        test_loop(test_loader, model1, criterion)
        # 60.3% Accurate
