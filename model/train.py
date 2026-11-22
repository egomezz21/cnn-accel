from lenet import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)

tf = transforms.ToTensor()   # pixels become floats in [0, 1]
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)   #serves the training images in batches of 64
test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

model = LeNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop repeated for each batch
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)  #measures how innaccurate the scores are compared to the true labels
        loss.backward()           #calculates the direction to nudge each weight to reduce the loss
        opt.step()          #nudges

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
    print(f"epoch {epoch}: test accuracy {correct / len(test):.4f}")

torch.save(model.state_dict(), "lenet_fp32.pt")