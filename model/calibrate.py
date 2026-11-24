from lenet import LeNet
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import json

model = LeNet()

model.eval()

model.load_state_dict(torch.load("lenet_fp32.pt"))

tf = transforms.ToTensor()   # pixels become floats in [0, 1]
train = datasets.MNIST("data", train=True, download=True, transform=tf)
train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=False)

max1 = max2 = max3 = max4 = 0.0

with torch.no_grad():
    for i, (x, y) in enumerate(train_loader):

        if i == 10:
            break

        x = F.relu(model.conv1(x))
        max1 = max(max1, x.max().item())
        x = F.max_pool2d(x, 2)

        x = F.relu(model.conv2(x))
        max2 = max(max2, x.max().item())
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(model.fc1(x))
        max3 = max(max3, x.max().item())

        x = F.relu(model.fc2(x))
        max4 = max(max4, x.max().item())


print(f"Max1: {max1} Max2: {max2} Max3: {max3} Max4: {max4} Scales: {max1/127} {max2/127} {max3/127} {max4/127}")       

scales = {"in": 1/127, "a1": max1/127, "a2": max2/127, "a3": max3/127, "a4": max4/127}
with open("scales.json", "w") as f:
    json.dump(scales, f, indent=2)