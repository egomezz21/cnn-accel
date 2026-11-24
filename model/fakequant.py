import torch
import json
import torch.nn.functional as F
from lenet import LeNet
from torchvision import datasets, transforms

def fq(x, scale, lo, hi):
    # TODO: divide by scale, round, clamp to [lo, hi], multiply by scale, return it
    q = torch.clamp(torch.round(x / scale), lo, hi)
    deq  = q * scale
    return deq

weights = torch.load("lenet_fp32.pt")

new_weights = {}

for name, tensor in weights.items():

    if name.endswith(".weight"):

        scale = tensor.abs().max() / 127

        new_weights[name] = fq(tensor, scale, -127, 127)

    else:
        new_weights[name] = tensor

model = LeNet()
model.load_state_dict(new_weights)
model.eval()
tf = transforms.ToTensor()
test = datasets.MNIST("data", train=False, download=True, transform=tf)
test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

correct = 0

scales = json.load(open("scales.json"))

with torch.no_grad():
    for x, y in test_loader:

        x = fq(x, scales["in"], 0, 127)
        x = F.relu(model.conv1(x))
        x = fq(x, scales["a1"], 0, 127)
        x = F.max_pool2d(x, 2)

        x = F.relu(model.conv2(x))
        x = fq(x, scales["a2"], 0, 127)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(model.fc1(x))
        x = fq(x, scales["a3"], 0, 127)

        x = F.relu(model.fc2(x))
        x = fq(x, scales["a4"], 0, 127)
        x = model.fc3(x)

        correct += (x.argmax(1) == y).sum().item()

print(f"full int8 weight + activations test accuracy: {correct / len(test):.4f}")