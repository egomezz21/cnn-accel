from lenet import LeNet
from torchvision import datasets, transforms
import torch

weights = torch.load("lenet_fp32.pt")

new_weights = {}

for name, tensor in weights.items():

    if name.endswith(".weight"):

        scale = tensor.abs().max() / 127
        q = torch.clamp(torch.round(tensor / scale), -127, 127)

        deq  = q* scale

        error = (tensor - deq).abs().max()

        new_weights[name] = deq 

        print(f"Name: {name} scale: {scale.item()} error: {error.item()}")
    else:
        new_weights[name] = tensor
        
model = LeNet()
model.load_state_dict(new_weights)

tf = transforms.ToTensor()
test = datasets.MNIST("data", train=False, download=True, transform=tf)
test_loader = torch.utils.data.DataLoader(test, batch_size=1000)
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        correct += (model(x).argmax(1) == y).sum().item()
print(f"int8 weight test accuracy: {correct / len(test):.4f}")