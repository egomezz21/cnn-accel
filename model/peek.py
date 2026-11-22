import torch

weights = torch.load("lenet_fp32.pt")
for name, tensor in weights.items():
    print(name, tuple(tensor.shape), "min", tensor.min().item(), "max", tensor.max().item())

print(weights["conv1.weight"][0])   # the first 5x5 filter of conv1