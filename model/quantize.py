import torch

weights = torch.load("lenet_fp32.pt")

for name, tensor in weights.items():

    if name.endswith(".weight"):

        scale = tensor.abs().max() / 127
        q = torch.clamp(torch.round(tensor / scale), -127, 127)

        deq  = q* scale

        error = (tensor - deq).abs().max()

        print(f"Name: {name} scale: {scale.item()} error: {error.item()}")
