import os
import numpy as np
from torchvision import datasets, transforms
from golden import forward
import numpy as np

v = np.load("vectors/img0.npz")
print(list(v.keys()))
print(v["conv1_out"].shape)   # expect (6, 24, 24)
print(v["scores"])            # 10 ints, biggest at index 7

os.makedirs("vectors", exist_ok=True)    # makes the folder; no error if it exists

tf = transforms.ToTensor()
test = datasets.MNIST("data", train=False, download=True, transform=tf)

# TODO: find the index of the first image the golden model gets WRONG
#       (loop over test, stop at the first where argmax(scores) != label)

for i in range(len(test)):
    img, label = test[i]
    scores = forward(img.numpy())
    if np.argmax(scores) != label:
        wrong_i = i
        break

# TODO: indices = the first 10 (0–9) plus that wrong one
if wrong_i >= 10:
    indices = list(range(10)) + [wrong_i]
else:
    indices = list(range(10))

for i in indices:
    img, label = test[i]
    trace = {}
    scores = forward(img.numpy(), trace)
    # TODO: add label and index to trace
    trace["label"] = label
    trace["index"] = i
    # TODO: save with np.savez(f"vectors/img{i}.npz", **trace)
    np.savez(f"vectors/img{i}.npz", **trace)
    pred = np.argmax(scores)
    # TODO: print index, label, prediction
    print(f"img {i}: label {label}, predicted {pred}")
