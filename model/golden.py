import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F

p = np.load("params.npz")

def requant(acc, m, n):
    # TODO: (acc * m + 2^(n-1)) >> n, then clamp to [0, 127]
    acc = np.asarray(acc, dtype=np.int64)
    return np.clip((acc * m + (1 << (n - 1))) >> n, 0, 127)

def conv2d_int(x, w, b):
    # x: (C, H, W) int8 input, w: (K, C, 5, 5) int8, b: (K,) int32
    # returns: (K, H-4, W-4) int64 accumulators (before requant)
    x = np.asarray(x, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    K, C, R, S = w.shape
    _, H, W = x.shape

    H_out = H - R + 1
    W_out = W - S + 1   

    out = np.zeros((K, H_out, W_out), dtype=np.int64)

    for k in range(K):
        for i in range(H_out):
            for j in range(W_out):

                p = x[:, i:i+R, j:j+S]*w[k]
                out[k, i, j] = np.sum(p) + b[k]

    return out

def maxpool2(x):
    # x: (C, H, W) → (C, H/2, W/2), max of each 2x2 block
    C, H, W = x.shape
    out = np.zeros((C, H // 2, W // 2), dtype=np.int64)

    for c in range(C):
        for i in range(H//2):
            for j in range(W//2):
                m = x[c, 2*i:2*i+2, 2*j:2*j+2]
                out[c, i, j] = np.max(m)

    return out

def fc_int(x, w, b):
    # x: (N,) input, w: (M, N), b: (M,) → (M,) int64 accumulators
    x = np.asarray(x, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    return w @ x +b

def forward(img):
    # img: (1, 28, 28) floats in [0, 1]
    # TODO: quantize input, then conv1 → requant → pool → conv2 → requant → pool
    #       → flatten → fc1 → requant → fc2 → requant → fc3 → return the 10 scores

    x = np.round(img * 127).astype(np.int64)    # quantize input: (1, 28, 28), values 0..127

    acc = conv2d_int(x, p["conv1_w"], p["conv1_b"])
    x = requant(acc, int(p["conv1_m"]), int(p["conv1_n"]))
    x = maxpool2(x)
    # TODO: conv2 → requant → pool
    acc = conv2d_int(x, p["conv2_w"], p["conv2_b"])
    x = requant(acc, int(p["conv2_m"]), int(p["conv2_n"]))
    x = maxpool2(x)
    # TODO: flatten
    x = x.flatten()
    # TODO: fc1 → requant, fc2 → requant
    acc = fc_int(x, p["fc1_w"], p["fc1_b"])
    x = requant(acc, int(p["fc1_m"]), int(p["fc1_n"]))

    acc = fc_int(x, p["fc2_w"], p["fc2_b"])
    x = requant(acc, int(p["fc2_m"]), int(p["fc2_n"]))
    # TODO: fc3 — no requant, return the raw int scores
    x = fc_int(x, p["fc3_w"], p["fc3_b"])

    return x



assert requant(1000, 24073, 24) == 1
assert requant(-500, 24073, 24) == 0
assert requant(10**6, 24073, 24) == 127
print("requant tests passed")

rng = np.random.default_rng(0)                       # seeded, so it's reproducible
x = rng.integers(0, 128, size=(6, 12, 12))           # like real int8 activations
w = rng.integers(-127, 128, size=(16, 6, 5, 5))
b = rng.integers(-1000, 1000, size=(16,))

mine = conv2d_int(x, w, b)
ref = F.conv2d(torch.tensor(x[None], dtype=torch.float64),
               torch.tensor(w, dtype=torch.float64),
               torch.tensor(b, dtype=torch.float64))[0].numpy()

assert mine.shape == (16, 8, 8), mine.shape
assert np.array_equal(mine, ref.astype(np.int64))
print("conv2d_int tests passed")

# maxpool
x = rng.integers(0, 128, size=(6, 24, 24))
ref = F.max_pool2d(torch.tensor(x, dtype=torch.float64), 2).numpy()
assert np.array_equal(maxpool2(x), ref.astype(np.int64))

# fc
x = rng.integers(0, 128, size=(256,))
w = rng.integers(-127, 128, size=(120, 256))
b = rng.integers(-1000, 1000, size=(120,))
ref = F.linear(torch.tensor(x, dtype=torch.float64),
               torch.tensor(w, dtype=torch.float64),
               torch.tensor(b, dtype=torch.float64)).numpy()
assert np.array_equal(fc_int(x, w, b), ref.astype(np.int64))

print("maxpool2 and fc_int tests passed")

tf = transforms.ToTensor()
test = datasets.MNIST("data", train=False, download=True, transform=tf)

N = 10000
correct = 0
for i in range(N):
    img, label = test[i]
    scores = forward(img.numpy())
    # TODO: if the index of the biggest score equals label, count it
    if np.argmax(scores) == label:
        correct += 1
print(f"golden model accuracy on {N} images: {correct / N:.4f}")