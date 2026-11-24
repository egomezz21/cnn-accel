import json
import torch
import numpy as np

params = {}   # make this before the loop

scales = json.load(open("scales.json"))
weights = torch.load("lenet_fp32.pt")

layers = [
    # (layer name, s_in key, s_out key)
    ("conv1", "in", "a1"),
    ("conv2", "a1", "a2"),
    ("fc1",   "a2", "a3"),
    ("fc2",   "a3", "a4"),
    ("fc3",   "a4", None),     # None = no requantization
]

for name, in_key, out_key in layers:
    w = weights[name + ".weight"]
    b = weights[name + ".bias"]
    s_in = scales[in_key]
    # TODO: s_w, q_w, q_bias
    s_w = (w.abs().max() / 127).item()
    q_w = torch.clamp(torch.round(w / s_w), -127, 127).to(torch.int8)
    q_bias = torch.round(b / (s_w * s_in)).to(torch.int32) 


    params[name + "_w"] = q_w.numpy()
    params[name + "_b"] = q_bias.numpy()

    # TODO: if out_key is not None: s_out, M, m, n — then print them

    if out_key is not None:
        s_out = scales[out_key]
        M = (s_w * s_in) / s_out
        n = 0
        while round(M * 2**(n + 1)) < 32768:
            n += 1
        m = round(M * 2**n)

        params[name + "_m"] = m
        params[name + "_n"] = n

        print(f"{name} M: {M} m: {m} n: {n}")

np.savez("params.npz", **params)