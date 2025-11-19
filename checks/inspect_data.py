import os
import numpy as np

p = "test/param0/1dbb02c20bb93af81c1b3b2ed8d13bf8/pressure.th"

# 1) try numpy load (some projects saved numpy arrays with .th)
try:
    a = np.load(p, allow_pickle=True)
    print("np.load ->", type(a), getattr(a, "shape", None), getattr(a, "dtype", None))
except Exception as e:
    print("np.load failed:", e)

# 2) try pickle (gives type info; dangerous for untrusted files)
import pickle
try:
    with open(p, "rb") as f:
        obj = pickle.load(f)
    print("pickle ->", type(obj))
    if hasattr(obj, "shape"):
        print("shape:", getattr(obj, "shape", None), "dtype:", getattr(obj, "dtype", None))
except Exception as e:
    print("pickle load failed:", e)

# 3) try torch (most likely)
try:
    import torch
    obj = torch.load(p, map_location="cpu")
    print("torch.load ->", type(obj))
    # typical: obj is a torch.Tensor or dict / OrderedDict
    if isinstance(obj, torch.Tensor):
        print("tensor shape:", tuple(obj.shape), "dtype:", obj.dtype)
    else:
        # print keys if dict-like
        if hasattr(obj, "keys"):
            print("dict keys:", list(obj.keys())[:10])
        else:
            print(repr(obj)[:200])
except Exception as e:
    print("torch.load failed:", e)

