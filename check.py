import torch

a = torch.load("quant_4.bin")

for key in a.keys():
    print(key)
    print(a[key])
    print("--------------")
