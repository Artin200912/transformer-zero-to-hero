import torch

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

# Example size
size = 5
mask = causal_mask(size)

print(mask)