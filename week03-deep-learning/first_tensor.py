import torch

# TODO 1: Create a tensor with numbers [1, 2, 3, 4, 5]
tensor1 = torch.tensor([1,2,3,4,5])

print("Tensor 1:", tensor1)

# TODO 2: Create a 2D tensor (matrix) of shape 2x3 with random numbers
tensor2 = torch.randn(2,3)

print("Tensor 2 shape:", tensor2.shape)

# TODO 3: Add 5 to every element in tensor1
tensor3 = tensor1 + 5

print("Tensor 3:", tensor3)