import torch

# Simple manual tokenization and vocabulary
text = "I really love learning about artificial intelligence in Sri Lanka"

# Manual tokenization
tokens = text.lower().split()
print("Tokens:", tokens)

# Simple vocabulary
vocab = {"<unk>": 0, "i": 1, "really": 2, "love": 3, "learning": 4, "about": 5, "artificial": 6, "intelligence": 7, "in": 8, "sri": 9, "lanka": 10}

# Convert to numbers
numerical = [vocab.get(word, 0) for word in tokens]
print("Numerical version:", numerical)

# Convert to tensor
tensor = torch.tensor(numerical)
print("Tensor:", tensor)
print("Tensor shape:", tensor.shape)
