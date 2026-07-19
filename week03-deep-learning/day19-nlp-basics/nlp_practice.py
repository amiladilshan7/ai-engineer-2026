import torch
from torchtext.data.utils import get_tokenizer

# TODO 1: Create tokenizer
tokenizer = get_tokenizer("basic_english")

text = "I really love learning about artificial intelligence in Sri Lanka"

# TODO 2: Tokenize the sentence
tokens = tokenizer(text)

print("Tokens:", tokens)

# TODO 3: Create a very small vocabulary manually
vocab = {"<unk>": 0, "i": 1, "love": 2, "learning": 3, "artificial": 4, "intelligence": 5}

# TODO 4: Convert tokens to numbers (numericalize)
numerical = [vocab.get(word, 0) for word in tokens]

print("Numerical version:", numerical)
