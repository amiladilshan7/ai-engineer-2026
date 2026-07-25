from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and take balanced subset
dataset = load_dataset("stanfordnlp/imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(2000))

# Load a simple tokenizer (we will use DistilBERT tokenizer)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Example tokenization
example_text = train_data[0]["text"]
tokens = tokenizer(example_text, truncation=True, max_length=128, padding="max_length")

print("Original Text (first 200 chars):")
print(example_text[:200], "...")
print("\nToken IDs (first 20):")
print(tokens["input_ids"][:20])
print("\nAttention Mask (first 20):")
print(tokens["attention_mask"][:20])
print("\n✅ Tokenization working!")
