from datasets import load_dataset

# Load the dataset
dataset = load_dataset("stanfordnlp/imdb")

# Shuffle and take balanced samples
train_data = dataset["train"].shuffle(seed=42).select(range(2000))
test_data = dataset["test"].shuffle(seed=42).select(range(500))

print("Train samples (subset):", len(train_data))
print("Test samples (subset):", len(test_data))

print("\n--- First example ---")
print("Text:", train_data[0]["text"][:300], "...")
print("Label:", train_data[0]["label"])

print("\n--- Label distribution ---")
labels = [example["label"] for example in train_data]
print("Positive (1):", labels.count(1))
print("Negative (0):", labels.count(0))
