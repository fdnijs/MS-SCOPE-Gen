import json, pandas as pd
from apetokenizer.ape_tokenizer import APETokenizer

# Load vocab
with open("data/tokenizer/vocab_no_fp.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Add 4096 FP tokens
for i in range(4096):
    vocab[f"<fp{i:04d}>"] = len(vocab)

# Save extended vocab
with open("data/tokenizer/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=4)