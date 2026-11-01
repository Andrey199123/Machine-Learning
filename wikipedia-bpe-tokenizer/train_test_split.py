# save as prepare_bins_v2.py
# Run with: python3 prepare_bins_v2.py
import numpy as np
import os
from transformers import PreTrainedTokenizerFast

# ---------------- CONFIG ----------------
TOK_PATH = "BPE_tokens.txt"
TOKENIZER_PATH = "tokenizer_bpe/tokenizer.json" # Adjust if your path is different
VAL_FRACTION = 0.01      
DTYPE = np.uint16        

# ---------------- SETUP SEPARATOR ----------------
# We need to find what number your tokenizer uses for a "New Line" or "Space"
# to separate articles.
if os.path.exists(TOKENIZER_PATH):
    print(f"Loading tokenizer from {TOKENIZER_PATH} to find separator ID...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    
    # We encode a pure newline to see what ID it gets
    sep_ids = tokenizer.encode("\n")
    if not sep_ids:
        # Fallback: try encoding a space if newline is ignored
        sep_ids = tokenizer.encode(" ")
    
    SEPARATOR_ID = sep_ids[0]
    print(f"✓ Using Separator Token ID: {SEPARATOR_ID} (Represents a newline/space)")
else:
    # If we can't find the tokenizer, we crash to prevent bad data
    print(f"ERROR: Could not find tokenizer at {TOKENIZER_PATH}")
    print("We need it to know what ID to put between articles.")
    exit(1)

# ---------------- FUNCTIONS ----------------
def count_tokens_and_lines(path):
    """
    Counts total tokens AND adds 1 separator token per line.
    """
    total_tokens = 0
    total_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The tokens in the line + 1 separator at the end
            total_tokens += len(line.split()) + 1 
            total_lines += 1
    return total_tokens, total_lines

# ---------------- MAIN ----------------
print("Counting tokens (including new separators)...")
n_tokens, n_lines = count_tokens_and_lines(TOK_PATH)
print(f"Total tokens: {n_tokens} (including {n_lines} inserted separators)")

n_val = int(n_tokens * VAL_FRACTION)
n_train = n_tokens - n_val
print(f"Train tokens: {n_train}")
print(f"Val tokens:   {n_val}")

# Create the binary files
train_mem = np.memmap("train_BPE.bin", dtype=DTYPE, mode="w+", shape=(n_train,))
val_mem   = np.memmap("val_BPE.bin",   dtype=DTYPE, mode="w+", shape=(n_val,))

i_train = 0
i_val = 0

print("Writing binary files with separators...")

with open(TOK_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # 1. Parse the IDs from text
        # We assume the file is space-separated integers
        ids = np.fromstring(line, sep=" ", dtype=DTYPE)
        
        if ids.size == 0:
            continue

        # 2. Append the Separator ID to the end of this article
        # This fixes the "AccessibleComputingAnarchism" bug!
        ids = np.append(ids, [SEPARATOR_ID]).astype(DTYPE)

        # 3. Write to memory map (Train first, then Val)
        remaining_train = n_train - i_train
        if remaining_train > 0:
            take = min(remaining_train, ids.size)
            train_mem[i_train:i_train + take] = ids[:take]
            i_train += take
            ids = ids[take:]

        if ids.size > 0:
            take = ids.size
            val_mem[i_val:i_val + take] = ids
            i_val += take

# Flush to disk to save
train_mem.flush()
val_mem.flush()

print("\n" + "="*40)
print("DONE! New binary files created.")
print(f"✓ Inserted {n_lines} separator tokens (ID: {SEPARATOR_ID})")
print("You MUST retrain your model now.")
print("="*40)