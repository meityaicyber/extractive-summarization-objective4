"""Diagnose local Gemma loading and generation speed. Run from src/."""

import os
import sys
import time
from collections import Counter

import torch

from summarizer.llm_client import GemmaClient

path = os.environ.get("GEMMA_MODEL_PATH")
if not path:
    sys.exit("Set GEMMA_MODEL_PATH first.")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

t0 = time.time()
client = GemmaClient(model_path=path)
print(f"Loaded in {time.time() - t0:.0f}s")

device_map = getattr(client.model, "hf_device_map", None)
if device_map:
    placements = Counter(str(v) for v in device_map.values())
    print("Where the weights live:", dict(placements))
    if any(p in ("cpu", "disk") for p in placements):
        print("WARNING: part of the model is offloaded to CPU or disk. Generation will be very slow.")
else:
    print("No device map reported; model device:", client.model.device)

print(f"Model dtype: {next(client.model.parameters()).dtype}")

t1 = time.time()
reply = client.chat("Reply with the single word OK.")
print(f"Reply ({time.time() - t1:.1f}s): {reply!r}")