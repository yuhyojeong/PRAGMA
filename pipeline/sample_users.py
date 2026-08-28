"""
Sample users from Privasis-Zero and extract lightweight profile fields.

Inputs:
    - Hugging Face dataset: nvidia/Privasis-Zero, corpus split=train

Outputs:
    - data/privasis.json
"""
from pathlib import Path
import json
RECBENCH_DIR = Path(__file__).resolve().parent.parent
from datasets import load_dataset
from tqdm import tqdm

from config import CONFIG

privasis = load_dataset("nvidia/Privasis-Zero", "corpus", split="train")

examples = privasis.shuffle(seed=CONFIG["sampling"]["seed"]).select(
    range(CONFIG["sampling"]["num_users"])
)

results = []

# extract persona, profiles
for user in tqdm(examples):
    profile = json.loads(user["profile"])
    results.append({
        "id": user["id"],
        "profiles": {
            "age": profile["profile"]["age"],
            "income_class": profile["profile"]["income_class"],
            "native_language": profile["profile"]["native_language"],
            "citizenship": profile["profile"]["citizenship"]
        },
        "event_list": profile["event_list"],
    })

with open(RECBENCH_DIR / "data/privasis.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False,indent=2)
