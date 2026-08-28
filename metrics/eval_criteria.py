"""
Generate per-query alignment and grounding evaluation rubrics.

Inputs:
    - data/metadata.json
    - data/query_data.json

Outputs:
    - data/response_metrics.json, containing:
        - query metadata
        - evidence summaries
        - alignment_metric
        - grounding_metric
"""
from pathlib import Path
import json
RECBENCH_DIR = Path(__file__).resolve().parent.parent
from tqdm import tqdm

with open(RECBENCH_DIR / "data/metadata.json", "r") as f:
    metadata = json.load(f)

with open(RECBENCH_DIR / "data/query_data.json", "r") as f:
    query_data = json.load(f)

qd_by_user = {qd["user_id"]: qd for qd in query_data}

results = []

for meta in tqdm(metadata, leave=False):
    query_type = meta["query_type"]
    events = meta["summarized_evidence"]
    axes = qd_by_user[meta["user_id"]]["axes"]
    
    # main metric (alignment)
    if query_type == "type1":
        align = f"""Does the response recommend a next step that meaningfully builds on the user’s prior attempts, rather than simply repeating past actions or suggesting something unrelated?
Output 1 if yes, 0 otherwise.
Only output the integer.
History: {events}
""".strip()
        ground = f"""Does the response explicitly use the user’s prior attempts as evidence for why this recommendation fits, rather than giving only generic advice?
Output 1 if yes, 0 otherwise.
Only output the integer.
History: {events}
""".strip()
    elif query_type == "type2":
        align = f"""Does the response identify the event-level inconsistency between the user's query and their prior experience?
Output 1 if yes, 0 otherwise.
Only output the integer.
History: {events}
""".strip()
        ground = f"""How many of the following history items does the response explicitly refer to?
Answer with the coverage score computed by (# referred items) / (# total history items).
Output ONLY the final score as a float between 0 and 1, rounded to 2 decimal places.
History: {events}
""".strip()
    elif query_type == "type3":
        align = f"""Does the response recommend a self-positioning strategy that is meaningfully aligned with the user’s trajectory across both axes, and grounds this positioning in specific past experiences, including how those experiences should be emphasized, downplayed, or omitted?
Output 1 if yes, 0 otherwise.
Only output the integer.
Progression: {events}
""".strip()
        ground =  f"""Does the response explicitly justify that recommendation using the user’s prior progression across both axes, rather than relying on generic advice or surface-level similarity?
Output 1 if yes, 0 otherwise.
Only output the integer.
Progression: {events}
""".strip()
    else: # type 4
        align = f"""Does the response identify the trajectory-level inconsistency between the user's intended decision and their prior trajectory?
Output 1 if yes, 0 otherwise.
Only output the integer.
History: {events}
""".strip()
        ground = f"""How many of the following history items does the response explicitly refer to?
Answer with the coverage score computed by (# referred items) / (# total history items).
Output ONLY the final score as a float between 0 and 1, rounded to 2 decimal places.
History: {events}
""".strip()
    
    results.append({
        "user_id": meta["user_id"],
        "query_type": query_type,
        "query": meta["query"],
        "evidence": events,
        "alignment_metric": align,
        "grounding_metric": ground
    })

with open(RECBENCH_DIR / "data/response_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
