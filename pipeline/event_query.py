"""
Generate event-based benchmark queries.

Inputs:
    - data/query_data.json, with events and event_timestamps

Outputs:
    - data/query_data.json, updated with:
        - type1
        - type2
        - type2_evid
        - type2_rewritten, for rewritten long corrective queries
"""
from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm

from config import CONFIG


RECBENCH_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = RECBENCH_DIR / "data/query_data.json"
TYPE_QUERY_MODEL = CONFIG["models"]["event_query"]
REWRITE_MODEL = CONFIG["models"]["query_rewrite"]

client = OpenAI()


def load_data() -> list[dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: list[dict]) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


TYPE1_SYSTEM_PROMPT = """
Given the topic and past user events, summarize the events then create a user query. Strictly adhere to the following rules.

Summary:
-from the user's perspective
-use "I"

Reference metadata:
- Query timestamp: 2026-05-01
- Use this only as temporal grounding when determining the relative timing of events.
- Do not mention, copy, or paraphrase this timestamp in the output unless explicitly required.

Query:
- should NOT give hints about the events
- should NOT restate the events and remain implicit
- SHOULD include the topic naturally in the sentence
- SHOULD ask for recommendation or advice on the topic
- SHOULD be implicitly grounded in the user's past experiences (i.e., the answer should depend on them, even if they are not mentioned)
- should NOT include meta phrases such as “personalized”, “based on my experiences”, or "given my ~"
- SHOULD remain underspecified, requiring the assistant to infer relevant context from prior experiences
- SHOULD sound like a natural, realistic user question
- SHOULD be concise

Examples:
Topic: Food
Events: ["The user ate steak in Texas.", "The user ate ramen in Tokyo.", "The user ate pasta in Italy."]
Summary: I had steak in Texas, ramen in Tokyo, and pasta in Italy.
Query: I liked the food I had while traveling. Are there any recommendations for similar food I could try?

Topic: Instruments
Events: ["The user played flute in middle school.", "The user played oboe in high school."]
Summary: I played flute in middle school and oboe in high school.
Query: I'd like to learn a new instrument. Could you recommend one that might be easier for me to pick up than the instruments I played when I was younger?

Output Format:
Summary: <summary>
Query: <query>

Input:
""".strip()

TYPE2_SYSTEM_PROMPT = """
Given the past user events, select multiple events then generate a user query. Strictly adhere to the following rules.

Reference metadata:
- Query timestamp: 2026-05-01
- Use this only as temporal grounding when determining the relative timing of events.
- Do not mention, copy, or paraphrase this timestamp in the output unless explicitly required.

Query:
- SHOULD construct a plausible but imperfect recollection by blending multiple past events
- SHOULD be CONFIDENTLY incorrect about the selected events
- SHOULD remain grounded in the original events (no completely new activities)
- should NOT restate the events and remain implicit (use different words from the events)
- SHOULD ask for recommendation or advice based on the incorrect understanding of the past events
- SHOULD be concise

Selected Events:
- identify all the mixed up events that were used in the query
- output the indices of the selected events in a list
""".strip()

TYPE2_FEWSHOT_1_USER = """
Events: ["The user ate steak in Texas.", "The user ate ramen in Tokyo.", "The user ate pasta in Italy."]
""".strip()

TYPE2_FEWSHOT_1_ASSISTANT = """
{
    "query": "Remember the pasta place I went to in Japan? Can you recommend similar places in that area?",
    "selected_events_idx": [1, 2]
}
""".strip()

TYPE2_FEWSHOT_2_USER = """
Events: ["The user played flute in middle schoool", "The user played oboe in high school."]
""".strip()

TYPE2_FEWSHOT_2_ASSISTANT = """
{
    "query": "I remember playing oboe in 7th grade. Do you think I should pick it up again?",
    "selected_events_idx": [0, 1]
}
""".strip()

TYPE2_SCHEMA = {
    "name": "type2_query_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
            "selected_events_idx": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "integer"},
            },
        },
        "required": ["query", "selected_events_idx"],
    },
}

TYPE2_REWRITE_SYSTEM_PROMPT = """
Rewrite the following query for higher readability.
Do NOT change the meaning of the query.

Output Format:
Rewritten Query: <rewritten query>
""".strip()


def generate_type1(item: dict) -> str:
    user_prompt = f"""
    Topic: {item['topic']}
    Events: {item['events']}
    """
    response = client.chat.completions.create(
        model=TYPE_QUERY_MODEL,
        messages=[
            {"role": "system", "content": TYPE1_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    return text.split("Query: ", 1)[1].strip()


def generate_type2(item: dict) -> tuple[str, list[int]]:
    user_prompt = f"""
    Events: {item['events']}
    """
    response = client.chat.completions.create(
        model=TYPE_QUERY_MODEL,
        messages=[
            {"role": "system", "content": TYPE2_SYSTEM_PROMPT},
            {"role": "user", "content": TYPE2_FEWSHOT_1_USER},
            {"role": "assistant", "content": TYPE2_FEWSHOT_1_ASSISTANT},
            {"role": "user", "content": TYPE2_FEWSHOT_2_USER},
            {"role": "assistant", "content": TYPE2_FEWSHOT_2_ASSISTANT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": TYPE2_SCHEMA},
    )
    parsed = json.loads(response.choices[0].message.content.strip())
    return parsed["query"], parsed["selected_events_idx"]


def rewrite_type2_query(query: str) -> str:
    response = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": TYPE2_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
    )
    text = response.choices[0].message.content.strip()
    return text.split("Rewritten Query: ", 1)[1].strip()


def main() -> None:
    data = load_data()

    for item in tqdm([item for item in data if "type1" not in item], desc="Generating type1 queries"):
        item["type1"] = generate_type1(item)
        save_data(data)

    for item in tqdm([item for item in data if "type2" not in item], desc="Generating type2 queries"):
        query, selected_events_idx = generate_type2(item)
        item["type2"] = query
        item["type2_evid"] = selected_events_idx
        save_data(data)

    for item in tqdm(data, desc="Rewriting long type2 queries", leave=False):
        if len(item.get("type2_evid", [])) >= 4 and not item.get("type2_rewritten"):
            item["type2"] = rewrite_type2_query(item["type2"])
            item["type2_rewritten"] = True
            save_data(data)

    save_data(data)


if __name__ == "__main__":
    main()
