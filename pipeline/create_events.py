"""
Generate event-grounded evidence items and timestamps for event query types.

Inputs:
    - data/query_data.json

Outputs:
    - data/query_data.json, updated with:
        - events
        - event_timestamps
"""
from pathlib import Path
from openai import OpenAI
import json
RECBENCH_DIR = Path(__file__).resolve().parent.parent
from tqdm import tqdm

from config import CONFIG

client = OpenAI()
MODEL = CONFIG["models"]["event_generation"]

with open(RECBENCH_DIR / "data/query_data.json", "r") as f:
    data = json.load(f)

system_prompt = f"""
Given the persona, axes, and topic, create a few specific past events for the user. Strictly adhere to the following rules.

Events:
- SHOULD focus on the topic
- MAY describe actions, observations, or feelings
- should NOT be relevant to the axes
- should NOT contradict the persona
- SHOULD be short and concise
- SHOULD include temporal or spacial words
- start with "The user"

Timestamps:
-SHOULD be between 2025-05-01 and 2026-04-30
-SHOULD be sorted in temporal order
""".strip()

fewshot1_user = """
Persona: A regular runner.
Axes: ["Marathon", "Hydration"]
Topic: Food
""".strip()

fewshot1_assistant = """
{
    "events": [
        {
            "event": "The user ate steak in Texas.",
            "timestamp": "2025-06-15"
        },
        {
            "event": "The user ate ramen in Tokyo.",
            "timestamp": "2025-09-10"
        },
        {
            "event": "The user ate pasta in Italy.",
            "timestamp": "2025-12-05"
        }
    ]
}
""".strip()

fewshot2_user = """
Persona: A professional singer who likes to read in her sparetime.
Axes: ["Reading", "Singing"]
Topic: Instruments
""".strip()

fewshot2_assistant = """
{
    "events": [
        {
            "event": "The user played flute in middle school.",
            "timestamp": "2025-05-20"
        },
        {
            "event": "The user played oboe in high school.",
            "timestamp": "2025-08-30"
        }
    ]
}
""".strip()

schema = {
    "name": "event_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "events": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event": { "type": "string" },
                        "timestamp": { "type": "string", "format": "date" }
                    },
                    "required": ["event", "timestamp"]
                }
            }
        },
        "required": ["events"]
    }
}

remaining = [item for item in data if "events" not in item]

for item in tqdm(remaining):
    persona = item["persona"]
    axes = item["axes"]
    topic = item["topic"]
    event = []
    timestamp = []
    user_prompt = f"""
    Persona: {persona}
    Axes: {axes}
    Topic: {topic}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fewshot1_user},
        {"role": "assistant", "content": fewshot1_assistant},
        {"role": "user", "content": fewshot2_user},
        {"role": "assistant", "content": fewshot2_assistant},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": schema
        }
    )
    response = response.choices[0].message.content.strip()
    response = json.loads(response)
    events = response["events"]
    for e in events:
        event.append(e["event"])
        timestamp.append(e["timestamp"])
    item["events"] = event
    item["event_timestamps"] = timestamp


with open(RECBENCH_DIR / "data/query_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
