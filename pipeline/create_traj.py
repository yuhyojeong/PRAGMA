"""
Generate trajectory-grounded evidence states and timestamps.

Inputs:
    - data/query_data.json

Outputs:
    - data/query_data.json, updated with:
        - trajectory
        - trajectory_timestamps
"""
from pathlib import Path
from openai import OpenAI
import json

RECBENCH_DIR = Path(__file__).resolve().parent.parent
from tqdm import tqdm

from config import CONFIG

client = OpenAI()
MODEL = CONFIG["models"]["trajectory_generation"]

with open(RECBENCH_DIR / "data/query_data.json", "r") as f:
    data = json.load(f)

system_prompt = f"""
Given the persona and axes, create a trajectory for the user. Strictly adhere to the following rules.

Trajectory:
- SHOULD be sequential with 4-8 states.
- there SHOULD be states that are relevant to only ONE axis
- all states SHOULD be relevant to either or both axes
- SHOULD mention the user's initial preference
- SHOULD include the user's drift in preference or events
- there SHOULD be earlier states that remains relevant for the rest of the trajectory
- the last state should NOT be equivalent to the user's current state
- Trajectories SHOULD vary in pattern (e.g., gradual shift, oscillation, plateau, partial reversal)
- Only answer with the trajectory, without any labels on patterns or axes
- Avoid overly clean or perfectly structured progression
- SHOULD be concise

Timestamps:
- SHOULD be between 2025-05-01 and 2026-04-30
- SHOULD be sorted in temporal order
""".strip()

fewshot1_user = """
Persona: A regular runner
Axes: ["Injury vs Marathon", "Latte vs Lactos Intolerance"]
""".strip()

fewshot1_assistant = """
{
    "trajectory": [
        {
            "state": "The user is preparing for a marathon.",
            "timestamp": "2025-06-01"
        },
        {
            "state": "The user is injured.",
            "timestamp": "2025-07-15"
        },
        {
            "state": "The user likes drinking coffee before running.",
            "timestamp": "2025-08-20"
        },
        {
            "state": "The user finds out that he is lactose intolerant.",
            "timestamp": "2025-10-05"
        }
    ]
}
""".strip()

fewshot2_user = """
Persona: A professional singer who likes to read in her sparetime
Axes: ["Romance vs Mystery novels", "Song Covers vs Song Writing"]
""".strip()
fewshot2_assistant = """
{
    "trajectory": [
        {
            "state": "The user likes romance novels.",
            "timestamp": "2025-05-20"
        },
        {
            "state": "The user prefers filling her concerts with song covers.",
            "timestamp": "2025-08-30"
        },
        {
            "state": "The user is not fond of romance novels anymore.",
            "timestamp": "2025-11-10"
        },
        {
            "state": "For the upcoming concert, the user wants to write a song about the novels she likes.",
            "timestamp": "2026-01-15"
        },
        {
            "state": "The user is interested in mystery novels now.",
            "timestamp": "2026-03-05"
        }
    ]
}
""".strip()

schema = {
    "name": "trajectory_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "trajectory": {
                "type": "array",
                "minItems": 4,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "state": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date"}
                    },
                    "required": ["state", "timestamp"]
                }
            }
        },
        "required": ["trajectory"]
    }
}
remaining = [item for item in data if "trajectory" not in item]

for item in tqdm(remaining):
    persona = item["persona"]
    axes = item["axes"]
    traj = []
    traj_timestamps = []
    user_prompt = f"""
    Persona: {persona}
    Axes: {axes}
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
        response_format = {"type": "json_schema", "json_schema": schema}
    )
    response = response.choices[0].message.content.strip()
    response = json.loads(response)
    trajectory = response["trajectory"]
    for t in trajectory:
        traj.append(t["state"])
        traj_timestamps.append(t["timestamp"])
    item["trajectory"] = traj
    item["trajectory_timestamps"] = traj_timestamps

with open(RECBENCH_DIR / "data/query_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
