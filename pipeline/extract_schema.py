"""
Generate user personas, latent axes, topics, and rewritten axis labels.

Inputs:
    - data/privasis.json

Outputs:
    - data/query_data.json
"""
import json
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from config import CONFIG


RECBENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = RECBENCH_DIR / "data"
MODEL = CONFIG["models"]["schema"]

client = OpenAI()


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


PROFILE_SYSTEM_PROMPT = """
Given the user's profile and event list, extract the following:

1. Persona
- Create a realistic persona for the user.
- One concise sentence, not overly detailed.
- Grounded in the profile and events, but avoid over-specific or speculative details.

2. Axes
- Create exactly two latent axes that can be used to generate diverse future events.
- Each axis should represent a meaningful continuum (X <-> Y) with interpretable directionality.
- The two ends do not need to be strict opposites, but should form a coherent spectrum.
- The axes should be distinct and not overlapping in meaning.
- Prefer general, reusable dimensions over event-specific descriptions.
- Avoid extreme or overly narrow wording; use smooth, balanced spectra.
- The axes should reflect properties that can vary or evolve over time.
- Keep each axis short and broad.
- Avoid axes where a generic LLM prior strongly favors one side as more prudent, safer, or more socially acceptable.
- Do not use primarily "risk vs safety," "cautious vs bold," or similar normative framings that can be resolved without personalization.
- Prefer axes where neither side is inherently better without knowing the user's trajectory.

3. Topic
- Create one broad topic related to the persona.
- The topic should be suitable for generating multiple types of events.
- The topic must be completely irrelevant to both axes.

Constraints:
- Axes must be dimensions, not topics, events, or categories.
- Do not simply restate the given event as an axis.
- Topic must not overlap semantically with either axis.
- Favor generality and reusability over fitting closely to the given event.
- Keep all outputs concise.
""".strip()

PROFILE_SCHEMA = {
    "name": "query_data",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "persona": {"type": "string"},
            "axes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
            "topic": {"type": "string"},
        },
        "required": ["persona", "axes", "topic"],
    },
}

AXIS_REWRITE_SYSTEM_PROMPT = """
Given the user's axes, rewrite each axis into a concise conceptual label without arrows.

Example:
axes: ["Institutional loyalty <-> Independent reinvention", "Quiet craft mastery <-> Visible public authorship"]
rewritten axes: ["ownership", "work expression"]
""".strip()

AXIS_REWRITE_SCHEMA = {
    "name": "query_data_axis_rewrite",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rewritten axes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["rewritten axes"],
    },
}


def generate_profile_fields(profile: dict) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(profile, ensure_ascii=False)},
        ],
        response_format={"type": "json_schema", "json_schema": PROFILE_SCHEMA},
    )
    return json.loads(response.choices[0].message.content.strip())


def rewrite_axes(axes: list[str]) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": AXIS_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Axes: {axes}"},
        ],
        response_format={"type": "json_schema", "json_schema": AXIS_REWRITE_SCHEMA},
    )
    parsed = json.loads(response.choices[0].message.content.strip())
    return parsed["rewritten axes"]


def build_query_data() -> list[dict]:
    users = load_json(DATA_DIR / "privasis.json", [])
    query_data = load_json(DATA_DIR / "query_data.json", [])
    done_ids = {entry["user_id"] for entry in query_data}
    remaining = [user for user in users if user["id"] not in done_ids]

    for profile in tqdm(remaining, desc="Generating personas/axes/topics"):
        generated = generate_profile_fields(profile)
        query_data.append(
            {
                "user_id": profile["id"],
                "persona": generated["persona"],
                "topic": generated["topic"],
                "axes": generated["axes"],
            }
        )
        save_json(DATA_DIR / "query_data.json", query_data)

    save_json(DATA_DIR / "query_data.json", query_data)
    return query_data


def build_rewritten_axes(query_data: list[dict]) -> None:
    for entry in tqdm(
        [item for item in query_data if "rewritten axes" not in item],
        desc="Rewriting axes",
    ):
        entry["rewritten axes"] = rewrite_axes(entry["axes"])
        save_json(DATA_DIR / "query_data.json", query_data)

    save_json(DATA_DIR / "query_data.json", query_data)


def main() -> None:
    query_data = build_query_data()
    build_rewritten_axes(query_data)


if __name__ == "__main__":
    main()
