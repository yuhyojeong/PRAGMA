"""Generate trajectory-based benchmark queries directly in data/query_data.json.

Inputs:
    - data/query_data.json, with rewritten axes, trajectory, and timestamps

Outputs:
    - data/query_data.json, updated with type3, type4, and type4_evid
"""
from pathlib import Path
import json
from openai import OpenAI
from tqdm import tqdm

from config import CONFIG


RECBENCH_DIR = Path(__file__).resolve().parent.parent
QUERY_DATA_PATH = RECBENCH_DIR / "data/query_data.json"
MODEL = CONFIG["models"]["trajectory_query"]

client = OpenAI()


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


TYPE3_SYSTEM_PROMPT = """
Given the persona, rewritten axes, and user trajectory, create a user query. Only output the query. Strictly adhere to the following rules.

Reference metadata:
- Query timestamp: 2026-05-01
- Use this only as temporal grounding when determining the relative timing of events.
- Do not mention, copy, or paraphrase this timestamp in the output unless explicitly required.

Query:
- Create a realistic self-positioning query about how the user should present themselves (e.g., CV, bio, introduction, application, etc.).
- The query SHOULD use a context appropriate to the user's trajectory and persona.
- The user SHOULD ask how to position themselves along the rewritten axes by incorporating their past experience.
- The user SHOULD ask how to use their past experience to support this positioning, including what to emphasize, downplay, or leave out so the overall positioning is clear and coherent.
- SHOULD include only the rewritten axes terms, without implying the user’s progression along them.
- should NOT state any of the past experiences.
- should NOT state any of the user's trajectory.
- should NOT state the user's persona in the query.
- SHOULD remain underspecified, without explicitly stating progression or trade-offs.
- Avoid phrasing that reduces the task to generic writing or stylistic advice.
- Keep the query concise, natural, and realistic.

Output Format:
Query: <query>

Input:
""".strip()


TYPE4_SYSTEM_PROMPT = """
Given the user's trajectory, select multiple states, then create a user query. Strictly adhere to the following rules.

Query:
- SHOULD present a concrete situation and a decision the user is considering
- The decision SHOULD be reasonable and commonly justifiable without additional context
- The decision SHOULD prioritize or increase a specific activity, commitment, or direction
- Any competing priorities SHOULD remain implicit and not be explicitly stated
- The situation SHOULD appear straightforward and not obviously problematic or risky
- The decision SHOULD be subtly misaligned with the user's trajectory, but this misalignment SHOULD NOT be detectable without considering past progression
- The misalignment SHOULD emerge only when considering changes across BOTH axes
- SHOULD NOT explicitly mention trade-offs, conflicts, or competing priorities
- SHOULD NOT reference or summarize the user's trajectory or past changes
- SHOULD NOT use language that signals doubt, hesitation, or obvious imbalance
- The assistant SHOULD be able to disagree, but only by leveraging implicit signals from the user's trajectory
- A correct response SHOULD require reasoning over BOTH axes; focusing on only one axis SHOULD be insufficient
- SHOULD be concise and natural

Selected States:
- identify all the states in the trajectory that could be used as counter-evidence to the user's decision
- output the indices of the selected states in a list
""".strip()

TYPE4_FEWSHOT_1_USER = """
Trajectory: ["The user is preparing for a marathon.", "The user is injured.", "The user likes drinking coffee before running.", "The user finds out that he is lactose intolerant."]
""".strip()

TYPE4_FEWSHOT_1_ASSISTANT = """
{
    "query": "I really want to keep up with my marathon training. So I'm planning to drink my usual coffee before hard training tomorrow. What do you think?",
    "selected_traj_idx": [1, 3]
}
""".strip()

TYPE4_FEWSHOT_2_USER = """
Trajectory: ["The user likes romance novels.", "The user is preparing for her concert.", "The user wants to write a song about the novels she has read.", "The user is sick of romance novels now.", "The user is interested in mystery novels now."]
""".strip()

TYPE4_FEWSHOT_2_ASSISTANT = """
{
    "query": "I’m working on a new song for my upcoming concert, and I’m thinking of sticking with the same themes I’ve always used since they’ve worked well before. What do you think?",
    "selected_traj_idx": [3, 4]
}
""".strip()

QUERY_SCHEMA = {
    "name": "query_output",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


TYPE4_SCHEMA = {
    "name": "type4_query_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
            "selected_traj_idx": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "integer"},
            },
        },
        "required": ["query", "selected_traj_idx"],
    },
}


def generate_type3(item: dict) -> str:
    user_prompt = f"""
    Persona: {item['persona']}
    Rewritten Axes: {item['rewritten axes']}
    Trajectory: {item['trajectory']}
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TYPE3_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": QUERY_SCHEMA},
    )
    return json.loads(response.choices[0].message.content.strip())["query"]


def generate_type4(item: dict) -> tuple[str, list[int]]:
    user_prompt = f"""
    Trajectory: {item['trajectory']}
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TYPE4_SYSTEM_PROMPT},
            {"role": "user", "content": TYPE4_FEWSHOT_1_USER},
            {"role": "assistant", "content": TYPE4_FEWSHOT_1_ASSISTANT},
            {"role": "user", "content": TYPE4_FEWSHOT_2_USER},
            {"role": "assistant", "content": TYPE4_FEWSHOT_2_ASSISTANT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": TYPE4_SCHEMA},
    )
    parsed = json.loads(response.choices[0].message.content.strip())
    return parsed["query"], parsed["selected_traj_idx"]


def main() -> None:
    query_data = load_json(QUERY_DATA_PATH)

    for item in tqdm(
        [item for item in query_data if "type3" not in item],
        desc="Generating type3 queries",
    ):
        item["type3"] = generate_type3(item)
        save_json(QUERY_DATA_PATH, query_data)

    for item in tqdm(
        [item for item in query_data if "type4" not in item],
        desc="Generating type4 queries",
    ):
        query, selected_traj_idx = generate_type4(item)
        item["type4"] = query
        item["type4_evid"] = selected_traj_idx
        save_json(QUERY_DATA_PATH, query_data)

    save_json(QUERY_DATA_PATH, query_data)


if __name__ == "__main__":
    main()
