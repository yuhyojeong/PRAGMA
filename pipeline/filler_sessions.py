"""
Generate filler topics and filler chat sessions.

Inputs:
    - data/query_data.json
    - data/filler_sessions.json, optional existing partial results for resume

Outputs:
    - data/filler_sessions.json, containing:
        - filler_topics
        - filler_turns
"""
from pathlib import Path

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

from config import CONFIG


RECBENCH_DIR = Path(__file__).resolve().parent.parent
QUERY_DATA_PATH = RECBENCH_DIR / "data/query_data.json"
FILLER_SESSIONS_PATH = RECBENCH_DIR / "data/filler_sessions.json"
TOPIC_MODEL = CONFIG["models"]["filler_topic"]
TURN_MODEL = CONFIG["models"]["filler_session"]
MAX_GENERATION_ATTEMPTS = CONFIG["generation"]["max_attempts"]
REQUEST_WORKERS = CONFIG["concurrency"]["filler_requests"]

thread_local = threading.local()


def validate_turns(turns):
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns must be a non-empty list")
    if len(turns) % 2:
        raise ValueError(f"turn count must be even, got {len(turns)}")
    cleaned = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"turn {index} must be an object")
        expected_role = "user" if index % 2 == 0 else "assistant"
        if turn.get("role") != expected_role:
            raise ValueError(
                f"turn {index}: expected {expected_role!r}, got {turn.get('role')!r}"
            )
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"turn {index}: content must be a non-empty string")
        cleaned.append({"role": expected_role, "content": content})
    return cleaned


def valid_sessions(value, expected_count):
    if not isinstance(value, list) or len(value) != expected_count:
        return False
    try:
        for session in value:
            validate_turns(session)
    except ValueError:
        return False
    return True


def get_client() -> OpenAI:
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI()
    return thread_local.client


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


TOPIC_SYSTEM_PROMPT = """
Given the persona, axes, and topic, create 30 other topics. Strictly adhere to the following rules.

Topics:
- should NOT be contradictory to the persona
- SHOULD be completely irrelevant to both the axes and the topic
- ONLY output the topics without numbers or any other text
- SHOULD be concise with only few words
- MUST be written entirely in English
""".strip()

TOPIC_SCHEMA = {
    "name": "filler_topics",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "filler_topics": {
                "type": "array",
                "minItems": 30,
                "maxItems": 30,
                "items": {"type": "string"},
                "examples": ["home gardening", "stargazing"],
            }
        },
        "required": ["filler_topics"],
    },
}

TURN_SYSTEM_PROMPT = """
Given the topic, generate multiple chat turns between the user and the assistant for each topic. Strictly adhere to the following rules.

Turns:
- Assistant turns should NOT claim personal experiences or real-world actions.
- SHOULD focus on the topic.
- SHOULD be lengthy, with at least 6 pairs of turns.
- MUST be written entirely in English.
- MUST begin with a user turn and end with an assistant turn.
- MUST alternate exactly: user, assistant, user, assistant, ...
- MUST contain an even number of turns.
- Every content field MUST be non-empty.
""".strip()

TURN_SCHEMA = {
    "name": "filler_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "filler_turns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": {"type": "string", "enum": ["user", "assistant"]},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            }
        },
        "required": ["filler_turns"],
    },
}


def generate_filler_topics(query_item: dict) -> list[str]:
    user_prompt = f"""
    Persona: {query_item['persona']}
    Axes: {query_item['axes']}
    Topic: {query_item['topic']}
    """
    response = get_client().chat.completions.create(
        model=TOPIC_MODEL,
        messages=[
            {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": TOPIC_SCHEMA},
    )
    return json.loads(response.choices[0].message.content.strip())["filler_topics"]


def generate_filler_turns(topic: str) -> list[dict]:
    last_error = None
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            response = get_client().chat.completions.create(
                model=TURN_MODEL,
                messages=[
                    {"role": "system", "content": TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Topic: {topic}"},
                ],
                response_format={"type": "json_schema", "json_schema": TURN_SCHEMA},
            )
            payload = json.loads(response.choices[0].message.content.strip())
            return validate_turns(payload["filler_turns"])
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to generate valid filler turns after "
        f"{MAX_GENERATION_ATTEMPTS} attempts"
    ) from last_error


def main() -> None:
    query_data = load_json(QUERY_DATA_PATH, [])
    order_by_user = {item["user_id"]: i for i, item in enumerate(query_data)}
    results = load_json(FILLER_SESSIONS_PATH, [])
    results_by_user = {item["user_id"]: item for item in results}

    for query_item in query_data:
        results_by_user.setdefault(
            query_item["user_id"],
            {"user_id": query_item["user_id"], "persona": query_item["persona"]},
        )

    def write_results() -> None:
        ordered = sorted(results_by_user.values(), key=lambda item: order_by_user[item["user_id"]])
        save_json(FILLER_SESSIONS_PATH, ordered)

    topic_jobs = [
        query_item
        for query_item in query_data
        if "filler_topics" not in results_by_user[query_item["user_id"]]
    ]
    with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as executor:
        futures = {
            executor.submit(generate_filler_topics, query_item): query_item["user_id"]
            for query_item in topic_jobs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating filler topics",
        ):
            user_id = futures[future]
            results_by_user[user_id]["filler_topics"] = future.result()
            write_results()

    jobs = []
    pending_by_user = {}
    turns_by_user = {}
    for query_item in query_data:
        user_id = query_item["user_id"]
        filler_item = results_by_user[user_id]
        topics = filler_item["filler_topics"]
        if valid_sessions(filler_item.get("filler_turns"), len(topics)):
            continue
        turns_by_user[user_id] = [None] * len(topics)
        pending_by_user[user_id] = len(topics)
        jobs.extend((user_id, index, topic) for index, topic in enumerate(topics))

    with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as executor:
        futures = {
            executor.submit(generate_filler_turns, topic): (user_id, index)
            for user_id, index, topic in jobs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating filler sessions",
        ):
            user_id, index = futures[future]
            turns_by_user[user_id][index] = future.result()
            pending_by_user[user_id] -= 1
            if pending_by_user[user_id] == 0:
                results_by_user[user_id]["filler_turns"] = turns_by_user[user_id]
                write_results()

    write_results()


if __name__ == "__main__":
    main()
