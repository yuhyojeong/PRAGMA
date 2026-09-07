
"""
Generate evidence chat sessions for event and trajectory evidence.

Inputs:
    - data/query_data.json, with events, event_timestamps, trajectory, and trajectory_timestamps
    - data/evid_sessions.json, optional existing partial results for resume

Outputs:
    - data/evid_sessions.json, containing:
        - event_timestamps
        - event_turns
        - trajectory_timestamps
        - trajectory_turns
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
EVID_SESSIONS_PATH = RECBENCH_DIR / "data/evid_sessions.json"
MODEL = CONFIG["models"]["evidence_session"]
MAX_GENERATION_ATTEMPTS = CONFIG["generation"]["max_attempts"]
REQUEST_WORKERS = CONFIG["concurrency"]["evidence_requests"]

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


TURN_SCHEMA = {
    "name": "evidence_sessions",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "turns": {
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
        "required": ["turns"],
    },
}


def generate_event_turns(event: str, timestamp: str) -> list[dict]:
    system_prompt = f"""
Given the timestamp and event, generate multiple chat turns between the user and the assistant for each event. Strictly adhere to the following rules.

Reference metadata:
- Chat timestamp: {timestamp}
- Use this only as temporal grounding when determining the relative timing of events.
- Do not mention, copy, or paraphrase this timestamp in the output unless explicitly required.

Turns:
- The event SHOULD be naturally revealed in the middle of the turns.
- Assistant turns should NOT claim personal experiences or real-world actions.
- The assistant should NOT mention new information that was NOT provided by the user.
- SHOULD be lengthy, with at least 6 pairs of turns.
- MUST be written entirely in English.
- MUST begin with a user turn and end with an assistant turn.
- MUST alternate exactly: user, assistant, user, assistant, ...
- MUST contain an even number of turns.
- Every content field MUST be non-empty.
""".strip()
    return request_turns(system_prompt, f"Event: {event}", "event")


def generate_trajectory_turns(state: str, timestamp: str) -> list[dict]:
    system_prompt = f"""
Given the timestamp and the user's state, generate multiple chat turns between the user and the assistant for the state. Strictly adhere to the following rules.

Reference metadata:
- Chat timestamp: {timestamp}
- Use this only as temporal grounding when determining the relative timing of events.
- Do not mention, copy, or paraphrase this timestamp in the output unless explicitly required.

Turns:
- SHOULD be about the state.
- The state SHOULD be naturally revealed in the middle of the turns.
- Assistant turns should NOT claim personal experiences or real-world actions.
- The assistant should NOT mention new information that was NOT provided by the user.
- SHOULD be lengthy, with at least 6 pairs of turns.
- MUST be written entirely in English.
- MUST begin with a user turn and end with an assistant turn.
- MUST alternate exactly: user, assistant, user, assistant, ...
- MUST contain an even number of turns.
- Every content field MUST be non-empty.
""".strip()
    return request_turns(system_prompt, f"State: {state}", "trajectory")


def request_turns(system_prompt, user_prompt, label):
    last_error = None
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_schema", "json_schema": TURN_SCHEMA},
            )
            payload = json.loads(response.choices[0].message.content.strip())
            return validate_turns(payload["turns"])
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Failed to generate valid {label} turns after "
        f"{MAX_GENERATION_ATTEMPTS} attempts"
    ) from last_error


def main() -> None:
    query_data = load_json(QUERY_DATA_PATH, [])
    order_by_user = {item["user_id"]: i for i, item in enumerate(query_data)}
    results = load_json(EVID_SESSIONS_PATH, [])
    results_by_user = {item["user_id"]: item for item in results}

    for query_item in query_data:
        results_by_user.setdefault(
            query_item["user_id"],
            {"user_id": query_item["user_id"], "persona": query_item["persona"]},
        )

    def write_results() -> None:
        ordered = sorted(results_by_user.values(), key=lambda item: order_by_user[item["user_id"]])
        save_json(EVID_SESSIONS_PATH, ordered)

    jobs = []
    pending_by_user = {}
    updates_by_user = {}
    for query_item in query_data:
        user_id = query_item["user_id"]
        evid_item = results_by_user[user_id]
        user_jobs = []
        updates = {}

        events = query_item["events"]
        event_timestamps = query_item["event_timestamps"]
        if not valid_sessions(evid_item.get("event_turns"), len(events)):
            updates["event_timestamps"] = event_timestamps
            updates["event_turns"] = [None] * len(events)
            user_jobs.extend(
                ("event_turns", index, generate_event_turns, event, timestamp)
                for index, (event, timestamp) in enumerate(zip(events, event_timestamps))
            )

        trajectory = query_item["trajectory"]
        trajectory_timestamps = query_item["trajectory_timestamps"]
        if not valid_sessions(evid_item.get("trajectory_turns"), len(trajectory)):
            updates["trajectory_timestamps"] = trajectory_timestamps
            updates["trajectory_turns"] = [None] * len(trajectory)
            user_jobs.extend(
                ("trajectory_turns", index, generate_trajectory_turns, state, timestamp)
                for index, (state, timestamp) in enumerate(zip(trajectory, trajectory_timestamps))
            )

        if user_jobs:
            pending_by_user[user_id] = len(user_jobs)
            updates_by_user[user_id] = updates
            jobs.extend((user_id, *job) for job in user_jobs)

    with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as executor:
        futures = {
            executor.submit(fn, text, timestamp): (user_id, field, index)
            for user_id, field, index, fn, text, timestamp in jobs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating evidence sessions",
        ):
            user_id, field, index = futures[future]
            updates_by_user[user_id][field][index] = future.result()
            pending_by_user[user_id] -= 1
            if pending_by_user[user_id] == 0:
                results_by_user[user_id].update(updates_by_user[user_id])
                write_results()

    write_results()


if __name__ == "__main__":
    main()
