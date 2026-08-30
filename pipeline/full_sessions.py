"""
Build full user histories and metadata for evaluation.

Inputs:
    - data/evid_sessions.json
    - data/filler_sessions.json
    - data/query_data.json

Outputs:
    - data/full_sessions.json
    - data/metadata.json
"""
from pathlib import Path
import json
RECBENCH_DIR = Path(__file__).resolve().parent.parent
from datetime import date, datetime, timedelta
from tqdm import tqdm


DATE_FMT = "%Y-%m-%d"

def parse_date(value: str) -> date:
    return datetime.strptime(value, DATE_FMT).date() # convert string to date


def format_date(value: date) -> str:
    return value.strftime(DATE_FMT) # convert date to string


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


def generate_filler_timestamps(user_sessions: list[dict], filler_count: int) -> list[str]:
    sorted_sessions = sorted(user_sessions, key=lambda x: x["timestamp"])
    base_dates = [parse_date(session["timestamp"]) for session in sorted_sessions]

    # create slots for filler timestamps
    slots: list[tuple[date, date]] = []

    first = base_dates[0]
    slots.append((parse_date("2025-05-01"), first))

    for left, right in zip(base_dates, base_dates[1:]):
        if left < right:
            slots.append((left, right))

    last = base_dates[-1]
    slots.append((last, parse_date("2026-05-01")))

    # how many fillers to assign to each slot
    assignments = [0] * len(slots)
    for i in range(filler_count):
        assignments[i % len(slots)] += 1

    filler_timestamps = []
    for (start, end), count in zip(slots, assignments):
        span_days = (end - start).days
        if span_days <= 1:
            candidate_days = [start for _ in range(count)]
        else:
            candidate_days = []
            for j in range(1, count + 1):
                offset = max(1, (span_days * j) // (count + 1)) # where to place the filler timestamp
                candidate = start + timedelta(days=offset)
                candidate_days.append(candidate)

        filler_timestamps.extend(format_date(day) for day in candidate_days)
    return filler_timestamps[:filler_count]


with open(RECBENCH_DIR / "data/evid_sessions.json", "r") as f:
    evid_sessions = json.load(f)

with open(RECBENCH_DIR / "data/filler_sessions.json", "r") as f:
    filler_sessions = json.load(f)

with open(RECBENCH_DIR / "data/query_data.json", "r") as f:
    query_data = json.load(f)

sessions = []
metadata = []
evid_by_user = {item["user_id"]: item for item in evid_sessions}
filler_by_user = {item["user_id"]: item for item in filler_sessions}

for query in tqdm(
    query_data,
    total=len(query_data),
    desc="Processing sessions...",
    leave=False,
):  # same user id
    evid = evid_by_user[query["user_id"]]
    filler = filler_by_user[query["user_id"]]
    user_sessions = []
    idx = 0

    for local_idx, (time, event) in enumerate(zip(evid["event_timestamps"], evid["event_turns"])):
        user_sessions.append(
            {
                "idx": idx, # original indices
                "local_idx": local_idx,
                "type": "event",
                "timestamp": time,
                "session": validate_turns(event),
            }
        )
        idx += 1

    for local_idx, (time, traj) in enumerate(zip(evid["trajectory_timestamps"], evid["trajectory_turns"])):
        user_sessions.append(
            {
                "idx": idx,
                "local_idx": local_idx,
                "type": "trajectory",
                "timestamp": time,
                "session": validate_turns(traj),
            }
        )
        idx += 1

    filler_turns = filler["filler_turns"]
    filler_timestamps = generate_filler_timestamps(user_sessions, len(filler_turns))

    for time, filler_turn in zip(filler_timestamps, filler_turns):
        user_sessions.append(
            {
                "idx": idx,
                "type": "filler",
                "timestamp": time,
                "session": validate_turns(filler_turn),
            }
        )
        idx += 1

    user_sessions.sort(key=lambda x: x["timestamp"])
    
    type1_timestamp = []
    type1_index = []
    type2_timestamp = []
    type2_index = []
    type3_timestamp = []
    type3_index = []
    type4_timestamp = []
    type4_index = []
    event_index_by_local = {}
    event_timestamp_by_local = {}
    trajectory_index_by_local = {}
    trajectory_timestamp_by_local = {}

    for index, session in enumerate(user_sessions):
        if session["type"] == "event":
            type1_timestamp.append(session["timestamp"])
            type1_index.append(index)
            event_index_by_local[session["local_idx"]] = index
            event_timestamp_by_local[session["local_idx"]] = session["timestamp"]
            session["idx"] = index # new indices
        elif session["type"] == "trajectory":
            type3_timestamp.append(session["timestamp"])
            type3_index.append(index)
            trajectory_index_by_local[session["local_idx"]] = index
            trajectory_timestamp_by_local[session["local_idx"]] = session["timestamp"]
            session["idx"] = index
        else: # fillers
            session["idx"] = index

    for local_idx in query["type2_evid"]:
        type2_timestamp.append(event_timestamp_by_local[local_idx])
        type2_index.append(event_index_by_local[local_idx])

    for local_idx in query["type4_evid"]:
        type4_timestamp.append(trajectory_timestamp_by_local[local_idx])
        type4_index.append(trajectory_index_by_local[local_idx])

    metadata.append({
        "user_id": query["user_id"],
        "persona": query["persona"],
        "query_type": "type1",
        "query": query["type1"],
        "query_timestamp": "2026-05-01",
        "evidence_session_timestamps": type1_timestamp,
        "evidence_session_index": type1_index,
        "summarized_evidence": query["events"],
    })
    metadata.append({
        "user_id": query["user_id"],
        "persona": query["persona"],
        "query_type": "type2",
        "query": query["type2"],
        "query_timestamp": "2026-05-01",
        "evidence_session_timestamps": type2_timestamp,
        "evidence_session_index": type2_index,
        "summarized_evidence": [query["events"][idx] for idx in query["type2_evid"]],
    })
    metadata.append({
        "user_id": query["user_id"],
        "persona": query["persona"],
        "query_type": "type3",
        "query": query["type3"],
        "query_timestamp": "2026-05-01",
        "evidence_session_timestamps": type3_timestamp,
        "evidence_session_index": type3_index,
        "summarized_evidence": query["trajectory"],
    })
    metadata.append({
        "user_id": query["user_id"],
        "persona": query["persona"],
        "query_type": "type4",
        "query": query["type4"],
        "query_timestamp": "2026-05-01",
        "evidence_session_timestamps": type4_timestamp,
        "evidence_session_index": type4_index,
        "summarized_evidence": [query["trajectory"][idx] for idx in query["type4_evid"]],
    })
            
    sessions.append(
        {
            "user_id": evid["user_id"],
            "persona": evid["persona"],
            "timestamps": [session["timestamp"] for session in user_sessions],
            "session_types": [session["type"] for session in user_sessions],
            "sessions": [session["session"] for session in user_sessions],
        }
    )

def validate_clean_sessions(all_users):
    errors = []
    for user in all_users:
        session_types = user.get("session_types", [])
        for index, session in enumerate(user["sessions"]):
            try:
                validate_turns(session)
            except ValueError as exc:
                session_type = session_types[index] if index < len(session_types) else None
                errors.append((user["user_id"], index, session_type, str(exc)))
    if errors:
        raise RuntimeError(
            f"Invalid sessions: count={len(errors)}, examples={errors[:5]}"
        )


validate_clean_sessions(sessions)

with open(RECBENCH_DIR / "data/full_sessions.json", "w", encoding="utf-8") as f:
    json.dump(sessions, f, indent=2, ensure_ascii=False)

with open(RECBENCH_DIR / "data/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
