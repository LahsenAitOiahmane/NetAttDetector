import copy
import json
from pathlib import Path

import pytest

from netattdetector.io import sha256_file
from netattdetector.provenance import inspect_provenance, utc_time, validate_capture_plan


def test_headers_do_not_invent_capture_or_time_metadata(tmp_path):
    path = tmp_path / "nf-unsw-nb15-v2.csv"
    path.write_text("IPV4_SRC_ADDR,FLOW_DURATION_MILLISECONDS,Label\n192.0.2.1,10,0\n")
    result = inspect_provenance(
        path, "nf-unsw-nb15-v2", "configs/data/source_roles.json", tmp_path / "header.json"
    )
    assert result["candidate_metadata_columns"]["host"] == ["IPV4_SRC_ADDR"]
    assert result["candidate_metadata_columns"]["event_time"] == []
    assert not result["capture_time_split_ready"]
    assert result["sha256"] == sha256_file(path)


@pytest.fixture
def capture_plan(tmp_path):
    roles = {
        "development": {"role": "development", "lineage": ["dev"]},
        "reserved": {"role": "reserved_unacquired", "lineage": ["new"]},
    }
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(json.dumps({"sources": roles}))
    captures = []
    for i, split in enumerate(("train", "calibration", "validation", "test")):
        source = "reserved" if split == "test" else "development"
        path = tmp_path / f"{split}.csv"
        path.write_text(f"synthetic declared export {split}\n")
        captures.append(
            {
                "source_id": source,
                "capture_id": f"capture-{i}",
                "split": split,
                "path": str(path),
                "sha256": sha256_file(path),
                "start_utc": f"2026-01-01T0{i}:00:00Z",
                "end_utc": f"2026-01-01T0{i}:10:00Z",
                "host_groups": [f"device-{i}"],
                "benign_rows": 30,
                "attack_rows": 30,
            }
        )
    return {
        "schema_version": 1,
        "source_roles": str(roles_path),
        "purge_seconds": 300,
        "min_rows_per_class": 25,
        "require_host_disjoint": True,
        "captures": captures,
        "training_enabled": False,
    }


def save_validate(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return validate_capture_plan(path, tmp_path / "checked.json")


def test_valid_metadata_structure_does_not_authorize_training(tmp_path, capture_plan):
    result = save_validate(tmp_path, capture_plan)
    assert result["structure_valid"]
    assert not result["training_authorized"]


@pytest.mark.parametrize(
    "case", ["capture", "host", "time", "checksum", "role", "support", "lineage"]
)
def test_capture_plan_blocks_leakage_and_invalid_support(tmp_path, capture_plan, case):
    plan = copy.deepcopy(capture_plan)
    if case == "capture":
        plan["captures"][1]["capture_id"] = plan["captures"][0]["capture_id"]
    elif case == "host":
        plan["captures"][1]["host_groups"] = plan["captures"][0]["host_groups"]
    elif case == "time":
        plan["captures"][1]["start_utc"] = "2026-01-01T00:11:00Z"
    elif case == "checksum":
        plan["captures"][0]["sha256"] = "0" * 64
    elif case == "role":
        plan["captures"][3]["source_id"] = "development"
    elif case == "support":
        plan["captures"][2]["attack_rows"] = 0
    elif case == "lineage":
        roles_path = Path(plan["source_roles"])
        roles = json.loads(roles_path.read_text())
        roles["sources"]["reserved"]["lineage"] = ["dev"]
        roles_path.write_text(json.dumps(roles))
    result = save_validate(tmp_path, plan)
    assert not result["structure_valid"]
    assert result["blockers"]


def test_capture_times_require_timezone():
    with pytest.raises(ValueError, match="UTC offset"):
        utc_time("2026-01-01T00:00:00")
