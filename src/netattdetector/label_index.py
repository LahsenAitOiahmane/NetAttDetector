"""Disk-backed interval candidates with exact nanosecond comparisons and bounded cache."""

import json
import sqlite3
from decimal import Decimal
from pathlib import Path


def nanoseconds(milliseconds):
    value = milliseconds * 1_000_000
    integer = int(value)
    if value != integer or not 0 <= integer < 2**63:
        raise ValueError("Label timestamps must fit exact nonnegative int64 nanoseconds.")
    return integer


class SQLiteLabelIndex:
    """Private local index; callers close it before removing its temporary file."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA page_size=4096")
        self.connection.execute("PRAGMA cache_size=-8192")  # 8 MiB, independent of row count.
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA max_page_count=131072")  # 512 MiB at 4096-byte pages.
        self.connection.execute(
            "CREATE TABLE labels (uid TEXT PRIMARY KEY, tuple_key TEXT, start_ns INTEGER, end_ns INTEGER, label INTEGER, detail TEXT)"
        )

    def add(self, key, row):
        # Unsupported protocol UIDs also enter the table to enforce capture-wide uniqueness.
        try:
            self.connection.execute(
                "INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["uid"],
                    json.dumps(key, separators=(",", ":")) if key is not None else None,
                    nanoseconds(row["start"]),
                    nanoseconds(row["end"]) if row["end"] is not None else None,
                    row["label"],
                    row["detail"],
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate Zeek UID within a capture.") from error

    def finish(self):
        self.connection.execute("CREATE INDEX tuple_time ON labels(tuple_key, start_ns)")
        self.connection.commit()

    def overlapping(self, key, start, end, slack):
        # At most two candidates are needed: two already make linkage ambiguous. In
        # particular, an unknown end remains a blocker for every later same-tuple flow.
        lower = max(Decimal(0), start - slack)
        rows = self.connection.execute(
            "SELECT uid, start_ns, end_ns, label, detail FROM labels WHERE tuple_key=? AND start_ns<=? AND (end_ns IS NULL OR end_ns>=?) LIMIT 2",
            (json.dumps(key, separators=(",", ":")), nanoseconds(end + slack), nanoseconds(lower)),
        )
        return [
            {
                "uid": uid,
                "start": Decimal(first) / 1_000_000,
                "end": Decimal(last) / 1_000_000 if last is not None else None,
                "label": label,
                "detail": detail,
            }
            for uid, first, last, label, detail in rows
        ]

    def close(self):
        self.connection.close()
