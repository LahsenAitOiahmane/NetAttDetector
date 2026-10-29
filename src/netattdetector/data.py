"""A versioned feature contract and bounded uniform sampling over entire CSVs."""

import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from netattdetector.io import sha256_file

# UQ extended NetFlow v2. Addresses, labels and dataset identifiers are excluded.
FEATURES = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]
SCHEMA_VERSION = "uq-netflow-v2-numeric-1"


@contextmanager
def csv_stream(path, member=None):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                x
                for x in archive.namelist()
                if x.lower().endswith(".csv")
                and not x.startswith("__MACOSX/")
                and Path(x).name.lower() != "netflow_v2_features.csv"
            )
            if member is None and len(names) != 1:
                raise ValueError(f"Specify --member; archive contains {names}")
            selected = member or names[0]
            if selected not in names:
                raise ValueError(f"CSV member {selected!r} not in archive: {names}")
            with archive.open(selected) as stream:
                yield stream
    else:
        with path.open("rb") as stream:
            yield stream


def numeric_features(frame, features=FEATURES):
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature schema mismatch; missing columns: {missing}")
    result = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float32)
    return result.replace([np.inf, -np.inf], np.nan)


def fingerprints(frame, features=FEATURES):
    # Same model-input representation => same fingerprint, independently of label or IP.
    return pd.util.hash_pandas_object(frame.loc[:, features], index=False).to_numpy()


def load_sample(path, max_rows=200_000, chunk_rows=50_000, seed=42, member=None, group_column=None):
    if max_rows < 100 or chunk_rows < 1:
        raise ValueError("Use max-rows >= 100 and chunk-rows >= 1.")
    rng = np.random.default_rng(seed)
    sampled = None
    priorities = np.empty(0)
    rows = 0
    invalid_labels = 0
    labels = Counter()
    categories = Counter()
    with csv_stream(path, member) as stream:
        for chunk in pd.read_csv(stream, chunksize=chunk_rows, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            if chunk.columns.duplicated().any():
                raise ValueError("Duplicate column names after whitespace normalization.")
            if "Label" not in chunk:
                raise ValueError("Expected UQ binary target 'Label' (0=benign, 1=attack).")
            if group_column and group_column not in chunk:
                raise ValueError(f"Missing grouping column: {group_column}")
            target = pd.to_numeric(chunk["Label"], errors="coerce")
            valid = target.isin([0, 1])
            rows += len(chunk)
            invalid_labels += int((~valid).sum())
            chunk = chunk.loc[valid].reset_index(drop=True)
            target = target.loc[valid].reset_index(drop=True).astype(np.int8)
            labels.update(int(x) for x in target)
            category = (
                chunk["Attack"].fillna("unknown").astype(str)
                if "Attack" in chunk
                else target.astype(str)
            )
            categories.update(category)
            clean = numeric_features(chunk)
            clean["__label"] = target
            clean["__attack"] = category
            if group_column:
                if chunk[group_column].isna().any():
                    raise ValueError("Split grouping/time column contains missing values.")
                clean["__group"] = chunk[group_column].astype(str)
            keys = rng.random(len(clean))
            sampled = clean if sampled is None else pd.concat([sampled, clean], ignore_index=True)
            priorities = np.concatenate([priorities, keys])
            if len(sampled) > max_rows:
                keep = np.argpartition(priorities, max_rows - 1)[:max_rows]
                sampled = sampled.iloc[keep].reset_index(drop=True)
                priorities = priorities[keep]
    if sampled is None or sampled.empty:
        raise ValueError("No valid labeled rows found.")
    # Deterministic ordering, independent of argpartition's incidental arrangement.
    order = np.argsort(priorities)
    sampled = sampled.iloc[order].reset_index(drop=True)
    before = len(sampled)
    hashes = fingerprints(sampled)
    sampled["__hash"] = hashes
    conflicts = sampled.groupby("__hash")["__label"].nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    conflict_rows = int(sampled["__hash"].isin(conflicting).sum())
    sampled = (
        sampled.loc[~sampled["__hash"].isin(conflicting)]
        .drop_duplicates("__hash")
        .reset_index(drop=True)
    )
    if sampled["__label"].nunique() != 2:
        raise ValueError("Need both benign and attack examples after cleaning.")
    profile = {
        "input": str(Path(path)),
        "sha256": sha256_file(path),
        "member": member,
        "schema": SCHEMA_VERSION,
        "rows_scanned": rows,
        "invalid_label_rows": invalid_labels,
        "full_label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
        "full_attack_counts": dict(sorted(categories.items())),
        "sampling": "uniform random priorities over all valid rows; no class rebalancing",
        "seed": seed,
        "requested_max_rows": max_rows,
        "sample_rows_before_deduplication": before,
        "conflicting_feature_rows_removed": conflict_rows,
        "duplicate_feature_rows_removed": before - conflict_rows - len(sampled),
        "sample_rows": len(sampled),
        "sample_label_counts": {
            str(k): int(v) for k, v in sampled["__label"].value_counts().sort_index().items()
        },
        "sample_attack_counts": {
            str(k): int(v) for k, v in sampled["__attack"].value_counts().items()
        },
        "duplicate_scope": "bounded sample, not full archive; float32 feature fingerprints",
    }
    return sampled, profile


def describe_sample(frame):
    features = frame[FEATURES]
    stats = {}
    for name in FEATURES:
        values = features[name]
        valid = values.dropna()
        stats[name] = {
            "missing": int(values.isna().sum()),
            "unique": int(valid.nunique()),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "mean": float(valid.astype(np.float64).mean()) if len(valid) else None,
            "median": float(valid.median()) if len(valid) else None,
        }
    corr = features.corr().abs()
    pairs = [
        (FEATURES[i], FEATURES[j], float(corr.iloc[i, j]))
        for i in range(len(FEATURES))
        for j in range(i)
        if pd.notna(corr.iloc[i, j])
    ]
    pairs.sort(key=lambda item: item[2], reverse=True)
    return {"sample_feature_statistics": stats, "top_absolute_correlations": pairs[:20]}
