"""Global fingerprint audits on disk and bounded, reusable development cohorts."""

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from netattdetector.data import FEATURES, SCHEMA_VERSION, csv_stream, fingerprints, numeric_features
from netattdetector.io import sha256_file, write_json

SHORTCUT_FEATURES = {"L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL", "PROTOCOL", "L7_PROTO"}
STABLE_FEATURES = [x for x in FEATURES if x not in SHORTCUT_FEATURES]
RECORD = np.dtype([("hash", "<u8"), ("label", "u1"), ("source", "u1")])


def dictionary_hash(path):
    if Path(path).suffix.lower() != ".zip":
        return None
    import hashlib

    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if Path(n).name.lower() == "netflow_v2_features.csv"]
        return hashlib.sha256(archive.read(names[0])).hexdigest() if len(names) == 1 else None


class DiskAudit:
    """At most one hash partition is sorted in memory; no full-flow table is retained."""

    def __init__(self, root, buckets=64):
        if not 2 <= buckets <= 256:
            raise ValueError("Use 2..256 audit buckets.")
        self.root = Path(root)
        self.buckets = buckets
        self.root.mkdir(parents=True, exist_ok=True)
        self.handles = {}
        for kind in ("full", "stable"):
            for bucket in range(buckets):
                self.handles[kind, bucket] = (self.root / f"{kind}-{bucket}.bin").open("xb")

    def add(self, frame, source):
        for kind, features in (("full", FEATURES), ("stable", STABLE_FEATURES)):
            records = np.empty(len(frame), dtype=RECORD)
            records["hash"] = fingerprints(frame, features)
            records["label"] = frame["__label"].to_numpy()
            records["source"] = source
            bucket = records["hash"] % self.buckets
            order = np.argsort(bucket, kind="stable")
            records, bucket = records[order], bucket[order]
            boundaries = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1, len(records)]
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                if stop > start:
                    self.handles[kind, int(bucket[start])].write(records[start:stop].tobytes())

    def close(self):
        for handle in self.handles.values():
            handle.close()

    def finish(self, cohort, source_names):
        self.close()
        clean = np.ones(len(cohort), dtype=bool)
        result = {}
        for kind, features in (("full", FEATURES), ("stable", STABLE_FEATURES)):
            totals = Counter()
            per_source = {name: Counter() for name in source_names}
            sample_hashes = fingerprints(cohort, features)
            for bucket in range(self.buckets):
                shard = self.root / f"{kind}-{bucket}.bin"
                records = np.fromfile(shard, dtype=RECORD)
                if not len(records):
                    np.save(self.root / f"{kind}-{bucket}.npy", np.empty(0, dtype=np.uint64))
                    shard.unlink()
                    continue
                records.sort(order="hash")
                starts = np.r_[0, np.flatnonzero(np.diff(records["hash"])) + 1]
                unique = records["hash"][starts]
                counts = np.diff(np.r_[starts, len(records)])
                low = np.minimum.reduceat(records["label"], starts)
                high = np.maximum.reduceat(records["label"], starts)
                conflict = low != high
                shared = np.minimum.reduceat(records["source"], starts) != np.maximum.reduceat(
                    records["source"], starts
                )
                totals.update(
                    {
                        "rows": len(records),
                        "unique_feature_groups": len(unique),
                        "repeated_rows": len(records) - len(unique),
                        "conflicting_feature_groups": int(conflict.sum()),
                        "rows_in_conflicting_groups": int(counts[conflict].sum()),
                        "groups_shared_across_sources": int(shared.sum()),
                    }
                )
                bad = unique[conflict]
                selected = np.flatnonzero(sample_hashes % self.buckets == bucket)
                if kind == "full":
                    clean[selected] &= ~np.isin(sample_hashes[selected], bad)
                # Retained index supports overlap exclusion against ALL development rows.
                np.save(self.root / f"{kind}-{bucket}.npy", unique)
                for source, name in enumerate(source_names):
                    subset = records[records["source"] == source]
                    if len(subset):
                        _, source_counts = np.unique(subset["hash"], return_counts=True)
                        per_source[name].update(
                            {
                                "rows": len(subset),
                                "unique_feature_groups": len(source_counts),
                                "repeated_rows": len(subset) - len(source_counts),
                            }
                        )
                shard.unlink()
            result[kind] = {
                **dict(totals),
                "per_source": {k: dict(v) for k, v in per_source.items()},
            }
        return clean, result


def overlaps_development(frame, root, buckets, kind="full"):
    if kind not in {"full", "stable"}:
        raise ValueError("Unknown fingerprint kind.")
    hashes = fingerprints(frame, FEATURES if kind == "full" else STABLE_FEATURES)
    overlap = np.zeros(len(frame), dtype=bool)
    for bucket in range(buckets):
        selected = np.flatnonzero(hashes % buckets == bucket)
        known = np.load(Path(root) / f"{kind}-{bucket}.npy", allow_pickle=False)
        overlap[selected] = np.isin(hashes[selected], known)
    return overlap


def source_diagnostics(frame):
    sources = sorted(frame["__source"].unique())
    if len(sources) != 2:
        raise ValueError("This diagnostic currently compares exactly two development sources.")
    reports = []
    for feature in FEATURES:
        record = {
            "feature": feature,
            "removed_in_stable_variant": feature in SHORTCUT_FEATURES,
            "sources": {},
        }
        for source in sources:
            part = frame.loc[frame["__source"] == source]
            valid = part[feature].dropna()
            corr = (
                spearmanr(part[feature], part["__label"], nan_policy="omit").statistic
                if valid.nunique() > 1
                else np.nan
            )
            record["sources"][source] = {
                "missing_fraction": float(part[feature].isna().mean()),
                "quantiles": [float(x) for x in valid.quantile([0.05, 0.5, 0.95])]
                if len(valid)
                else None,
                "spearman_label_association": float(corr) if np.isfinite(corr) else None,
            }
        for label, suffix in ((None, "all"), (0, "benign"), (1, "attack")):
            values = [
                frame.loc[
                    (frame["__source"] == s)
                    & (True if label is None else frame["__label"] == label),
                    feature,
                ].dropna()
                for s in sources
            ]
            record[f"ks_{suffix}"] = (
                float(ks_2samp(*values, method="asymp").statistic)
                if all(len(v) for v in values)
                else None
            )
        reports.append(record)
    reports.sort(key=lambda x: x["ks_benign"] or 0, reverse=True)
    return {
        "scope": "cleaned bounded development sample; KS is distribution distance, not proof of cause",
        "source_order": sources,
        "features": reports,
    }


def prepare_sources(paths, output, rows_per_source=100_000, chunk_rows=50_000, seed=42, buckets=64):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Preparation output must be fresh; existing caches are never silently overwritten."
        )
    if not 100 <= rows_per_source <= 500_000 or chunk_rows < 1 or not 2 <= len(paths) <= 10:
        raise ValueError(
            "Use 2..10 sources, 100..500000 sampled rows/source, and positive chunk_rows."
        )
    output.mkdir(parents=True, exist_ok=True)
    audit = DiskAudit(output / "hash-index", buckets)
    samples, profiles = [], {}
    try:
        for source_id, (source, path) in enumerate(paths.items()):
            print(f"Auditing full development source {source}...", flush=True)
            rng = np.random.default_rng(seed)
            sample, priorities = None, np.empty(0)
            counts, attacks = Counter(), Counter()
            rows = invalid_labels = label_disagreements = empty_rows = 0
            missing = np.zeros(len(FEATURES), dtype=np.int64)
            negative = missing.copy()
            invalid_numeric = missing.copy()
            ranges = Counter()
            with csv_stream(path) as stream:
                for chunk in pd.read_csv(stream, chunksize=chunk_rows, low_memory=False):
                    chunk.columns = chunk.columns.str.strip()
                    if chunk.columns.duplicated().any() or "Label" not in chunk:
                        raise ValueError("Duplicate columns or missing binary Label.")
                    rows += len(chunk)
                    y = pd.to_numeric(chunk["Label"], errors="coerce")
                    valid = y.isin([0, 1])
                    invalid_labels += int((~valid).sum())
                    chunk = chunk.loc[valid].reset_index(drop=True)
                    y = y.loc[valid].reset_index(drop=True).astype(np.int8)
                    clean = numeric_features(chunk)
                    missing += clean.isna().sum().to_numpy()
                    invalid_numeric += (clean.isna() & chunk[FEATURES].notna()).sum().to_numpy()
                    negative += (clean < 0).sum().to_numpy()
                    for feature, maximum in (
                        ("L4_SRC_PORT", 65535),
                        ("L4_DST_PORT", 65535),
                        ("PROTOCOL", 255),
                        ("MIN_TTL", 255),
                        ("MAX_TTL", 255),
                    ):
                        ranges[feature] += int(
                            ((clean[feature] < 0) | (clean[feature] > maximum)).sum()
                        )
                    category = (
                        chunk["Attack"].fillna("unknown").astype(str)
                        if "Attack" in chunk
                        else y.astype(str)
                    )
                    benign_name = category.str.strip().str.lower().isin(["benign", "normal"])
                    label_disagreements += (
                        int(((y == 0) != benign_name).sum()) if "Attack" in chunk else 0
                    )
                    counts.update(int(x) for x in y)
                    attacks.update(category)
                    clean["__label"] = y
                    clean["__attack"] = category
                    clean["__source"] = source
                    usable = ~clean[FEATURES].isna().all(axis=1)
                    empty_rows += int((~usable).sum())
                    clean = clean.loc[usable].reset_index(drop=True)
                    audit.add(clean, source_id)
                    keys = rng.random(len(clean))
                    sample = (
                        clean if sample is None else pd.concat([sample, clean], ignore_index=True)
                    )
                    priorities = np.r_[priorities, keys]
                    if len(sample) > rows_per_source:
                        keep = np.argpartition(priorities, rows_per_source - 1)[:rows_per_source]
                        sample, priorities = (
                            sample.iloc[keep].reset_index(drop=True),
                            priorities[keep],
                        )
            if sample is None or sample.empty:
                raise ValueError(f"No usable development rows: {source}")
            sample = sample.iloc[np.argsort(priorities)].reset_index(drop=True)
            samples.append(sample)
            profiles[source] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "feature_dictionary_sha256": dictionary_hash(path),
                "rows_scanned": rows,
                "invalid_labels": invalid_labels,
                "empty_feature_rows": empty_rows,
                "binary_label_vs_attack_name_disagreements": label_disagreements,
                "label_counts": {str(k): int(v) for k, v in counts.items()},
                "attack_counts": dict(attacks),
                "missing_or_nonfinite": dict(zip(FEATURES, map(int, missing), strict=True)),
                "coerced_or_nonfinite_nonmissing": dict(
                    zip(FEATURES, map(int, invalid_numeric), strict=True)
                ),
                "negative_values": dict(zip(FEATURES, map(int, negative), strict=True)),
                "out_of_range": dict(ranges),
                "sample_before_global_cleaning": len(sample),
            }
        cohort = pd.concat(samples, ignore_index=True)
        keep, global_audit = audit.finish(cohort, list(paths))
    finally:
        audit.close()
    global_removed = int((~keep).sum())
    cohort = cohort.loc[keep].reset_index(drop=True)
    cohort["__hash"] = fingerprints(cohort, STABLE_FEATURES)
    cohort["__full_hash"] = fingerprints(cohort, FEATURES)
    # Global conflicts were excluded above; a second explicit check guards cache/split invariants.
    if (cohort.groupby("__full_hash")["__label"].nunique() > 1).any():
        raise ValueError("Conflicting model-input groups remain after global audit.")
    before = len(cohort)
    cohort = cohort.drop_duplicates("__full_hash").reset_index(drop=True)
    for source in paths:
        if cohort.loc[cohort["__source"] == source, "__label"].nunique() != 2:
            raise ValueError(f"Global cleaning removed a class from {source}.")
    cache = output / "cohort.npz"
    np.savez_compressed(
        cache,
        values=cohort[FEATURES].to_numpy(dtype=np.float32),
        label=cohort["__label"].to_numpy(dtype=np.int8),
        attack=cohort["__attack"].to_numpy(dtype=str),
        source=cohort["__source"].to_numpy(dtype=str),
        fingerprint=cohort["__hash"].to_numpy(dtype=np.uint64),
        full_fingerprint=cohort["__full_hash"].to_numpy(dtype=np.uint64),
    )
    report = {
        "schema": SCHEMA_VERSION,
        "seed": seed,
        "rows_per_source": rows_per_source,
        "chunk_rows": chunk_rows,
        "buckets": buckets,
        "sources": profiles,
        "global_audit": global_audit,
        "sample_global_conflict_rows_removed": global_removed,
        "sample_repeated_full_rows_removed": before - len(cohort),
        "cleaning_policy": "full-feature conflicts/duplicates removed; reduced ambiguity retained",
        "cohort_rows": len(cohort),
        "source_label_counts": {
            f"{s}:{y}": len(g) for (s, y), g in cohort.groupby(["__source", "__label"])
        },
        "grouping_features": STABLE_FEATURES,
        "cache_sha256": sha256_file(cache),
        "fingerprint_limit": "64-bit hashes of float32 numeric representations; theoretical collisions and near-duplicates are not resolved",
        "split_limit": "No capture/session/timestamp metadata recovered; source identity is retained outside features.",
    }
    write_json(output / "audit.json", report)
    write_json(output / "diagnostics.json", source_diagnostics(cohort))
    return report


def load_prepared(root):
    root = Path(root)
    report = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if (
        report["schema"] != SCHEMA_VERSION
        or sha256_file(root / "cohort.npz") != report["cache_sha256"]
    ):
        raise ValueError("Prepared cohort checksum/schema mismatch.")
    with np.load(root / "cohort.npz", allow_pickle=False) as data:
        frame = pd.DataFrame(data["values"], columns=FEATURES)
        for column, key in (
            ("__label", "label"),
            ("__attack", "attack"),
            ("__source", "source"),
            ("__hash", "fingerprint"),
        ):
            frame[column] = data[key]
        frame["__full_hash"] = (
            data["full_fingerprint"] if "full_fingerprint" in data else fingerprints(frame)
        )
    return frame, report
