#!/usr/bin/env python3
"""Profile CSV files in data/raw without modifying source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CATEGORY_COLUMNS = {
    "account",
    "address",
    "currency",
    "contingencytype",
    "execinst",
    "exectype",
    "lastliquidityind",
    "network",
    "ordrejreason",
    "ordstatus",
    "ordtype",
    "settlcurrency",
    "side",
    "symbol",
    "timeinforce",
    "tradepublishindicator",
    "transactstatus",
    "transacttype",
    "triggered",
    "workingindicator",
}
DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_ONLY_PATTERN = re.compile(r"^(?:\d{2}:)?\d{2}:\d{2}(?:\.\d+)?$")
DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"
)
EXPLICIT_ZONE_PATTERN = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, chunked profiling for CSV files under data/raw."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="CSV files to inspect")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit/csv_profile.json")
    )
    return parser.parse_args()


def normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_identifier(name: str) -> bool:
    return normalized_name(name).endswith("id")


def is_temporal(name: str) -> bool:
    normalized = normalized_name(name)
    return normalized == "date" or normalized.endswith("timestamp") or normalized.endswith(
        "transacttime"
    )


def json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time_format(value: str) -> str:
    if DATE_ONLY_PATTERN.fullmatch(value):
        return "date_only"
    if TIME_ONLY_PATTERN.fullmatch(value):
        return "time_only"
    if DATETIME_PATTERN.fullmatch(value):
        return "datetime_with_zone" if EXPLICIT_ZONE_PATTERN.search(value) else "datetime_naive"
    return "other"


def update_identifier_stats(stats: dict[str, Any], series: pd.Series) -> None:
    values = series.dropna().astype(str)
    values = values[values.str.strip().ne("")]
    counts = values.value_counts(dropna=False)
    for value, count_value in counts.items():
        count = int(count_value)
        if value in stats["seen"]:
            stats["duplicate_occurrences"] += count
        else:
            stats["seen"].add(value)
            stats["duplicate_occurrences"] += count - 1


def update_row_duplicates(
    seen_hashes: set[int], chunk: pd.DataFrame
) -> tuple[int, set[int]]:
    row_hashes = pd.util.hash_pandas_object(chunk, index=False, categorize=False)
    counts = row_hashes.value_counts()
    duplicates = 0
    for hash_value, count_value in counts.items():
        count = int(count_value)
        hash_integer = int(hash_value)
        if hash_integer in seen_hashes:
            duplicates += count
        else:
            seen_hashes.add(hash_integer)
            duplicates += count - 1
    return duplicates, seen_hashes


def profile_file(
    path: Path,
    chunk_size: int,
    sample_rows: int,
    global_identifier_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before = path.stat()
    sample = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        nrows=sample_rows,
        on_bad_lines="error",
    )
    columns = list(sample.columns)
    identifier_columns = [column for column in columns if is_identifier(column)]
    temporal_columns = [column for column in columns if is_temporal(column)]
    numeric_candidates = {
        column for column in columns if pd.api.types.is_numeric_dtype(sample[column])
    }
    column_stats: dict[str, dict[str, Any]] = {
        column: {
            "observed_pandas_dtypes": set(),
            "null_count": 0,
            "blank_string_count": 0,
        }
        for column in columns
    }
    identifier_stats = {
        column: {"seen": set(), "duplicate_occurrences": 0}
        for column in identifier_columns
    }
    category_counts = {
        column: Counter() for column in columns if normalized_name(column) in CATEGORY_COLUMNS
    }
    temporal_stats = {
        column: {
            "format_counts": Counter(),
            "min_lexical": None,
            "max_lexical": None,
            "explicit_timezone_count": 0,
            "distinct_values": set() if normalized_name(column) == "date" else None,
            "previous_value": None,
            "lexical_order_violation_count": 0,
        }
        for column in temporal_columns
    }
    numeric_stats = {
        column: {"min": None, "max": None, "parse_failure_count": 0}
        for column in numeric_candidates
    }
    row_count = 0
    fully_empty_row_count = 0
    row_duplicate_count = 0
    row_hashes: set[int] = set()
    date_timestamp_mismatches: dict[str, int] = {}
    quality_checks = {
        "non_positive_lastqty_count": 0,
        "negative_leavesqty_count": 0,
        "cumqty_less_than_lastqty_count": 0,
        "order_balance_mismatch_count": 0,
        "trade_missing_side_count": 0,
        "trade_missing_lastpx_count": 0,
        "trade_missing_lastqty_count": 0,
    }

    reader = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        chunksize=chunk_size,
        on_bad_lines="error",
    )
    for chunk in reader:
        row_count += len(chunk)
        fully_empty_row_count += int(chunk.isna().all(axis=1).sum())
        duplicate_increment, row_hashes = update_row_duplicates(row_hashes, chunk)
        row_duplicate_count += duplicate_increment

        if {"lastqty", "leavesqty", "cumqty", "orderqty"}.issubset(chunk.columns):
            quality_checks["non_positive_lastqty_count"] += int(
                chunk["lastqty"].le(0).sum()
            )
            quality_checks["negative_leavesqty_count"] += int(
                chunk["leavesqty"].lt(0).sum()
            )
            quality_checks["cumqty_less_than_lastqty_count"] += int(
                chunk["cumqty"].lt(chunk["lastqty"]).sum()
            )
            quality_checks["order_balance_mismatch_count"] += int(
                chunk["orderqty"].ne(chunk["leavesqty"] + chunk["cumqty"]).sum()
            )

        if "exectype" in chunk.columns:
            trade_rows = chunk["exectype"].eq("Trade")
            for column in ("side", "lastpx", "lastqty"):
                if column in chunk.columns:
                    quality_checks[f"trade_missing_{column}_count"] += int(
                        (trade_rows & chunk[column].isna()).sum()
                    )

        for column in columns:
            series = chunk[column]
            stats = column_stats[column]
            stats["observed_pandas_dtypes"].add(str(series.dtype))
            stats["null_count"] += int(series.isna().sum())
            strings = series.astype("string")
            stats["blank_string_count"] += int(strings.str.strip().eq("").sum())

            if column in identifier_stats:
                update_identifier_stats(identifier_stats[column], series)
                global_stats = global_identifier_stats.setdefault(
                    column, {"seen": set(), "duplicate_occurrences": 0}
                )
                update_identifier_stats(global_stats, series)

            if column in category_counts:
                category_counts[column].update(strings.dropna().tolist())

            if column in temporal_stats:
                values = strings.dropna()
                values = values[values.str.strip().ne("")]
                if not values.empty:
                    temporal = temporal_stats[column]
                    value_min = str(values.min())
                    value_max = str(values.max())
                    temporal["min_lexical"] = (
                        value_min
                        if temporal["min_lexical"] is None
                        else min(temporal["min_lexical"], value_min)
                    )
                    temporal["max_lexical"] = (
                        value_max
                        if temporal["max_lexical"] is None
                        else max(temporal["max_lexical"], value_max)
                    )
                    temporal["format_counts"].update(values.map(time_format).tolist())
                    temporal["explicit_timezone_count"] += int(
                        values.str.contains(EXPLICIT_ZONE_PATTERN).sum()
                    )
                    if temporal["distinct_values"] is not None:
                        temporal["distinct_values"].update(values.unique().tolist())
                    previous = temporal["previous_value"]
                    if previous is not None and str(values.iloc[0]) < previous:
                        temporal["lexical_order_violation_count"] += 1
                    temporal["lexical_order_violation_count"] += int(
                        values.lt(values.shift()).sum()
                    )
                    temporal["previous_value"] = str(values.iloc[-1])

            if column in numeric_stats:
                parsed = pd.to_numeric(series, errors="coerce")
                failures = series.notna() & parsed.isna()
                numeric_stats[column]["parse_failure_count"] += int(failures.sum())
                valid = parsed.dropna()
                if not valid.empty:
                    value_min = json_scalar(valid.min())
                    value_max = json_scalar(valid.max())
                    current_min = numeric_stats[column]["min"]
                    current_max = numeric_stats[column]["max"]
                    numeric_stats[column]["min"] = (
                        value_min if current_min is None else min(current_min, value_min)
                    )
                    numeric_stats[column]["max"] = (
                        value_max if current_max is None else max(current_max, value_max)
                    )

        if "date" in chunk.columns:
            dates = chunk["date"].astype("string")
            for column in temporal_columns:
                if column == "date":
                    continue
                timestamps = chunk[column].astype("string")
                has_full_date = timestamps.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
                comparable = dates.notna() & timestamps.notna() & has_full_date
                mismatches = comparable & dates.ne(timestamps.str.slice(0, 10))
                date_timestamp_mismatches[column] = (
                    date_timestamp_mismatches.get(column, 0) + int(mismatches.sum())
                )

    after = path.stat()
    sample_top_values = {
        column: {
            str(value): int(count)
            for value, count in sample[column]
            .astype("string")
            .value_counts(dropna=False)
            .head(5)
            .items()
        }
        for column in columns
    }

    for column, stats in column_stats.items():
        stats["observed_pandas_dtypes"] = sorted(stats["observed_pandas_dtypes"])
        stats["null_rate"] = stats["null_count"] / row_count if row_count else None
        stats["sample_inferred_dtype"] = str(sample[column].dtype)
        stats["sample_distinct_count"] = int(sample[column].nunique(dropna=True))
        stats["sample_top_values"] = sample_top_values[column]

    rendered_identifiers = {
        column: {
            "non_blank_distinct_count": len(stats["seen"]),
            "duplicate_occurrences": stats["duplicate_occurrences"],
        }
        for column, stats in identifier_stats.items()
    }
    rendered_categories = {
        column: {
            "distinct_count": len(counts),
            "value_counts": dict(counts.most_common()),
        }
        for column, counts in category_counts.items()
    }
    rendered_temporal = {
        column: {
            "format_counts": dict(stats["format_counts"]),
            "min_lexical": stats["min_lexical"],
            "max_lexical": stats["max_lexical"],
            "explicit_timezone_count": stats["explicit_timezone_count"],
            "timezone_encoded": stats["explicit_timezone_count"] > 0,
            "distinct_count": (
                len(stats["distinct_values"])
                if stats["distinct_values"] is not None
                else None
            ),
            "lexical_order_violation_count": stats[
                "lexical_order_violation_count"
            ],
        }
        for column, stats in temporal_stats.items()
    }

    return {
        "path": str(path),
        "size_bytes": before.st_size,
        "sha256": sha256_file(path),
        "source_unchanged_during_scan": (
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        ),
        "row_count": row_count,
        "non_empty_row_count": row_count - fully_empty_row_count,
        "fully_empty_row_count": fully_empty_row_count,
        "column_count": len(columns),
        "columns": columns,
        "column_profiles": column_stats,
        "identifier_profiles": rendered_identifiers,
        "categorical_profiles": rendered_categories,
        "temporal_profiles": rendered_temporal,
        "numeric_profiles": numeric_stats,
        "date_timestamp_mismatch_counts": date_timestamp_mismatches,
        "quality_checks": quality_checks,
        "duplicate_row_occurrences_64bit_hash": row_duplicate_count,
        "duplicate_method": "pandas stable 64-bit row hash; collision risk is non-zero",
        "sample_method": f"first {min(sample_rows, row_count)} rows",
    }


def resolve_paths(raw_dir: Path, requested: list[Path]) -> list[Path]:
    raw_resolved = raw_dir.resolve()
    paths = requested or sorted(raw_dir.glob("*.csv"))
    resolved: list[Path] = []
    for path in paths:
        candidate = path.resolve()
        if raw_resolved != candidate.parent and raw_resolved not in candidate.parents:
            raise ValueError(f"input must be under {raw_dir}: {path}")
        if not candidate.is_file():
            raise FileNotFoundError(path)
        resolved.append(candidate)
    if not resolved:
        raise FileNotFoundError(f"no CSV files found under {raw_dir}")
    return resolved


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.sample_rows <= 0:
        raise ValueError("chunk size and sample rows must be positive")

    raw_dir = args.raw_dir.resolve()
    output = args.output.resolve()
    if output == raw_dir or raw_dir in output.parents:
        raise ValueError("output must not be written inside the raw data directory")

    paths = resolve_paths(args.raw_dir, args.paths)
    global_identifier_stats: dict[str, dict[str, Any]] = {}
    profiles = [
        profile_file(
            path,
            args.chunk_size,
            args.sample_rows,
            global_identifier_stats,
        )
        for path in paths
    ]
    rendered_global_identifiers = {
        column: {
            "non_blank_distinct_count": len(stats["seen"]),
            "duplicate_occurrences": stats["duplicate_occurrences"],
            "cross_file_repeated_distinct_occurrences": sum(
                profile["identifier_profiles"].get(column, {}).get(
                    "non_blank_distinct_count", 0
                )
                for profile in profiles
            )
            - len(stats["seen"]),
        }
        for column, stats in global_identifier_stats.items()
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "raw_dir": str(raw_dir),
            "chunk_size": args.chunk_size,
            "sample_rows": args.sample_rows,
        },
        "identifier_profiles_all_files": rendered_global_identifiers,
        "files": profiles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"report: {output}")
    for profile in profiles:
        print(
            f"{Path(profile['path']).name}: "
            f"rows={profile['row_count']:,}, columns={profile['column_count']}, "
            f"duplicate_rows={profile['duplicate_row_occurrences_64bit_hash']:,}, "
            f"source_unchanged={profile['source_unchanged_during_scan']}"
        )


if __name__ == "__main__":
    main()
