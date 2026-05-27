from __future__ import annotations

import csv
import re
from pathlib import Path


SOURCE_PATH = Path("/Users/radhikaarora/Downloads/trades_20260526092550.csv")
OUTPUT_DIR = Path("/Users/radhikaarora/Documents/Trading ML V4/reports/manual_labels")
OUTPUT_PATH = OUTPUT_DIR / "trades_20260526092550_normalized.csv"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "unspecified"


def _split_reason_tags(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    return [_slug(part) for part in parts if part.strip()]


def _normalize_setup_valid(value: str) -> str:
    text = (value or "").strip().lower()
    if text in {"", "yes", "y", "true", "valid"}:
        return "yes"
    if text in {"no", "n", "false", "invalid"}:
        return "no"
    return text


def _quality_bucket(setup_valid: str, valid_tags: list[str]) -> str:
    if setup_valid == "no":
        return "avoid"
    if any(
        tag in {"great_structure_strength", "perfect_strong_1m_continuation_candle"}
        for tag in valid_tags
    ):
        return "high_quality"
    if valid_tags:
        return "valid"
    return "valid"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        valid_tags = _split_reason_tags(row.get("Valid Reason", ""))
        invalid_tags = _split_reason_tags(row.get("Invalid Reason", ""))
        setup_valid = _normalize_setup_valid(row.get("Setup Valid", ""))
        normalized_rows.append(
            {
                **row,
                "setup_valid_normalized": setup_valid,
                "quality_bucket": _quality_bucket(setup_valid, valid_tags),
                "valid_reason_tags": "|".join(valid_tags),
                "invalid_reason_tags": "|".join(invalid_tags),
                "primary_valid_reason": valid_tags[0] if valid_tags else "",
                "primary_invalid_reason": invalid_tags[0] if invalid_tags else "",
            }
        )

    fieldnames = list(normalized_rows[0].keys()) if normalized_rows else []
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
