#!/usr/bin/env python3
"""POST sample diffs to the review API and save JSON responses."""

import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent
REVIEW_DELAY_SEC = 15
MAX_RETRIES = 3

DIFFS = [
    ("diff1_python.txt", "python", "diff1_review.json"),
    ("diff2_javascript.txt", "javascript", "diff2_review.json"),
    ("diff3_typescript.txt", "typescript", "diff3_review.json"),
]


def post_review(client: httpx.Client, diff_content: str, language: str) -> dict:
    for attempt in range(MAX_RETRIES):
        response = client.post(
            "/review",
            json={"diff": diff_content, "language": language},
        )
        if response.status_code == 502 and attempt < MAX_RETRIES - 1:
            wait = REVIEW_DELAY_SEC * (attempt + 1)
            print(f"  Server error, retrying in {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Review request failed after retries")


def main() -> int:
    reviews_dir = ROOT / "reviews"
    diffs_dir = ROOT / "diffs"
    reviews_dir.mkdir(exist_ok=True)

    try:
        with httpx.Client(base_url=BASE_URL, timeout=600.0) as client:
            for index, (diff_file, language, output_file) in enumerate(DIFFS):
                if index > 0:
                    time.sleep(REVIEW_DELAY_SEC)

                diff_path = diffs_dir / diff_file
                out_path = reviews_dir / output_file

                print(f"Reviewing {diff_file}...")
                diff_content = diff_path.read_text(encoding="utf-8")
                report = post_review(client, diff_content, language)

                out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"Saved to reviews/{output_file}")

    except httpx.ConnectError:
        print(
            "Error: Could not connect to the API. "
            "Start the server with: uvicorn app.main:app --reload",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"Error: API returned {exc.response.status_code}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: Missing file {exc.filename}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
