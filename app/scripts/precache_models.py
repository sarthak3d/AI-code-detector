from __future__ import annotations

import argparse
import os
from typing import List

from huggingface_hub import snapshot_download


MODEL_ALIASES = {
    "starcoder2-3b": "bigcode/starcoder2-3b",
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache detector models into Hugging Face cache.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model alias to cache (repeatable).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra HF repo IDs to cache directly (repeatable).",
    )
    parser.add_argument(
        "--include-t5",
        action="store_true",
        help="Also cache t5-small used by T5-NPR features.",
    )
    return parser.parse_args()


def resolve_repo_ids(models: List[str], extras: List[str], include_t5: bool) -> List[str]:
    selected = models or list(MODEL_ALIASES.keys())
    repo_ids: List[str] = []

    for alias in selected:
        if alias not in MODEL_ALIASES:
            raise ValueError(f"Unknown model alias: {alias}. Valid aliases: {sorted(MODEL_ALIASES.keys())}")
        repo_ids.append(MODEL_ALIASES[alias])

    repo_ids.extend(extras)
    if include_t5:
        repo_ids.append("t5-small")

    deduped: List[str] = []
    seen = set()
    for repo_id in repo_ids:
        if repo_id not in seen:
            deduped.append(repo_id)
            seen.add(repo_id)
    return deduped


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    repo_ids = resolve_repo_ids(args.model, args.extra, args.include_t5)

    print(f"Caching {len(repo_ids)} model repositories...")
    for repo_id in repo_ids:
        print(f" - {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            token=token,
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite"],
        )
    print("Done.")


if __name__ == "__main__":
    main()
