from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def render_messages(row: dict, tokenizer: AutoTokenizer) -> str:
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    tools_raw = row.get("tools", "") or ""
    tools = json.loads(tools_raw) if isinstance(tools_raw, str) and tools_raw.strip() else []
    if not isinstance(tools, list):
        tools = []
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts) + "\n\n"


def prepare_swe_parquet(
    train_parquet: str | Path,
    val_parquet: str | Path,
    output_dir: str | Path,
    model_name: str,
    text_field: str = "text",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    for split, src in [("train", train_parquet), ("validation", val_parquet)]:
        df = pd.read_parquet(src)
        df[text_field] = df.apply(lambda row: render_messages(row.to_dict(), tokenizer), axis=1)
        out = df[[text_field]].reset_index(drop=True)
        out_path = output_dir / f"{split}.parquet"
        out.to_parquet(out_path, index=False, compression="snappy")
        print(f"wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--text-field", default="text")
    args = p.parse_args()
    prepare_swe_parquet(args.train, args.val, args.output_dir, args.model, args.text_field)