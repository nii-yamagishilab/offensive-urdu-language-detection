import argparse
import gc
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# -----------------------------
# Helpers: config + prompts
# -----------------------------
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompt_template(path: Path) -> tuple[str, str]:
    txt = path.read_text(encoding="utf-8")
    if "SYSTEM:" not in txt or "USER:" not in txt:
        raise ValueError(f"Prompt file must contain SYSTEM: and USER: sections: {path}")
    system = txt.split("SYSTEM:", 1)[1].split("USER:", 1)[0].strip()
    user = txt.split("USER:", 1)[1].strip()
    return system, user

def build_chat_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -----------------------------
# Label set loading
# -----------------------------
def load_label_set(label_cfg_path: Path, label_set_name: str):
    """
    Load a label set from configs/label_sets.yaml.

    Returns:
      labels: list[str] canonical labels to match/return
      aliases: dict[str -> str] mapping normalized output -> canonical label
    """
    cfg = load_yaml(label_cfg_path)
    if "label_sets" not in cfg or label_set_name not in cfg["label_sets"]:
        available = list(cfg.get("label_sets", {}).keys())
        raise KeyError(
            f"Label set '{label_set_name}' not found in {label_cfg_path}. "
            f"Available: {available}"
        )

    ls = cfg["label_sets"][label_set_name]
    labels = ls["labels"]
    aliases = {k.lower(): v for k, v in ls.get("aliases", {}).items()}
    return labels, aliases


# -----------------------------
# Output post-processing
# -----------------------------
def postprocess_label(gen_text: str, labels: list[str], aliases: dict[str, str]) -> str:
    """
    Convert model output into a canonical label from the chosen label set.

    Strategy:
      1) clean special tokens
      2) take first line
      3) exact alias match
      4) handle 'not abusive' vs 'abusive' ordering
      5) substring match against canonical labels
      6) Unknown fallback
    """
    raw = re.sub(r"<\|.*?\|>", "", str(gen_text)).strip()
    first = raw.splitlines()[0].strip().rstrip(".:;").strip()
    key = first.lower()

    # 1) Exact alias match
    if key in aliases:
        return aliases[key]

    # 2) Common ambiguity: "Not Abusive" should be checked before "Abusive"
    if key.startswith("not abusive") and "not abusive" in aliases:
        return aliases["not abusive"]
    if key.startswith("abusive") and "abusive" in aliases:
        return aliases["abusive"]

    # 3) Substring match against canonical labels
    for lab in labels:
        if lab.lower() in key:
            return lab

    return "Unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", required=True, help="Path to YAML model config")
    ap.add_argument("--prompt", required=True, help="Path to prompt template .txt (SYSTEM:/USER:)")
    ap.add_argument("--data", required=True, help="CSV path with columns: sentences,label")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)

    # HF login optional
    ap.add_argument("--hf-login", action="store_true",
                    help="Call huggingface_hub.login() interactively")

    # NEW: label set selection
    ap.add_argument("--label_cfg", default="configs/label_sets.yaml",
                    help="Path to label set config YAML")
    ap.add_argument("--label_set", required=True,
                    help="Label set name (e.g., task2_fiveclass_roman, ruhsold_fine_en, coarse_2class_urdu)")

    args = ap.parse_args()

    if args.hf_login:
        login()

    # Load model config
    cfg = load_yaml(Path(args.model_config))
    model_id = cfg["model_id"]
    max_new_tokens = int(cfg.get("max_new_tokens", 10))
    dtype = torch.bfloat16 if str(cfg.get("dtype", "bfloat16")).lower() == "bfloat16" else torch.float16

    # Load label set config
    labels, aliases = load_label_set(Path(args.label_cfg), args.label_set)

    # Load model + pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    # Load prompt template
    system_text, user_template = load_prompt_template(Path(args.prompt))

    # Load data
    df = pd.read_csv(args.data)
    df = df.iloc[args.start:args.end].reset_index(drop=True)

    # Prepare output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        out_path.write_text("Index,sentences,Predicted,label\n", encoding="utf-8")

    # Run
    for i, row in df.iterrows():
        sentence = str(row["sentences"])
        gold = str(row.get("label", ""))

        user_text = user_template.replace("{sentence}", sentence)
        prompt = build_chat_prompt(tokenizer, system_text, user_text)

        out = gen(prompt, num_return_sequences=1)[0]["generated_text"]
        pred = postprocess_label(out, labels, aliases)

        # CSV-safe quoting
        sent_clean = sentence.replace('"', '""')
        pred_clean = pred.replace('"', '""')
        gold_clean = gold.replace('"', '""')

        with out_path.open("a", encoding="utf-8") as f:
            f.write(f'{i},"{sent_clean}","{pred_clean}","{gold_clean}"\n')

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(df)}")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"✅ Done. Saved: {out_path}")


if __name__ == "__main__":
    main()
