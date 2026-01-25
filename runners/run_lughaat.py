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
# Helpers: config + prompt
# -----------------------------
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# -----------------------------
# Label set loading
# -----------------------------
def load_label_set(label_cfg_path: Path, label_set_name: str):
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
def extract_label_from_text(gen_text: str) -> str:
    """
    Extract a label-like string from Lughaat output.
    Tries: after 'Label:' else first line.
    """
    raw = re.sub(r"<\|.*?\|>", "", str(gen_text)).strip()

    if "Label:" in raw:
        cand = raw.split("Label:", 1)[-1].strip()
    else:
        cand = raw

    cand = cand.splitlines()[0].strip().rstrip(".:;").strip()
    return cand


def normalize_to_label_set(candidate: str, labels: list[str], aliases: dict[str, str]) -> str:
    """
    Map candidate text to canonical label using aliases and substring matches.
    """
    key = candidate.lower()

    # exact alias match
    if key in aliases:
        return aliases[key]

    # common ordering issue (Not Abusive vs Abusive)
    if key.startswith("not abusive") and "not abusive" in aliases:
        return aliases["not abusive"]
    if key.startswith("abusive") and "abusive" in aliases:
        return aliases["abusive"]

    # substring match against canonical labels
    for lab in labels:
        if lab.lower() in key:
            return lab

    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", required=True, help="Path to Lughaat model YAML config")
    parser.add_argument("--data", required=True, help="CSV path with columns: sentences,label")
    parser.add_argument("--prompt", required=True, help="Prompt template .txt with {sentence}")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    # NEW: label-set selection
    parser.add_argument("--label_cfg", default="configs/label_sets.yaml",
                        help="Path to label_sets.yaml")
    parser.add_argument("--label_set", required=True,
                        help="Label set name (e.g., task2_fiveclass_roman, coarse_2class_urdu, ruhsold_fine_en)")

    # HF login optional
    parser.add_argument("--hf_login", action="store_true",
                        help="Call huggingface_hub.login() interactively")

    args = parser.parse_args()

    if args.hf_login:
        login()
    else:
        try:
            login()
        except Exception:
            pass

    cfg = load_yaml(Path(args.model_cfg))
    labels, aliases = load_label_set(Path(args.label_cfg), args.label_set)

    model_id = cfg["model_id"]
    max_new_tokens = int(cfg.get("max_new_tokens", 150))
    return_full_text = bool(cfg.get("return_full_text", True))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=torch.float16
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=return_full_text
    )

    df = pd.read_csv(args.data)
    df = df.iloc[args.start:args.end].reset_index(drop=True)

    prompt_template = load_prompt(Path(args.prompt))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        with out_path.open("w", encoding="utf-8") as f:
            f.write("Index,sentences,Predicted,label\n")

    for idx, row in df.iterrows():
        try:
            sentence = str(row["sentences"])
            gold = str(row.get("label", ""))

            prompt = prompt_template.format(sentence=sentence)

            out = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
            gen_text = out[0]["generated_text"]

            candidate = extract_label_from_text(gen_text)
            pred = normalize_to_label_set(candidate, labels, aliases)

            sent_clean = sentence.replace('"', '""')
            pred_clean = pred.replace('"', '""')
            gold_clean = gold.replace('"', '""')

            with out_path.open("a", encoding="utf-8") as f:
                f.write(f'{idx},"{sent_clean}","{pred_clean}","{gold_clean}"\n')

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(df)}")

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"❌ Error at index {idx}: {e}")

    print(f"✅ Done. Saved: {out_path}")


if __name__ == "__main__":
    main()
