# Urdu Hate Speech ICL Pipeline

This repository provides a unified, extensible pipeline for running large language models (LLMs) on Urdu hate-speech classification tasks under consistent prompting and evaluation settings.

---

## Key Features

- **Multiple prompt languages**
  - English
  - Roman Urdu
  - Urdu Script

- **Prompting strategies**
  - Zero-shot
  - Few-shot (ICL)

- **Model families**
  - LLaMA (e.g., 8B, 70B; later 1B/3B)
  - Qwen (e.g., 7B, 72B)
  - Lughaat (8B)

- **Config-driven and reproducible**
  - Models configured via YAML (`models/`)
  - Prompts stored as `.txt` templates (`prompts/`)
  - Label taxonomies selected via YAML (`configs/label_sets.yaml`)
  - Outputs saved incrementally to CSV (`outputs/`)

---
---

## Installation

```bash
conda create -n urdu-hate python=3.10
conda activate urdu-hate
pip install -r requirements.txt
huggingface-cli login

Example: LLaMA (Roman Urdu, Fine-Grained, Zero-shot)

python runners/run_llama.py \
  --model-config models/llama/llama_8b.yaml \
  --data data/dataset1/hate_speech_task_2.csv \
  --prompt prompts/roman_urdu/zero_shot_fine.txt \
  --output outputs/llama/llama8b/task2_roman_urdu_zero_shot.csv \
  --label_set task2_fiveclass_romane

## Citation

~~~
This code is the official implementation of the following paper. Please cite the paper below when using it.

Iffat Maab, Usman Haider, and Junichi Yamagishi. 2026. 
Prompt-driven Detection of Offensive Urdu Language using Large Language Models.
In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4302–4327, Rabat, Morocco. Association for Computational Linguistics.

Link: https://aclanthology.org/2026.eacl-long.201/
~~~
