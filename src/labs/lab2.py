"""Toy Lab 2 submission using Hugging Face's GPT-2 rather than Lab 1's model."""

import hashlib
import json
import sys
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LETTERS = "ABCD"
BATCH_SIZE = 16


def _question_key(index: int, row: dict) -> str:
    payload = {
        "index": index,
        "subject": row["subject"],
        "question": row["question"],
        "choices": row["choices"],
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _format_question(row: dict, answer: str | None = None) -> str:
    options = " ".join(
        f"({letter}) {choice}"
        for letter, choice in zip(LETTERS, row["choices"], strict=True)
    )
    return f"{row['question']}\n{options}\nAnswer: {answer or ''}"


def _prompt(row: dict, exemplars: list[dict]) -> str:
    subject = row["subject"].replace("_", " ")
    examples = [
        _format_question(example, LETTERS[example["answer"]]) for example in exemplars
    ]
    return (
        f"The following are multiple choice questions about {subject}.\n\n"
        + "\n\n".join([*examples, _format_question(row)])
    )


def mmlu_eval() -> dict[str, str]:
    print("[lab2 sample] loading pinned MMLU dev and test splits", flush=True)
    dev = load_dataset("cais/mmlu", "all", split="dev", revision=DATASET_REVISION)
    test = load_dataset("cais/mmlu", "all", split="test", revision=DATASET_REVISION)
    exemplars: dict[str, list[dict]] = defaultdict(list)
    for row in dev:
        if len(exemplars[row["subject"]]) < 4:
            exemplars[row["subject"]].append(row)

    print("[lab2 sample] loading Hugging Face GPT-2 Small", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=dtype)
    torch.nn.Module.to(model, device=device)
    model.eval()
    token_ids = [
        tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS
    ]

    answers: dict[str, str] = {}
    print(f"[lab2 sample] evaluating {len(test)} questions on {device}", flush=True)
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(test), BATCH_SIZE),
            desc="[lab2 sample] GPT-2 inference",
            file=sys.stdout,
            mininterval=5,
        ):
            rows = [
                test[index]
                for index in range(start, min(start + BATCH_SIZE, len(test)))
            ]
            prompts = [_prompt(row, exemplars[row["subject"]]) for row in rows]
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=model.config.n_positions,
                return_tensors="pt",
            ).to(device)
            positions = (inputs["attention_mask"].cumsum(dim=1) - 1).clamp_min(0)
            logits = model(**inputs, position_ids=positions).logits[:, -1, :]
            choices = logits[:, token_ids].argmax(dim=-1).tolist()
            for offset, choice in enumerate(choices):
                answers[_question_key(start + offset, rows[offset])] = LETTERS[choice]

    print(f"[lab2 sample] completed {len(answers)} predictions", flush=True)
    return answers
