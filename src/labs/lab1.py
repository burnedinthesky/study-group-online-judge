import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", dtype=torch.float16
    ).eval()

    prompt_ids = [tokenizer.encode(prompt) for prompt in input]
    lengths = [len(ids) for ids in prompt_ids]
    width = max(lengths)
    eos_id = tokenizer.eos_token_id
    input_ids = torch.full((len(input), width), eos_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(prompt_ids):
        input_ids[index, -len(ids) :] = torch.tensor(ids)
        attention_mask[index, -len(ids) :] = 1

    generated: list[list[int]] = [[] for _ in input]
    finished = torch.tensor([length >= max_seq_length for length in lengths])
    step_logits: list[torch.Tensor] = []

    with torch.inference_mode():
        while not bool(finished.all()):
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)
            logits = (
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                .logits[:, -1, :]
                .clone()
            )
            active = ~finished
            logits[~active] = 0
            next_ids = logits.argmax(dim=-1)
            next_ids[~active] = eos_id
            step_logits.append(logits)

            for index in range(len(input)):
                if active[index]:
                    token = int(next_ids[index])
                    generated[index].append(token)
                    lengths[index] += 1
                    if token == eos_id or lengths[index] >= max_seq_length:
                        finished[index] = True

            input_ids = torch.cat((input_ids, next_ids[:, None]), dim=1)
            attention_mask = torch.cat(
                (attention_mask, active[:, None].to(dtype=attention_mask.dtype)), dim=1
            )

    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    logits = (
        torch.stack(step_logits, dim=1)
        if step_logits
        else torch.empty((len(input), 0, model.config.vocab_size), dtype=torch.float16)
    )
    return completions, logits
