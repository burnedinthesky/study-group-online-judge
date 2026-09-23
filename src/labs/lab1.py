import torch


def gpt2_complete(
    input: list[str],
    max_seq_length: int = 1024,
) -> tuple[list[str], torch.Tensor]:
    raise ValueError("Intentional Lab 1 failure for judge smoke test")
