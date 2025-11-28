import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """

    ans = {}
    prompt_tokenized = tokenizer(prompt_strs)["input_ids"]
    output_tokenized = tokenizer(output_strs)["input_ids"]
    tokenized = [p + o for p, o in zip(prompt_tokenized, output_tokenized)]
    max_len = max(len(t) for t in tokenized) - 1
    bs = len(tokenized)

    input_ids = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.zeros((bs, max_len), dtype=torch.long)
    response_mask = torch.zeros((bs, max_len), dtype=torch.bool)

    for i, tokens in enumerate(tokenized):
        input_ids[i, :len(tokens)-1] = torch.tensor(tokens[:-1])
        labels[i, :len(tokens)-1] = torch.tensor(tokens[1:])
        if len(tokens) < max_len:
            labels[i, len(tokens)-1:] = tokenizer.eos_token_id
        response_mask[i, len(prompt_tokenized[i])-1:len(tokens)-1] = True

    last_col_idx = max_len - 1
    mask = input_ids[:, last_col_idx] == 0
    input_ids[mask, last_col_idx] = tokenizer.eos_token_id

    ans["input_ids"] = input_ids
    ans["labels"] = labels
    ans["response_mask"] = response_mask

    return ans

