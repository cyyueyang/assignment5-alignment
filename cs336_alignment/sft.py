import json
import random

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

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算概率分布的熵，用于衡量模型预测的不确定性
    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
    Returns:
        torch.Tensor of shape (batch_size, seq_len)
    """
    log_logits = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_logits)
    return -torch.sum(probs * torch.log(probs), dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    ans = {}

    logits = model(input_ids).logits   # [batchsize, seq_len, vocab_size]
    ans["log_probs"] = torch.log(torch.gather(logits.softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
    if return_token_entropy:
        ans["token_entropy"] = compute_entropy(logits)

    return ans

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return torch.sum(tensor.masked_fill(~mask,0), dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    loss = -masked_normalize(policy_log_probs, response_mask, -1, normalize_constant).mean() / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "loss": loss.detach().cpu(),
        "policy_log_probs_mean": policy_log_probs.mean().detach().cpu(),
        "policy_log_probs_std": policy_log_probs.std().detach().cpu(),
        "num_masked_tokens": response_mask.sum().item(),
        "normalize_constant": normalize_constant,
    }

    return (loss, metadata)

class PackedSFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_path, seg_len, shuffle=True):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.seg_len = seg_len

        import json
        import random
        with open(dataset_path, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f]
        if shuffle:
            random.shuffle(examples)

        # Alpaca模板 输入 (prompt, response) 开始输出 response
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"

        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id

        # 将文档合并为单一序列
        self.all_token_ids = []

        for i, example in enumerate(examples):
            # 1. 格式化文本
            formatted_text = template.format(instruction=example["prompt"], response=example["response"])
            # 2，tokenization
            tokens = self.tokenizer(formatted_text, add_special_tokens=False)["input_ids"]

            # 添加 bos
            if bos_token_id is not None:
                self.all_token_ids.append(bos_token_id)

            self.all_token_ids.extend(tokens)

            # 添加 eos
            if eos_token_id is not None:
                self.all_token_ids.append(eos_token_id)

            self.sequences = []

        total_tokens = len(self.all_token_ids)

        i = 0
        while i + self.seg_len <= total_tokens:
            chunk = self.all_token_ids[i:i+self.seg_len]
            self.sequences.append(chunk)
            i += self.seg_len

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            input_ids = torch.tensor(sequence, dtype=torch.long)

            if idx < len(self.sequences) - 1:
                next_sequence = self.sequences[idx+1]
                labels = torch.tensor(sequence[1:] + [next_sequence[0]], dtype=torch.long)
            else:
                start_pos = idx * self.seg_len
                end_pos = start_pos + self.seg_len

                if end_pos < len(self.all_token_ids):
                    next_token = self.all_token_ids[end_pos]
                    labels = torch.tensor(sequence[1:] + [next_token], dtype=torch.long)
                else:
                    labels = torch.tensor(sequence[1:] + [-100], dtype=torch.long)

            return {
                "input_ids": input_ids,
                "labels": labels,
            }

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_path):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        responses = example["responses"]

        return tokenize_prompt_and_output(prompt, responses, self.tokenizer)



