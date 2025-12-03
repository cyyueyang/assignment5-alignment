from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader, Dataset

def run_tokenize_prompt_and_output(
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
            input_ids[i, :len(tokens) - 1] = torch.tensor(tokens[:-1])
            labels[i, :len(tokens) - 1] = torch.tensor(tokens[1:])
            if len(tokens) < max_len:
                labels[i, len(tokens) - 1:] = tokenizer.eos_token_id
            response_mask[i, len(prompt_tokenized[i]) - 1:len(tokens) - 1] = True

        last_col_idx = max_len - 1
        mask = input_ids[:, last_col_idx] == 0
        input_ids[mask, last_col_idx] = tokenizer.eos_token_id

        ans["input_ids"] = input_ids
        ans["labels"] = labels
        ans["response_mask"] = response_mask

        return ans

    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """

    def compute_group_normalized_rewards(
            reward_fn: Callable,
            rollout_responses: list[str],
            repeated_ground_truths: list[str],
            group_size: int,
            advantage_eps: float,
            normalize_by_std: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute rewards for each group of rollout responses,
        normalized by the group size.

        For more on GRPO, see:
            DeepSeekMath: https://arxiv.org/abs/2402.03300
            DeepSeek-R1: https://arxiv.org/abs/2501.12948

        Args:
            reward_fn: Callable[[str, str], dict[str, float]],
                scores the rollout responses against the ground truths,
                producing a dict with keys
                "reward", "format_reward", and "answer_reward".
            rollout_responses: list[str], rollouts from the policy.
                The length of this list is
                `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
            repeated_ground_truths: list[str], the ground truths for the examples.
                The length of this list is `rollout_batch_size`,
                because the ground truth for each example is repeated `group_size` times.
            group_size: int, number of rollouts per group.
            advantage_eps: float, epsilon to avoid division by zero
                during group normalization.
            normalize_by_std: bool, whether to normalize the rewards by
                std(rewards).

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
                torch.Tensor of shape (rollout_batch_size,):
                    group-normalized rewards for each rollout response.
                torch.Tensor of shape (rollout_batch_size,):
                    raw rewards for each rollout response.
                dict[str, float]: metadata for the rewards of the rollout batch.
                    You may choose what you wish to log here
                    (some statistics of the rewards, etc.).
        """

        metadata = {}

        rollout_responses_size = len(rollout_responses)
        rewards = [reward_fn(rollout_responses[i], repeated_ground_truths[i])["reward"] for i in
                   range(rollout_responses_size)]
        raw_rewards = torch.tensor(rewards)

        metadata["raw_rewards_mean"] = float(torch.mean(raw_rewards))
        metadata["raw_rewards_std"] = float(torch.std(raw_rewards))
        metadata["raw_rewards_min"] = float(torch.min(raw_rewards))
        metadata["raw_rewards_max"] = float(torch.max(raw_rewards))

        advantages = raw_rewards.reshape(-1, group_size).clone()
        if normalize_by_std:
            advantages = (advantages - advantages.mean(dim=-1, keepdim=True)) / (advantages.std(dim=-1, keepdim=True) + advantage_eps)
        else:
            advantages = advantages - advantages.mean(dim=-1, keepdim=True)

        normalized_rewards = advantages.reshape(-1)

        metadata["normalized_rewards_mean"] = float(torch.mean(normalized_rewards))
        metadata["normalized_rewards_std"] = float(torch.std(normalized_rewards))
        metadata["normalized_rewards_min"] = float(torch.min(normalized_rewards))
        metadata["normalized_rewards_max"] = float(torch.max(normalized_rewards))

        return normalized_rewards, raw_rewards, metadata

    return compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std
    )


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""

    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        log_logits = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_logits)
        return -torch.sum(probs * torch.log(probs), dim=-1)

    return compute_entropy(logits)


def run_get_response_log_probs(
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

        logits = model(input_ids).logits  # [batchsize, seq_len, vocab_size]
        ans["log_probs"] = torch.log(
            torch.gather(logits.softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
        if return_token_entropy:
            ans["token_entropy"] = run_compute_entropy(logits)

        return ans
    return get_response_log_probs(model, input_ids, labels, return_token_entropy)

def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """

    def compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages: torch.Tensor,
            policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy gradient loss using either raw rewards or advantages.

        Args:
            raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
                the raw rewards or advantages for each rollout response.
            policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
                the log-probs of the policy.

        Returns:
            torch.Tensor of shape (batch_size, sequence_length):
                the policy gradient per-token loss.
        """
        seq_len = policy_log_probs.size()[-1]
        raw_rewards_or_advantages = raw_rewards_or_advantages.expand(-1, seq_len)
        return -raw_rewards_or_advantages * policy_log_probs

    return compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """

    def compute_grpo_clip_loss(
            advantages: torch.Tensor,
            policy_log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the GRPO-Clip loss.

        Args:
            advantages: torch.Tensor of shape (batch_size, 1):
                the advantages for each rollout response.
            policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
                the log-probs of the policy.
            old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
                the log-probs of the old policy.
            cliprange: float, the clip range for the ratio.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                torch.Tensor of shape (batch_size, sequence_length):
                    the GRPO-Clip per-token loss.
                dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                    (used to compute clip fraction).
        """
        metadata = {}
        seq_len = policy_log_probs.size()[-1]
        advantages = advantages.expand(-1, seq_len)

        ratios = torch.exp(policy_log_probs - old_log_probs)
        ratios_clipped = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
        unclipped_loss = ratios * advantages
        clipped_loss = ratios_clipped * advantages

        clipped_flag = (ratios < 1 - cliprange) | (ratios > 1 + cliprange)

        metadata["ratios_mean"] = float(torch.mean(ratios).item())
        metadata["ratios_std"] = float(torch.std(ratios).item())
        metadata["ratios_min"] = float(torch.min(ratios).item())
        metadata["ratios_max"] = float(torch.max(ratios).item())

        metadata["clip_fraction"] = float(clipped_flag.float().mean())
        metadata["num_clipped_tokens"] = int(clipped_flag.float().sum().item())
        metadata["total_tokens"] = int(clipped_flag.numel())

        metadata["unclipped_loss_mean"] = float(torch.mean(unclipped_loss).item())
        metadata["clipped_loss_mean"] = float(torch.mean(clipped_loss).item())

        loss = -torch.minimum(unclipped_loss, clipped_loss)

        metadata["final_loss_mean"] = float(torch.mean(loss).item())

        return loss, metadata

    return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """

    def compute_policy_gradient_loss(
            policy_log_probs: torch.Tensor,
            loss_type: str,
            raw_rewards: torch.Tensor,
            advantages: torch.Tensor,
            old_log_probs: torch.Tensor,
            cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Wrapper that delegates to the appropriate policy gradient loss function above.
        args:
            policy_log_probs：形状为 (batch_size, sequence_length)，当前策略给出的逐 token 对数概率。
            loss_type：可选值为 "no_baseline"、"reinforce_with_baseline" 或 "grpo_clip"。
            raw_rewards：当 loss_type == "no_baseline" 时必须提供；形状为 (batch_size, 1)。
            advantages：当 loss_type 为 "reinforce_with_baseline" 或 "grpo_clip" 时必须提供；形状为 (batch_size, 1)。
            old_log_probs：当 loss_type 为 "grpo_clip" 时必须提供；形状为 (batch_size, sequence_length)。
            cliprange：当 loss_type 为 "grpo_clip" 时必须提供；用于剪切标量 ε。
        Returns:
            元组 (torch.Tensor, dict[str, torch.Tensor])：
            loss：形状为 (batch_size, sequence_length) 的逐 token 损失。
            metadata：字典，包含底层例程返回的统计信息（例如 GRPO-Clip 的剪切比例）。
        """

        metadata = {}

        if loss_type == "no_baseline":
            assert raw_rewards is not None
            loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

            metadata["raw_rewards_mean"] = float(torch.mean(raw_rewards).item())
            metadata["raw_rewards_std"] = float(torch.std(raw_rewards).item())
            metadata["raw_rewards_min"] = float(torch.min(raw_rewards).item())
            metadata["raw_rewards_max"] = float(torch.max(raw_rewards).item())
            metadata["loss_mean"] = float(torch.mean(loss).item())
            metadata["loss_std"] = float(torch.std(loss).item())
            metadata["loss_min"] = float(torch.min(loss).item())
            metadata["loss_max"] = float(torch.max(loss).item())

        elif loss_type == "reinforce_with_baseline":
            assert advantages is not None

            loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)

            metadata["advantages_mean"] = float(torch.mean(advantages).item())
            metadata["advantages_std"] = float(torch.std(advantages).item())
            metadata["advantages_min"] = float(torch.min(advantages).item())
            metadata["advantages_max"] = float(torch.max(advantages).item())
            metadata["loss_mean"] = float(torch.mean(loss).item())
            metadata["loss_std"] = float(torch.std(loss).item())
            metadata["loss_min"] = float(torch.min(loss).item())
            metadata["loss_max"] = float(torch.max(loss).item())

        elif loss_type == "grpo_clip":
            assert advantages is not None
            assert old_log_probs is not None
            assert cliprange is not None

            loss, clip_matadata = run_compute_grpo_clip_loss(
                advantages, policy_log_probs, old_log_probs, cliprange
            )
            metadata.update(clip_matadata)

            metadata["advantages_mean"] = float(torch.mean(advantages).item())
            metadata["advantages_std"] = float(torch.std(advantages).item())
            metadata["advantages_min"] = float(torch.min(advantages).item())
            metadata["advantages_max"] = float(torch.max(advantages).item())
            metadata["loss_mean"] = float(torch.mean(loss).item())
            metadata["loss_std"] = float(torch.std(loss).item())
            metadata["loss_min"] = float(torch.min(loss).item())
            metadata["loss_max"] = float(torch.max(loss).item())

        else:
            raise ValueError("Unknown loss type")

        return loss, metadata
    return compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)

def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """

    def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
        """Compute the mean of the tensor along a dimension,
        considering only the elements with mask value 1.

        Args:
            tensor: torch.Tensor, the tensor to compute the mean of.
            mask: torch.Tensor, the mask. We only take the mean over
                the elements with mask value 1.
            dim: int | None, the dimension to compute the mean along.
                If None, sum over all non-masked elements and average
                by their total count.

        Returns:
            torch.Tensor, the mean of the tensor along the specified
                dimension, considering only the elements with mask value 1.
        """
        mask_count = torch.count_nonzero(mask, dim=dim)

        return torch.sum(tensor.masked_fill(~mask.bool(), 0.0), dim=dim) / mask_count

    return masked_mean(tensor, mask, dim)

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """

    def sft_microbatch_train_step(
            policy_log_probs: torch.Tensor,
            response_mask: torch.Tensor,
            gradient_accumulation_steps: int,
            normalize_constant: int | None = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the policy gradient loss and backprop its gradients for a microbatch.
        """
        loss = -run_masked_normalize(policy_log_probs, response_mask, -1,
                                 normalize_constant).mean() / gradient_accumulation_steps
        loss.backward()

        metadata = {
            "loss": loss.detach().cpu(),
            "policy_log_probs_mean": policy_log_probs.mean().detach().cpu(),
            "policy_log_probs_std": policy_log_probs.std().detach().cpu(),
            "num_masked_tokens": response_mask.sum().item(),
            "normalize_constant": normalize_constant,
        }

        return (loss, metadata)
    return sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant)

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """

    def grpo_microbatch_train_step(
            policy_log_probs: torch.Tensor,
            response_mask: torch.Tensor,
            gradient_accumulation_steps: int,
            loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            raw_rewards: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            old_log_probs: torch.Tensor | None = None,
            cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the policy gradient loss and backprop its gradients for a microbatch.

        Args:
            policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
                the log-probs of the policy.
            response_mask: torch.Tensor of shape (batch_size, sequence_length):
                the mask for the response.
            gradient_accumulation_steps: int, the number of gradient accumulation steps.
            loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                the type of loss function to use.
            raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
                Needed for loss_type="no_baseline".
            advantages: torch.Tensor | None, the advantages for each rollout response.
                Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
            old_log_probs: torch.Tensor | None, the log-probs of the old policy.
                Needed for loss_type="grpo_clip".
            cliprange: float | None, the clip range for the ratio.
                Needed for loss_type="grpo_clip".
            constant_normalize_factor: int | None, provided if we want to sum over
                the sequence dimension and normalize by this constant factor
                (as in Dr. GRPO).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                the policy gradient loss and its metadata.
        """

        losses, loss_metadata = run_compute_policy_gradient_loss(
            policy_log_probs=policy_log_probs,
            loss_type=loss_type,
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

        masked_loss = run_masked_mean(losses, response_mask, dim=-1)
        loss = masked_loss.mean(dim=0) / gradient_accumulation_steps

        metadata = {}
        metadata.update(loss_metadata)

        metadata["microbatch_loss"] = float(torch.mean(loss).item())
        metadata["masked_losses_mean"] = float(masked_loss.mean().item())
        metadata["masked_losses_std"] = float(masked_loss.std().item())
        metadata["masked_losses_min"] = float(masked_loss.min().item())
        metadata["masked_losses_max"] = float(masked_loss.max().item())

        metadata["sequence_length"] = int(policy_log_probs.shape[-1])
        metadata["batch_size"] = int(policy_log_probs.shape[0])
        metadata["response_mask_sum"] = int(response_mask.sum().item())
        metadata["response_mask_mean"] = float(response_mask.float().mean().item())

        metadata["gradient_accumulation_steps"] = gradient_accumulation_steps
        metadata["effective_batch_size"] = int(policy_log_probs.shape[0] * gradient_accumulation_steps)

        loss.backward()

        return loss, metadata

    return grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
        )


def run_masked_normalize(
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
        return torch.sum(tensor.masked_fill(~mask, 0), dim=dim) / normalize_constant
    return masked_normalize(tensor, mask, dim, normalize_constant)


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """

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
                chunk = self.all_token_ids[i:i + self.seg_len]
                self.sequences.append(chunk)
                i += self.seg_len

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            input_ids = torch.tensor(sequence, dtype=torch.long)

            if idx < len(self.sequences) - 1:
                next_sequence = self.sequences[idx + 1]
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
    return PackedSFTDataset(tokenizer, dataset_path, seg_len=seq_length, shuffle=shuffle)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """

    def iterate_batches(
            dataset: Dataset,
            batch_size: int,
            shuffle: bool,
    ):
        """
        Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
        Iterating through the returned iterable should constitute one epoch over the Dataset.

        Args:
            dataset: Dataset
                Dataset to emit batches from.
            batch_size: int
                Number of examples to include per batch.
            shuffle: bool
                If true, shuffle examples before batching them.

        Returns:
            Iterable over batches, where each batch has size `batch_size`.
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return iterate_batches(dataset, batch_size, shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    import re
    def parse_mmlu_response(
            mmlu_example: dict[str, Any],
            model_output: str,
    ) -> str | None:
        """
        Given an MMLU example and a model output, parse the model output into a
        predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
        cannot be parsed into a prediction option letter, return None.

        mmlu_example: dict[str, Any]
            Dictionary with an MMLU example. Contains the following keys:
            - "subject": str with the subject of the question.
            - "question": str with the text of the question.
            - "options": list[str] with the four answer options (in order).
                         The first option refers to letter "A", the second to "B", etc.
            - "answer": str with the option of the correct answer (e.g., "A")
        model_output: str
            str with the model's output to the MMLU example.

        Returns:
            str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
            else None.
        """
        pattern = r"The correct answer is\s+([ABCD])"
        match = re.search(pattern, model_output, re.IGNORECASE)

        if match:
            return match.group(1).upper()

        return None

    return parse_mmlu_response(mmlu_example, model_output)


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """

    def parse_gsm8k_response(
            model_output: str,
    ) -> str | None:
        """
        Given a GSM8K model output, parse the model output into a predicted numeric answer by
        taking the last number that occurs in the output.

        model_output: str
            str with the model's output to a GSM8K example.

        Returns:
            str with the predicted numeric answer if the model output can be parsed into a prediction,
            else None.
        """
        import re
        pattern = r'\d+(?:\.\d+)?'
        matches = re.findall(pattern, model_output)

        # 如果找到数字，返回最后一个
        if matches:
            return matches[-1]

        return None

    return parse_gsm8k_response(model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """

    def compute_per_instance_dpo_loss(
            lm: torch.nn.Module,
            lm_ref: torch.nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            beta: float,
            prompt: str,
            response_chosen: str,
            response_rejected: str,
    ) -> torch.Tensor:
        """
        Given two language models (`lm`, and the "reference model" `lm_ref`),
        their tokenizer, the DPO beta hyperparameter, a prompt and a pair
        of responses to the prompt, computes the value of the DPO loss for this example.

        lm: torch.nn.Module
            Language model being trained.
        lm_ref: torch.nn.Module
            Reference language model.
        tokenizer: PreTrainedTokenizerBase
            Tokenizer for both language models.
        beta: float
            DPO beta hyperparameter.
        prompt: str
            Prompt for this instance of preference pair.
        response_chosen: str
            Preferred response to the prompt.
        response_rejected: str
            Rejected response to the prompt.

        Returns:
            torch.Tensor with the DPO loss for this example.
        """
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
        lm.eval()
        lm_ref.eval()

        eos_token = tokenizer.eos_token
        if isinstance(eos_token, list):
            eos_token = eos_token[0]

        text_chosen = template.format(instruction=prompt, response=response_chosen) + eos_token
        text_rejected = template.format(instruction=prompt, response=response_rejected) + eos_token

        tokens_chosen = tokenizer(text_chosen, return_tensors="pt", add_special_tokens=False)["input_ids"]
        tokens_rejected = tokenizer(text_rejected, return_tensors="pt", add_special_tokens=False)["input_ids"]

        if tokens_chosen.dim > 1:
            token_chosen = tokens_chosen.squeeze(0)  # [seq_len]
        if tokens_rejected.dim > 1:
            tokens_rejected = tokens_rejected.squeeze(0)

        with torch.no_grad():
            logits_lm_chosen = lm(tokens_chosen.unsqueeze(0)).logits.squeeze(
                0)  # [1, seq_len]->[1, seq_len, vocab_size]->[seq_len, vocab_size]
            logits_lm_rejected = lm(tokens_rejected.unsqueeze(0)).logits.squeeze(0)
            logits_ref_chosen = lm_ref(tokens_chosen.unsqueeze(0)).logits.squeeze(0)
            logits_ref_rejected = lm_ref(tokens_rejected.unsqueeze(0)).logits.squeeze(0)

        log_probs_lm_chosen = torch.log_softmax(logits_lm_chosen, dim=-1)
        log_probs_lm_rejected = torch.log_softmax(logits_lm_rejected, dim=-1)
        log_probs_ref_chosen = torch.log_softmax(logits_ref_chosen, dim=-1)
        log_probs_ref_rejected = torch.log_softmax(logits_ref_rejected, dim=-1)

        target_chosen_ids = tokens_chosen[1:]
        target_rejected_ids = tokens_rejected[1:]

        log_probs_lm_chosen = torch.gather(log_probs_lm_chosen, dim=-1, index=target_chosen_ids.unsqueeze(-1)).squeeze(
            -1)
        log_probs_ref_chosen = torch.gather(log_probs_ref_chosen, dim=-1,
                                            index=target_chosen_ids.unsqueeze(-1)).squeeze(-1)
        log_probs_lm_rejected = torch.gather(log_probs_lm_rejected, dim=-1,
                                             index=target_rejected_ids.unsqueeze(-1)).squeeze(-1)
        log_probs_ref_rejected = torch.gather(logits_ref_rejected, dim=-1,
                                              index=target_rejected_ids.unsqueeze(-1)).squeeze(-1)

        log_probs_lm_chosen = log_probs_lm_chosen.sum()
        log_probs_lm_rejected = log_probs_lm_rejected.sum()
        log_probs_ref_chosen = log_probs_ref_chosen.sum()
        log_probs_ref_rejected = log_probs_ref_rejected.sum()

        log_ratio_chosen = log_probs_lm_chosen - log_probs_ref_chosen
        log_ratio_rejected = log_probs_lm_rejected - log_probs_ref_rejected
        import torch.nn.functional as F
        loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

        return loss

    return compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected)
