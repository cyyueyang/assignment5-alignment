import torch
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Callable, Optional, Iterable, Tuple, Union, Literal
import torch.nn.functional as F

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
    rewards = [reward_fn(rollout_responses[i], repeated_ground_truths[i])["reward"] for i in range(rollout_responses_size)]
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
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

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

        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

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

        loss, clip_matadata = compute_grpo_clip_loss(
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

    losses, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    masked_loss = masked_mean(losses, response_mask, dim=-1)
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
        logits_lm_chosen = lm(tokens_chosen.unsqueeze(0)).logits.squeeze(0)  # [1, seq_len]->[1, seq_len, vocab_size]->[seq_len, vocab_size]
        logits_lm_rejected = lm(tokens_rejected.unsqueeze(0)).logits.squeeze(0)
        logits_ref_chosen = lm_ref(tokens_chosen.unsqueeze(0)).logits.squeeze(0)
        logits_ref_rejected = lm_ref(tokens_rejected.unsqueeze(0)).logits.squeeze(0)

    log_probs_lm_chosen = torch.log_softmax(logits_lm_chosen, dim=-1)
    log_probs_lm_rejected = torch.log_softmax(logits_lm_rejected, dim=-1)
    log_probs_ref_chosen = torch.log_softmax(logits_ref_chosen, dim=-1)
    log_probs_ref_rejected = torch.log_softmax(logits_ref_rejected, dim=-1)

    target_chosen_ids = tokens_chosen[1:]
    target_rejected_ids = tokens_rejected[1:]

    log_probs_lm_chosen = torch.gather(log_probs_lm_chosen, dim=-1, index=target_chosen_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_ref_chosen = torch.gather(log_probs_ref_chosen, dim=-1, index=target_chosen_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_lm_rejected = torch.gather(log_probs_lm_rejected, dim=-1, index=target_rejected_ids.unsqueeze(-1)).squeeze(-1)
    log_probs_ref_rejected = torch.gather(logits_ref_rejected, dim=-1, index=target_rejected_ids.unsqueeze(-1)).squeeze(-1)

    log_probs_lm_chosen = log_probs_lm_chosen.sum()
    log_probs_lm_rejected = log_probs_lm_rejected.sum()
    log_probs_ref_chosen = log_probs_ref_chosen.sum()
    log_probs_ref_rejected = log_probs_ref_rejected.sum()

    log_ratio_chosen = log_probs_lm_chosen - log_probs_ref_chosen
    log_ratio_rejected = log_probs_lm_rejected - log_probs_ref_rejected

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))

    return loss