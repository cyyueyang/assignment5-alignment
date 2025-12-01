import torch
from torch import Tensor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Callable, Optional

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

