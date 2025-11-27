from vllm import LLM, SamplingParams
from typing import Callable, List
import json
import os

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    with open("/data/a5-alignment/MATH/validation.json") as f:
        validation_data = json.load(f)

