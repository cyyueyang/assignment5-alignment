from vllm import LLM, SamplingParams
from typing import Callable, List
import json
import os
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import re

def load_data(data_file: str):
    validation_data = []

    with open(data_file, "r") as f:
        for line in f:
            data = json.loads(line)
            validation_data.append(data)
    return validation_data

def format_r1_zero_prompt(question: str) -> str:
    prompt_template = """{question}"""

    return prompt_template.format(question=question)

# def extract_final_answer(text: str) -> str:
#     answer_match = re.search(r"<answer>(.+?)</answer>", text, re.DOTALL)
#     if answer_match:
#         return answer_match.group(1).strip()
#     return ""

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    print(f"Generating Responses for {len(prompts)} examples")

    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    results = []
    correct_count = 0
    format_correct_count = 0
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    print(f"Evaluating responses for {len(prompts)} examples")
    for i, output in enumerate(tqdm(outputs)):
        generated_text = output.outputs[0].text
        # answer = extract_final_answer(generated_text)
        print("======================================8888888888888888")
        print(generated_text)
        print("======================================8888888888888888")
        ground_truth = ground_truths[i]
        reward_result = reward_fn(generated_text, ground_truth)
        result = {
            "example_id": i,
            "prompt": output.prompt,
            "generation": generated_text,
            "ground_truth": ground_truth,
            "format_reward": reward_result["format_reward"],
            "answer_reward": reward_result["answer_reward"],
            "reward": reward_result["reward"],
            "is_correct": reward_result["reward"] == 1.0,
            "format_correct": reward_result["format_reward"] == 1.0
        }
        results.append(result)

        if reward_result["answer_reward"] == 1.0:
            correct_count += 1
        if reward_result["format_reward"] == 1.0:
            format_correct_count += 1

        total_reward += reward_result["reward"]
        total_format_reward += reward_result["format_reward"]
        total_answer_reward += reward_result["answer_reward"]

    accuracy = correct_count / len(results)
    format_accuracy = format_correct_count / len(results)
    avg_reward = total_reward / len(results)
    avg_format_reward = total_format_reward / len(results)
    avg_answer_reward = total_answer_reward / len(results)

    metrics = {
        "total_examples": len(results),
        "correct_count": correct_count,
        "format_correct_count": format_correct_count,
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "average_reward": avg_reward,
        "average_format_reward": avg_format_reward,
        "average_answer_reward": avg_answer_reward
    }

    # 保存结果
    # output_data = {
    #     "model_name": "Qwen2.5-Math-1.5B",
    #     "evaluation_set": "MATH-validation",
    #     "metrics": metrics,
    #     "results": results
    # }
    #
    # print(output_data)
    print("="*80)
    print(f"avg_format_reward: {avg_format_reward}")
    print(f"avg_answer_reward: {avg_answer_reward}")
    print("="*80)

if __name__ == '__main__':
    MATH_VALIDATION_PATH = "../data/gsm8k/test.jsonl"
    MODEL_NAME = "/home/cyyang/Qwen/Qwen2.5-Math-1.5B"

    print("Loading MATH validation data...")
    examples = load_data(MATH_VALIDATION_PATH)
    print(f"Loaded {len(examples)} examples")

    print("Formatting prompts and extracting ground truths...")
    prompts = []
    ground_truths = []

    for example in examples:
        question = example.get("question", "")
        answer = example.get("answer", "")

        prompt = format_r1_zero_prompt(question)
        prompts.append(prompt)
        ground_truths.append(answer)

    # 3. 初始化vLLM模型
    print(f"Loading model: {MODEL_NAME}")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,  # 根据GPU数量调整
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=2048  # 确保有足够的长度
    )


    # 4. 配置采样参数
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.90,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 5. 执行评估
    print("Starting evaluation...")
    evaluate_vllm(llm, reward_fn=question_only_reward_fn, prompts=prompts, ground_truths=ground_truths, eval_sampling_params=sampling_params)



