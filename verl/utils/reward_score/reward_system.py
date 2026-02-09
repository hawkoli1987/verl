import os
from typing import Optional
import uuid

import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

MAX_RETRIES = 3
BASE_DELAY = 10
MAX_WORKERS = 16
TIMEOUT = 600

load_dotenv()

run_id = str(uuid.uuid4())

def get_reward_score(
    instruction: str,
    response: str,
    ground_truth: str,
    agents_to_force_enable: Optional[list[str]] = ['policy_critic', 'constraint_analyzer'],
    agents_to_force_disable: Optional[list[str]] = ["code_quality_critic", "math_critic", "factuality_checker"],
    extra_info=None,
) -> float:
    """
    Call external reward API to get reward score.

    Args:
        prompt (str): The input prompt.
        response (str): The generated response.
        api_url (str): The reward API endpoint.
        metadata (dict, optional): Additional metadata to send in the request.

    Returns:
        float: Reward score returned by
    """
    agents_to_force_enable = ['policy_critic', 'constraint_analyzer']
    agents_to_force_disable = ["code_quality_critic", "math_critic", "factuality_checker"]

    payload = {
        "instruction": instruction,
        "response": response,
        "ground_truth": ground_truth,
        "langsmith_tracing": {
            "run_name": "agentic-reward-system-run",
            "run_id": run_id,
            "tags": ["string"],
            "metadata": {"additionalProp1": {}},
        },
        "agents_to_force_enable": agents_to_force_enable,
        "agents_to_force_disable": agents_to_force_disable,
    }
    print(f"[RewardAPI] Payload: {payload}")

    API_URL = "http://smc-pod-32:8001/invoke_agentic_reward_system" #os.getenv("API_URL")

    try:
        res = requests.post(API_URL, json=payload, timeout=TIMEOUT)
        print(res)
        res.raise_for_status()
        data = res.json()
        reward = data.get("final_eval_score")
        print(f"[RewardAPI] Response: {data}")
        print(f"[RewardAPI] Reward: {reward}")
        if reward is None:
            raise ValueError("Missing final_eval_score in API response")
        return reward
    except Exception as e:
        print(f"[RewardAPI Error] {e}")
        return 5.0  # Fallback reward

def get_reward_score_batch(
    instructions: list[str],
    responses: list[str],
    ground_truths: list[str],
    agents_to_force_enable_list: list[list[str]] = None,
    agents_to_force_disable_list: list[list[str]] = None,
    extra_infos: list = None,
) -> list[float]:
    """
    Batch version with threading for parallel calls.
    """
    print("in get_reward_score_batch")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, instruction in enumerate(instructions):
            agents_enable = ['policy_critic','constraint_analyzer']
            # agents_enable = (
            #     agents_to_force_enable_list[i] if agents_to_force_enable_list else None
            # )
            agents_disable = ["code_quality_critic", "math_critic"]
            # agents_disable = (
            #     agents_to_force_disable_list[i] if agents_to_force_disable_list else None
            # )
            extra_info = extra_infos[i] if extra_infos else None

            futures.append(
                executor.submit(
                    get_reward_score,
                    instruction,
                    responses[i],
                    ground_truths[i],
                    agents_enable,
                    agents_disable,
                    extra_info,
                )
            )

        results = [future.result() for future in futures]
        print("RESULTS:", results)
    return results

if __name__ == "__main__":
    score = get_reward_score(
        instruction="இந்திய தெய்வத்தின் பெயர் என்ன, ஒரு பத்தியில் எனக்குக் கொடுங்கள்.",
        response="Ganesh, also known as Vinayaka, is a widely revered deity in Hinduism, recognized as the remover of obstacles and the god of beginnings.",
        ground_truth="விநாயகர் (Ganesh) — 장애ங்களை நீக்கும் தமிழ் மற்றும் இந்து மக்கள் வழிபடும் தெய்வம்.",
        agents_to_force_enable=None, #["policy_critic", "factuality_checker"],
        agents_to_force_disable=["code_quality_critic",]
    )
    print(f"Reward score: {score}")