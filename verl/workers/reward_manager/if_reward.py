from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("if_reward")
class IFRewardManager:
    """The reward manager for instruction following tasks."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the IFRewardManager instance.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        print(f"NUM_EXAMINE: {self.num_examine}")
        print(f"COMPUTE_SCORE: {self.compute_score}")
        print(f"REWARD_FN_KEY: {self.reward_fn_key}")

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        instructions = []
        responses = []
        ground_truths = []
        extra_infos = []
        data_sources = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # Collect inputs for batch scoring
            instructions.append(prompt_str)
            responses.append(response_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info["response"])
            data_sources.append(data_source)

        # print(f"Data: {data}")
        scores = self.compute_score(
            prompts=instructions,
            responses=responses,
            instruction_id_list=ground_truths,
            instruct_kwargs=extra_infos,
        )

        for i, score in enumerate(scores):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            reward_tensor[i, valid_response_length - 1] = score

            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", instructions[i])
                print("[response]", responses[i])
                print("[ground_truth]", ground_truths[i])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

        # return scores  # Placeholder for actual scores