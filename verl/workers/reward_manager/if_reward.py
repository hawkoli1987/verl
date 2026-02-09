
import json
import os
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
        self.num_examine = num_examine  # the number of decoded responses to print per data_source
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self._eval_counter = 0
        self._log_dir = os.environ.get("VERL_LOG_DIR", "")

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

        scores = self.compute_score(
            prompts=instructions,
            responses=responses,
            instruction_id_list=ground_truths,
            instruct_kwargs=extra_infos,
        )

        # Collect examined samples for JSONL saving
        examined_samples = []

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
                # Save this examined sample for JSONL
                examined_samples.append({
                    "data_source": data_source,
                    "prompt": instructions[i],
                    "response": responses[i],
                    "ground_truth": ground_truths[i],
                    "score": float(score) if not isinstance(score, dict) else score,
                })

        # Save only the examined samples (1 per data_source) as JSONL
        if examined_samples and self._log_dir:
            self._save_eval_jsonl(examined_samples)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _save_eval_jsonl(self, samples):
        """Append examined eval samples to a JSONL file in the log directory."""
        self._eval_counter += 1
        eval_dir = os.path.join(self._log_dir, "eval_generations")
        os.makedirs(eval_dir, exist_ok=True)
        filepath = os.path.join(eval_dir, f"eval_{self._eval_counter:04d}.jsonl")
        with open(filepath, "w") as f:
            for record in samples:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[eval_jsonl] Saved {len(samples)} sample(s) to {filepath}")
