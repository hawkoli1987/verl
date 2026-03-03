import uuid
from typing import Dict, Any, List

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from instruction_following_eval.evaluation_lib import test_instruction_following_strict, InputExample

MAX_RETRIES = 3
BASE_DELAY = 10
MAX_WORKERS = 16
TIMEOUT = 600

load_dotenv()

run_id = str(uuid.uuid4())


def get_reward_score_batch(
    prompts: List[str],
    responses: list[str],
    instruction_id_list: list[str],
    instruct_kwargs: list = None,
) -> list[float]:

    # print(f"NUMBER OF PROMPTS: {len(prompts)}")
    # print(f"NUMBER OF RESPONSES: {len(responses)}")
    # print(f"NUMBER OF INSTRUCTION ID LIST: {len(instruction_id_list)}")
    # print(f"NUMBER OF INSTRUCT KWARGS: {len(instruct_kwargs)}")
    try:
        response_dict = {}
        for prompt, response in zip(prompts, responses):
            response_dict[prompt] = response

        out_kw = []
        for out_val in instruct_kwargs:
            in_kw = []
            for inval in out_val:
                in_kw.append({k: v for k, v in inval.items() if v is not None})
            out_kw.append(in_kw)

        inputs = [
            InputExample(key=i, instruction_id_list=inst_id, prompt=prompt, kwargs=kws)
            for i, (prompt, inst_id, kws) in enumerate(zip(prompts, instruction_id_list, out_kw))
        ]

        all_outputs = []
        for inp in inputs:
            output = test_instruction_following_strict(inp, response_dict)
            all_outputs.append(output.follow_all_instructions)
    except Exception as e:
        print(f"Error in get_reward_score_batch: {e}")
        all_outputs = [0.0] * len(prompts)
    print(f"All outputs: {all_outputs}")
    return all_outputs

        # kw = kwargs.get("instruct_kwargs", [{}])
        # inst_id_list = kwargs.get("ground_truth", [])
        # out_kw = []
        # for out_val in kw:
        #     in_kw = []
        #     for inval in out_val:
        #         in_kw.append({k: v for k, v in inval.items() if v is not None})
        #     out_kw.append(in_kw)
        # # Create InputExample instances
        # inputs = [
        #     InputExample(key=i, instruction_id_list=inst_id, prompt=prompt, kwargs=kws)
        #     for i, (prompt, inst_id, kws) in enumerate(zip(prompts, inst_id_list, out_kw))
        # ]

        # all_outputs = []
        # for inp in inputs:
        #     output = test_instruction_following_strict(inp, response_dict)
        #     all_outputs.append(output.follow_all_instructions)
        # return len(responses) * [1.0]  # Placeholder for actual scores