import re
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


def _escape_regex_in_list(words):
    """Escape regex special characters in a list of words."""
    if words is None:
        return None
    if isinstance(words, list):
        return [re.escape(str(word)) for word in words]
    return words


def get_reward_score_batch(
    prompts: List[str],
    responses: list[str],
    instruction_id_list: list[str],
    instruct_kwargs: list = None,
    **kwargs  # Accept additional keyword arguments passed by reward manager
) -> list[float]:
    # Uncomment below for debugging
    # import sys
    # print(f"=== get_reward_score_batch CALLED ===", flush=True)
    # print(f"NUMBER OF PROMPTS: {len(prompts)}", flush=True)
    # print(f"NUMBER OF RESPONSES: {len(responses)}", flush=True)
    # print(f"NUMBER OF INSTRUCTION ID LIST: {len(instruction_id_list)}", flush=True)
    # print(f"INSTRUCT_KWARGS TYPE: {type(instruct_kwargs)}", flush=True)
    # if instruct_kwargs:
    #     print(f"NUMBER OF INSTRUCT KWARGS: {len(instruct_kwargs)}", flush=True)
    #     print(f"FIRST INSTRUCT_KWARGS ITEM TYPE: {type(instruct_kwargs[0])}", flush=True)
    #     print(f"FIRST INSTRUCT_KWARGS ITEM REPR: {repr(instruct_kwargs[0])[:200]}", flush=True)
    # sys.stdout.flush()
    try:
        response_dict = {}
        for prompt, response in zip(prompts, responses):
            response_dict[prompt] = response

        # Fields that should be integers (not floats) for instruction following eval
        INTEGER_FIELDS = {
            'nth_paragraph', 'num_paragraphs', 'num_bullets', 'num_highlights',
            'num_sections', 'num_sentences', 'num_words', 'num_placeholders',
            'capital_frequency', 'let_frequency', 'frequency'
        }

        # Fields that contain words/phrases used in regex patterns and need escaping
        REGEX_ESCAPE_FIELDS = {
            'forbidden_words', 'keywords', 'keyword', 'first_word', 'end_phrase'
        }

        out_kw = []
        for i, out_val in enumerate(instruct_kwargs):
            try:
                in_kw = []
                # Check if out_val is iterable and not a string/dict
                if isinstance(out_val, (list, tuple)):
                    for inval in out_val:
                        if isinstance(inval, dict):
                            # Convert numeric fields to integers and escape regex special chars
                            cleaned_dict = {}
                            for k, v in inval.items():
                                if v is not None:
                                    # Convert floats to ints for integer fields
                                    if k in INTEGER_FIELDS and isinstance(v, float):
                                        cleaned_dict[k] = int(v)
                                    # Escape regex special characters in string/list fields
                                    elif k in REGEX_ESCAPE_FIELDS:
                                        if isinstance(v, list):
                                            cleaned_dict[k] = _escape_regex_in_list(v)
                                        elif isinstance(v, str):
                                            cleaned_dict[k] = re.escape(v)
                                        else:
                                            cleaned_dict[k] = v
                                    else:
                                        cleaned_dict[k] = v
                            in_kw.append(cleaned_dict)
                        else:
                            print(f"Warning: instruct_kwargs[{i}] item is not a dict: {type(inval)} = {inval}")
                            in_kw.append(inval)
                else:
                    print(f"Warning: instruct_kwargs[{i}] is not a list/tuple: {type(out_val)} = {out_val}")
                    # If it's already a dict, use it directly
                    if isinstance(out_val, dict):
                        cleaned_dict = {}
                        for k, v in out_val.items():
                            if v is not None:
                                if k in INTEGER_FIELDS and isinstance(v, float):
                                    cleaned_dict[k] = int(v)
                                elif k in REGEX_ESCAPE_FIELDS:
                                    if isinstance(v, list):
                                        cleaned_dict[k] = _escape_regex_in_list(v)
                                    elif isinstance(v, str):
                                        cleaned_dict[k] = re.escape(v)
                                    else:
                                        cleaned_dict[k] = v
                                else:
                                    cleaned_dict[k] = v
                        in_kw = [cleaned_dict]
                    else:
                        in_kw = [out_val]
                out_kw.append(in_kw)
            except Exception as e:
                print(f"Error processing instruct_kwargs[{i}]: {e}, value: {out_val}, type: {type(out_val)}")
                raise

        # Uncomment below for detailed debugging
        # print(f"Sample converted kwargs (first item): {out_kw[0] if out_kw else 'empty'}", flush=True)
        # print(f"Creating {len(prompts)} InputExample objects...", flush=True)

        inputs = []
        for i, (prompt, inst_id, kws) in enumerate(zip(prompts, instruction_id_list, out_kw)):
            try:
                inp = InputExample(key=i, instruction_id_list=inst_id, prompt=prompt, kwargs=kws)
                inputs.append(inp)
            except Exception as e:
                print(f"Error creating InputExample {i}: {e}", flush=True)
                print(f"  prompt: {prompt[:100]}", flush=True)
                print(f"  inst_id: {inst_id}", flush=True)
                print(f"  kws: {kws}", flush=True)
                raise

        # print(f"Evaluating {len(inputs)} inputs...", flush=True)
        all_outputs = []
        for idx, inp in enumerate(inputs):
            try:
                output = test_instruction_following_strict(inp, response_dict)
                all_outputs.append(output.follow_all_instructions)
            except Exception as e:
                print(f"Error evaluating input {idx}: {e}", flush=True)
                print(f"  instruction_id_list: {inp.instruction_id_list}", flush=True)
                print(f"  kwargs: {inp.kwargs}", flush=True)
                raise
    except Exception as e:
        import traceback
        print(f"Error in get_reward_score_batch: {e}", flush=True)
        traceback.print_exc()
        all_outputs = [0.0] * len(prompts)

    # Uncomment below to see reward distribution
    # print(f"Reward summary: {sum(all_outputs)}/{len(all_outputs)} passed ({100*sum(all_outputs)/len(all_outputs):.1f}%)", flush=True)
    return all_outputs