import os
import re
import uuid
from typing import Dict, Any, List
from multiprocessing import get_context

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from instruction_following_eval.evaluation_lib import test_instruction_following_strict, InputExample

MAX_RETRIES = 3
BASE_DELAY = 10
MAX_WORKERS = 16
TIMEOUT = 600
REWARD_PARALLEL_WORKERS = int(os.environ.get("REWARD_PARALLEL_WORKERS", "0"))

load_dotenv()

run_id = str(uuid.uuid4())


def _evaluate_single(args):
    """Worker function for multiprocessing -- must be top-level and picklable."""
    inp, response_dict = args
    try:
        output = test_instruction_following_strict(inp, response_dict)
        return output.follow_all_instructions
    except Exception as e:
        print(f"Error evaluating input {inp.key}: {e}", flush=True)
        return 0.0


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

        # --- Evaluate: parallel or sequential ---
        if REWARD_PARALLEL_WORKERS > 0:
            import time as _time
            _t0 = _time.time()
            ctx = get_context("fork")  # fork inherits sys.modules so pickle can find dynamically-loaded modules
            with ctx.Pool(processes=REWARD_PARALLEL_WORKERS) as pool:
                all_outputs = pool.map(_evaluate_single, [(inp, response_dict) for inp in inputs])
            _elapsed = _time.time() - _t0
            print(f"[reward_if] Parallel eval ({REWARD_PARALLEL_WORKERS} workers): "
                  f"{len(inputs)} samples in {_elapsed:.2f}s "
                  f"({_elapsed/len(inputs)*1000:.1f} ms/sample)", flush=True)
        else:
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

    return all_outputs
