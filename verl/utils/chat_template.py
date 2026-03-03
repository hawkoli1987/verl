# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    try:
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
        # get system prompt tokens
        system_prompt = token1[: -(len(token2) - len(token1))]
        return system_prompt
    except Exception as e:
        # Some models (e.g., Gemma2, Gemma3) require strict role alternation and will fail with consecutive user messages
        # For these models, the prefix is just the BOS token (no additional system prompt)
        logger.warning(
            f"Failed to extract system prompt using double user message method: {e}. "
            "Returning BOS token as system prompt (model likely enforces role alternation)."
        )
        # Return just the BOS token as the prefix
        return [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []


def extract_system_prompt_and_generation(tokenizer):
    try:
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
        # get system prompt tokens
        system_prompt = token1[: -(len(token2) - len(token1))]
    except Exception as e:
        # Some models (e.g., Gemma2, Gemma3) require strict role alternation and will fail with consecutive user messages
        # For these models, the prefix is just the BOS token (no additional system prompt)
        logger.warning(
            f"Failed to extract system prompt using double user message method: {e}. "
            "Returning BOS token as system prompt (model likely enforces role alternation)."
        )
        token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
        )
        # Return just the BOS token as the prefix
        system_prompt = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


# # Copyright 2025 Bytedance Ltd. and/or its affiliates
# import logging
# import os

# logger = logging.getLogger(__name__)
# logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
#     """
#     Initialize system prompt tokens for chat templates that support them.

#     Args:
#         tokenizer: The tokenizer with a chat template
#         **apply_chat_template_kwargs: Additional arguments for apply_chat_template

#     Returns:
#         List of token IDs for the system prompt, or empty list if not supported
#     """
#     token1 = tokenizer.apply_chat_template(
#         [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
#     )
#     token2 = tokenizer.apply_chat_template(
#         [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
#     )
#     # get system prompt tokens
#     system_prompt = token1[: -(len(token2) - len(token1))]
#     return system_prompt


# def extract_system_prompt_and_generation(tokenizer):
#     token1 = tokenizer.apply_chat_template(
#         [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
#     )
#     token2 = tokenizer.apply_chat_template(
#         [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
#     )
#     # get system prompt tokens
#     system_prompt = token1[: -(len(token2) - len(token1))]
#     # get generate prompt tokens
#     token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
#     generate_prompt = token3[len(token1) :]

#     return system_prompt, generate_prompt
