# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Utilities for onpolicyKD_RL teacher scoring.

This module provides a small OpenAI-compatible client helper to query a teacher
vLLM endpoint for top-k logprobs, plus tensor helpers to construct batched
teacher top-k tensors for actor training.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import torch


def build_teacher_payload(
    model: str,
    prompt: str,
    topk: int,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Build an OpenAI-compatible completion payload for teacher scoring.

    Uses echo=True and max_tokens=0 so the endpoint returns logprobs over the
    input tokens without generating new ones.
    """
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,
        "temperature": temperature,
        "logprobs": topk,
        "echo": True,
    }


def _token_str_to_id(tok: str, tokenizer) -> int:
    """Map a decoded token string to its integer token ID.

    Falls back to encoding the string and taking the first token ID when
    ``convert_tokens_to_ids`` returns the unknown-token id or None.
    """
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tok_id is None or tok_id == unk_id:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if encoded:
            return encoded[0]
        return 0
    return int(tok_id)


def _do_request(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """POST *payload* to *url* with a read timeout of *timeout_s* seconds."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_topk_tensors_from_openai_response(
    response: dict[str, Any],
    tokenizer,
    topk: int,
    response_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-token top-k ids/logprobs from an OpenAI-style completions response.

    When the response was obtained with ``echo=True``, ``top_logprobs`` contains
    entries for every input token (prompt + response).  This helper takes the
    **last** ``response_length`` entries so the returned tensors correspond to
    the response region only.

    Returns:
        topk_token_ids : LongTensor  [T, K]   where T = response_length
        topk_logprobs  : FloatTensor [T, K]   (-inf for missing slots)
    """
    topk_token_ids = torch.zeros((response_length, topk), dtype=torch.long, device=device)
    topk_logprobs = torch.full(
        (response_length, topk), fill_value=float("-inf"), dtype=torch.float32, device=device
    )

    choices = response.get("choices", [])
    if not choices:
        return topk_token_ids, topk_logprobs

    all_top_logprobs = choices[0].get("logprobs", {}).get("top_logprobs", [])
    # Take only the last response_length positions (response region from echo)
    token_logprobs = all_top_logprobs[-response_length:] if len(all_top_logprobs) >= response_length else all_top_logprobs
    offset = response_length - len(token_logprobs)  # leading rows stay zero/-inf if too short

    for rel_t, token_map in enumerate(token_logprobs):
        t = offset + rel_t
        if not isinstance(token_map, dict):
            continue
        items = list(token_map.items())[:topk]
        for k, (tok_str, lp) in enumerate(items):
            topk_token_ids[t, k] = _token_str_to_id(tok_str, tokenizer)
            lp_val = float(lp)
            topk_logprobs[t, k] = lp_val

    return topk_token_ids, topk_logprobs


def _score_single(
    base_url: str,
    model: str,
    tokenizer,
    sequence: str,
    topk: int,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    seq_idx: int,
    response_length: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, bool]:
    """Score one sequence against the teacher endpoint.

    Returns ``(seq_idx, topk_ids [T,K], topk_lps [T,K], success)``.
    """
    url = f"{base_url.rstrip('/')}/completions"
    payload = build_teacher_payload(model, sequence, topk, temperature)

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = _do_request(url, payload, timeout_s)
            ids, lps = extract_topk_tensors_from_openai_response(
                response, tokenizer, topk, response_length, device
            )
            return seq_idx, ids, lps, True
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))

    # All attempts exhausted — return zero/−inf tensors and signal failure
    ids = torch.zeros((response_length, topk), dtype=torch.long, device=device)
    lps = torch.full((response_length, topk), float("-inf"), dtype=torch.float32, device=device)
    return seq_idx, ids, lps, False


def score_sequences(
    base_url: str,
    model: str,
    tokenizer,
    sequences: list[str],
    response_length: int,
    topk: int = 64,
    temperature: float = 1.0,
    timeout_s: float = 30.0,
    max_retries: int = 2,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a batch of sequences against the teacher vLLM endpoint.

    Issues one HTTP request per sequence in parallel via a thread pool, with
    exponential-back-off retries on failure.

    Args:
        base_url: Teacher endpoint base URL ending with ``/v1`` (e.g.
            ``https://host/teacher/v1``).  ``/completions`` is appended
            internally.
        model: Model name string forwarded in the ``model`` field of the
            completions payload.
        tokenizer: HuggingFace tokenizer used to map decoded token strings back
            to integer IDs.
        sequences: List of B full-context strings (prompt + response concatenated).
        response_length: Number of tokens in the response region.  Only the
            last *response_length* positions of each returned ``top_logprobs``
            list are kept.
        topk: Number of top tokens to request per position.
        temperature: Softmax temperature forwarded to the teacher.
        timeout_s: Per-request HTTP timeout in seconds.
        max_retries: Number of additional attempts after the first failure.
        device: Torch device for the output tensors (default: CPU).

    Returns:
        topk_token_ids : LongTensor  [B, T, K]
        topk_logprobs  : FloatTensor [B, T, K]  (−inf for missing slots)
        valid_mask     : BoolTensor  [B]         (True when request succeeded)
    """
    if device is None:
        device = torch.device("cpu")

    B = len(sequences)
    topk_token_ids = torch.zeros((B, response_length, topk), dtype=torch.long, device=device)
    topk_logprobs = torch.full(
        (B, response_length, topk), float("-inf"), dtype=torch.float32, device=device
    )
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)

    with ThreadPoolExecutor(max_workers=min(B, 32)) as pool:
        futures = {
            pool.submit(
                _score_single,
                base_url,
                model,
                tokenizer,
                seq,
                topk,
                temperature,
                timeout_s,
                max_retries,
                i,
                response_length,
                device,
            ): i
            for i, seq in enumerate(sequences)
        }
        for future in as_completed(futures):
            idx, ids, lps, success = future.result()
            topk_token_ids[idx] = ids
            topk_logprobs[idx] = lps
            valid_mask[idx] = success

    return topk_token_ids, topk_logprobs, valid_mask
