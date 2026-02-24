"""Unit tests for the teacher client: payload, response parsing, token ID mapping, retry."""
import json
from unittest.mock import MagicMock, patch, call
import pytest
import torch

from verl.trainer.ppo.onpolicy_kd import (
    build_teacher_payload,
    extract_topk_tensors_from_openai_response,
    score_sequences,
    _token_str_to_id,
)


# ---------------------------------------------------------------------------
# Mock tokenizer fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_tokenizer():
    tok = MagicMock()
    tok.unk_token_id = 0
    # convert_tokens_to_ids maps " hello" → 100, " world" → 200, others → unk
    def _convert(tok_str):
        mapping = {" hello": 100, " world": 200, "▁the": 300}
        return mapping.get(tok_str, 0)  # 0 == unk_token_id → triggers fallback
    tok.convert_tokens_to_ids.side_effect = _convert
    # encode fallback: returns [999] for any string
    tok.encode.return_value = [999]
    return tok


# ---------------------------------------------------------------------------
# build_teacher_payload
# ---------------------------------------------------------------------------

def test_payload_has_required_keys():
    payload = build_teacher_payload("my-model", "Hello world", topk=32, temperature=0.8)
    assert payload["model"] == "my-model"
    assert payload["prompt"] == "Hello world"
    assert payload["logprobs"] == 32
    assert payload["echo"] is True
    assert payload["max_tokens"] == 0
    assert payload["temperature"] == pytest.approx(0.8)


def test_payload_default_temperature():
    payload = build_teacher_payload("m", "p", topk=4)
    assert payload["temperature"] == 1.0


# ---------------------------------------------------------------------------
# _token_str_to_id
# ---------------------------------------------------------------------------

def test_token_str_to_id_known_token(mock_tokenizer):
    assert _token_str_to_id(" hello", mock_tokenizer) == 100


def test_token_str_to_id_unknown_falls_back_to_encode(mock_tokenizer):
    # " xyz" → convert returns 0 (unk) → fallback encode returns [999]
    result = _token_str_to_id(" xyz", mock_tokenizer)
    assert result == 999
    mock_tokenizer.encode.assert_called_once_with(" xyz", add_special_tokens=False)


def test_token_str_to_id_encode_empty_returns_zero(mock_tokenizer):
    mock_tokenizer.encode.return_value = []
    result = _token_str_to_id(" xyz", mock_tokenizer)
    assert result == 0


# ---------------------------------------------------------------------------
# extract_topk_tensors_from_openai_response
# ---------------------------------------------------------------------------

def _make_response(top_logprobs_list):
    """Helper: wrap a list of {tok: lp} dicts in an OpenAI-style response."""
    return {
        "choices": [
            {
                "logprobs": {
                    "top_logprobs": top_logprobs_list,
                }
            }
        ]
    }


def test_extract_topk_shape(mock_tokenizer):
    """3 tokens, K=4 → tensor shape [3, 4]."""
    tok_map = {" hello": -0.5, " world": -1.2, "▁the": -2.0, " foo": -3.0}
    response = _make_response([tok_map, tok_map, tok_map])
    ids, lps = extract_topk_tensors_from_openai_response(
        response, mock_tokenizer, topk=4, response_length=3, device=torch.device("cpu")
    )
    assert ids.shape == (3, 4)
    assert lps.shape == (3, 4)


def test_extract_topk_takes_last_response_length_tokens(mock_tokenizer):
    """When echo=True returns more tokens than response_length, take the last ones."""
    tok_map_a = {" hello": -0.1}
    tok_map_b = {" world": -0.2}
    tok_map_c = {"▁the": -0.3}
    # 3 tokens total (e.g. prompt + response), response_length=2 → keep last 2
    response = _make_response([tok_map_a, tok_map_b, tok_map_c])
    ids, lps = extract_topk_tensors_from_openai_response(
        response, mock_tokenizer, topk=2, response_length=2, device=torch.device("cpu")
    )
    # Row 0 should correspond to tok_map_b (" world" → 200)
    assert ids[0, 0].item() == 200
    # Row 1 should correspond to tok_map_c ("▁the" → 300)
    assert ids[1, 0].item() == 300


def test_extract_topk_padding_with_inf(mock_tokenizer):
    """Fewer tokens than response_length → leading rows padded with -inf / 0."""
    tok_map = {" hello": -0.5}
    response = _make_response([tok_map])  # only 1 position
    ids, lps = extract_topk_tensors_from_openai_response(
        response, mock_tokenizer, topk=2, response_length=3, device=torch.device("cpu")
    )
    assert ids.shape == (3, 2)
    # First two rows (offset) should be all zeros / -inf
    assert (ids[0] == 0).all()
    assert (lps[0] == float("-inf")).all()
    assert (ids[1] == 0).all()
    assert (lps[1] == float("-inf")).all()
    # Last row has actual data
    assert ids[2, 0].item() == 100  # " hello" → 100


def test_extract_topk_empty_response(mock_tokenizer):
    """Empty choices → all-zero ids and -inf logprobs."""
    response = {"choices": []}
    ids, lps = extract_topk_tensors_from_openai_response(
        response, mock_tokenizer, topk=4, response_length=5, device=torch.device("cpu")
    )
    assert (ids == 0).all()
    assert (lps == float("-inf")).all()


# ---------------------------------------------------------------------------
# score_sequences — retry logic
# ---------------------------------------------------------------------------

def _make_openai_response(token_str: str, logprob: float, n_positions: int):
    """Build a minimal OpenAI response with n_positions identical top-logprob dicts."""
    top_lps = [{token_str: logprob}] * n_positions
    return {"choices": [{"logprobs": {"top_logprobs": top_lps}}]}


def test_retry_on_failure(mock_tokenizer):
    """score_sequences retries up to max_retries times then succeeds."""
    response = _make_openai_response(" hello", -0.5, n_positions=2)
    call_count = {"n": 0}

    def fake_urlopen(req, timeout):
        call_count["n"] += 1
        if call_count["n"] < 3:  # first 2 calls fail
            raise OSError("connection refused")
        # third call succeeds
        cm = MagicMock()
        cm.__enter__ = lambda s: s
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = json.dumps(response).encode()
        return cm

    with patch("verl.trainer.ppo.onpolicy_kd.urllib.request.urlopen", side_effect=fake_urlopen):
        with patch("verl.trainer.ppo.onpolicy_kd.time.sleep"):  # speed up
            ids, lps, valid = score_sequences(
                base_url="http://fake/v1",
                model="m",
                tokenizer=mock_tokenizer,
                sequences=["hello world"],
                response_length=2,
                topk=1,
                max_retries=2,
            )
    assert call_count["n"] == 3
    assert valid[0].item() is True


def test_fail_open_on_all_retries_exhausted(mock_tokenizer):
    """When all retries fail, valid_mask is False and tensors are zero/-inf."""
    with patch("verl.trainer.ppo.onpolicy_kd.urllib.request.urlopen", side_effect=OSError("timeout")):
        with patch("verl.trainer.ppo.onpolicy_kd.time.sleep"):
            ids, lps, valid = score_sequences(
                base_url="http://fake/v1",
                model="m",
                tokenizer=mock_tokenizer,
                sequences=["hello"],
                response_length=3,
                topk=2,
                max_retries=1,
            )
    assert valid[0].item() is False
    assert (ids[0] == 0).all()
    assert (lps[0] == float("-inf")).all()


def test_score_sequences_batch_shape(mock_tokenizer):
    """score_sequences returns correct [B,T,K] shapes."""
    B, T, K = 3, 2, 4
    response = _make_openai_response(" hello", -0.5, n_positions=T)

    def fake_urlopen(req, timeout):
        cm = MagicMock()
        cm.__enter__ = lambda s: s
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = json.dumps(response).encode()
        return cm

    with patch("verl.trainer.ppo.onpolicy_kd.urllib.request.urlopen", side_effect=fake_urlopen):
        ids, lps, valid = score_sequences(
            base_url="http://fake/v1",
            model="m",
            tokenizer=mock_tokenizer,
            sequences=["seq"] * B,
            response_length=T,
            topk=K,
        )
    assert ids.shape == (B, T, K)
    assert lps.shape == (B, T, K)
    assert valid.shape == (B,)
    assert valid.all()


def test_score_sequences_url_uses_completions_endpoint(mock_tokenizer):
    """Verifies the constructed URL ends with /completions, not /v1/completions."""
    captured_urls = []

    def fake_urlopen(req, timeout):
        captured_urls.append(req.full_url)
        cm = MagicMock()
        cm.__enter__ = lambda s: s
        cm.__exit__ = MagicMock(return_value=False)
        cm.read.return_value = json.dumps({"choices": []}).encode()
        return cm

    with patch("verl.trainer.ppo.onpolicy_kd.urllib.request.urlopen", side_effect=fake_urlopen):
        score_sequences(
            base_url="https://host/teacher/v1",
            model="m",
            tokenizer=mock_tokenizer,
            sequences=["test"],
            response_length=1,
            topk=1,
        )
    assert len(captured_urls) == 1
    assert captured_urls[0] == "https://host/teacher/v1/completions"
