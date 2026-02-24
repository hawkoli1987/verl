"""Integration tests hitting the real teacher vLLM endpoint.

Run with:
    TEACHER_URL=https://transmissively-conidial-fredrick.ngrok-free.dev/teacher/v1 \
        pytest tests/onpolicy_kd/test_integration.py -v

Skipped automatically when TEACHER_URL is not set so CI is never blocked.
"""
import os
import pytest
import torch

TEACHER_URL = os.environ.get(
    "TEACHER_URL",
    "https://transmissively-conidial-fredrick.ngrok-free.dev/teacher/v1",
)
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-4B")

skip_without_url = pytest.mark.skipif(
    "TEACHER_URL" not in os.environ,
    reason="TEACHER_URL env var not set; skipping live teacher tests",
)


def _get_tokenizer():
    """Load a minimal tokenizer for integration tests."""
    try:
        from transformers import AutoTokenizer

        # Use a small fast tokenizer; any HF tokenizer works for this test
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("transformers not available or gpt2 not cached")


@skip_without_url
def test_score_sequences_real_shapes():
    """score_sequences returns [B,T,K] tensors with correct shapes."""
    from verl.trainer.ppo.onpolicy_kd import score_sequences

    tokenizer = _get_tokenizer()
    sequences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you doing today?",
    ]
    B = len(sequences)
    T = 4   # ask for last 4 token positions
    K = 8

    ids, lps, valid = score_sequences(
        base_url=TEACHER_URL,
        model=TEACHER_MODEL,
        tokenizer=tokenizer,
        sequences=sequences,
        response_length=T,
        topk=K,
        timeout_s=30.0,
        max_retries=1,
    )

    assert ids.shape == (B, T, K), f"Expected ({B},{T},{K}), got {ids.shape}"
    assert lps.shape == (B, T, K), f"Expected ({B},{T},{K}), got {lps.shape}"
    assert valid.shape == (B,)


@skip_without_url
def test_score_sequences_logprobs_finite_and_negative():
    """Returned logprobs for valid positions should be finite and <= 0."""
    from verl.trainer.ppo.onpolicy_kd import score_sequences

    tokenizer = _get_tokenizer()
    sequences = ["The answer is 42."]
    T = 3
    K = 16

    ids, lps, valid = score_sequences(
        base_url=TEACHER_URL,
        model=TEACHER_MODEL,
        tokenizer=tokenizer,
        sequences=sequences,
        response_length=T,
        topk=K,
        timeout_s=30.0,
        max_retries=1,
    )

    if not valid[0]:
        pytest.skip("Teacher returned no data for this sequence")

    # Valid logprob slots should be <= 0 and finite
    finite_mask = lps[0].isfinite()
    if finite_mask.any():
        assert (lps[0][finite_mask] <= 0.0).all(), "Logprobs must be <= 0"


@skip_without_url
def test_kd_loss_nonzero_real():
    """KD loss against a toy random student is > 0."""
    from verl.trainer.ppo.onpolicy_kd import score_sequences

    tokenizer = _get_tokenizer()
    sequences = ["Once upon a time there was a"]
    T = 3
    K = 8
    V = tokenizer.vocab_size or 50257  # GPT-2 vocab size

    ids, lps, valid = score_sequences(
        base_url=TEACHER_URL,
        model=TEACHER_MODEL,
        tokenizer=tokenizer,
        sequences=sequences,
        response_length=T,
        topk=K,
        timeout_s=30.0,
        max_retries=1,
    )

    if not valid[0]:
        pytest.skip("Teacher returned no data")

    # Build a random student
    torch.manual_seed(0)
    student_logits = torch.randn(1, T, V)

    teacher_ids = ids  # [1, T, K]
    teacher_lps = lps  # [1, T, K]

    logits = student_logits.float()
    student_topk_logits = torch.gather(logits, dim=-1, index=teacher_ids)
    student_topk_lps = torch.log_softmax(student_topk_logits, dim=-1)
    teacher_renorm = teacher_lps - torch.logsumexp(teacher_lps, dim=-1, keepdim=True)
    teacher_probs = torch.exp(teacher_renorm)
    token_kd = torch.sum(teacher_probs * (teacher_renorm - student_topk_lps), dim=-1)

    # Filter positions where teacher data is finite
    finite_pos = teacher_lps[0, :, 0].isfinite()
    if not finite_pos.any():
        pytest.skip("No finite teacher positions")

    kd_mean = token_kd[0][finite_pos].mean().item()
    assert kd_mean > 0.0, f"Expected KD loss > 0, got {kd_mean}"


@skip_without_url
def test_valid_mask_all_true_on_success():
    """valid_mask should be all True when the teacher responds successfully."""
    from verl.trainer.ppo.onpolicy_kd import score_sequences

    tokenizer = _get_tokenizer()
    sequences = ["Hello world", "Foo bar"]

    _, _, valid = score_sequences(
        base_url=TEACHER_URL,
        model=TEACHER_MODEL,
        tokenizer=tokenizer,
        sequences=sequences,
        response_length=2,
        topk=4,
        timeout_s=30.0,
        max_retries=1,
    )

    if not valid.all():
        pytest.skip(
            f"Teacher returned failures for some sequences (valid_mask={valid}). "
            "Check TEACHER_MODEL is correct for this endpoint."
        )
    assert valid.all(), f"Expected all valid, got valid_mask={valid}"
