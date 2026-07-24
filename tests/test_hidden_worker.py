"""Unit tests for the batched activation extraction in ``_hidden_worker``.

The worker's contract is numeric, not behavioural: batching exists purely for
throughput, so its output must be identical (up to dtype rounding) to the
one-sample-at-a-time computation, in the input's row order. These tests pin
that down with a stub tokenizer and model, so they run on CPU with nothing
downloaded.

The stub tokenizer is character-level, which sidesteps real BPE's boundary
merging -- fine here because the *slicing* logic under test only ever sees
token counts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from method._hidden_worker import response_avg_hidden


class CharTokenizer:
    """Character-level stand-in: one token per character, id = codepoint."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(c) for c in text]


class LinearModel:
    """Deterministic stand-in for a causal LM with hidden-state output.

    ``hidden_states[l][b, t, :] = input_ids[b, t] * (l + 1)`` in every hidden
    dimension, so the expected response average is computable by hand and any
    off-by-one in the slice boundaries shows up as a wrong mean.
    """

    def __init__(self, n_layers: int = 2, hidden: int = 4):
        self.config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=hidden)
        self.device = torch.device("cpu")

    def __call__(self, *, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states
        assert input_ids.shape == attention_mask.shape
        base = input_ids.unsqueeze(-1).expand(
            *input_ids.shape, self.config.hidden_size
        ).float()
        return SimpleNamespace(
            hidden_states=tuple(
                base * (layer + 1)
                for layer in range(self.config.num_hidden_layers + 1)
            )
        )


PROMPTS = ["ab", "c", "defg", "hi", "j"]
ANSWERS = ["xyz", "uv", "w", "stuv", "q"]


def expected_response_mean(prompt: str, answer: str, layer: int) -> float:
    return sum(ord(c) for c in answer) / len(answer) * (layer + 1)


class TestResponseAvgHidden:
    def test_per_sample_rows_match_hand_computed_means_in_order(self):
        means, per_sample = response_avg_hidden(
            LinearModel(), CharTokenizer(), PROMPTS, ANSWERS, layer=1, batch_size=2
        )

        assert per_sample.shape == (len(PROMPTS), 4)
        for i, (prompt, answer) in enumerate(zip(PROMPTS, ANSWERS)):
            expected = expected_response_mean(prompt, answer, layer=1)
            torch.testing.assert_close(
                per_sample[i], torch.full((4,), expected), atol=1e-5, rtol=1e-5
            )

    def test_mean_by_layer_averages_the_per_sample_means(self):
        means, _ = response_avg_hidden(
            LinearModel(), CharTokenizer(), PROMPTS, ANSWERS, layer=1, batch_size=2
        )

        assert means.shape == (3, 4)  # embeddings + 2 layers
        for layer in range(3):
            expected = sum(
                expected_response_mean(p, a, layer) for p, a in zip(PROMPTS, ANSWERS)
            ) / len(PROMPTS)
            torch.testing.assert_close(
                means[layer], torch.full((4,), expected), atol=1e-4, rtol=1e-5
            )

    @pytest.mark.parametrize("batch_size", [1, 2, 3, len(PROMPTS), 100])
    def test_result_is_invariant_to_batch_size(self, batch_size):
        """Batching is a throughput knob; ragged padding must not leak into
        any mean, wherever the batch boundaries fall."""
        reference = response_avg_hidden(
            LinearModel(), CharTokenizer(), PROMPTS, ANSWERS, layer=1, batch_size=1
        )
        batched = response_avg_hidden(
            LinearModel(),
            CharTokenizer(),
            PROMPTS,
            ANSWERS,
            layer=1,
            batch_size=batch_size,
        )
        torch.testing.assert_close(batched[0], reference[0])
        torch.testing.assert_close(batched[1], reference[1])

    def test_empty_response_raises_instead_of_skipping(self):
        """DeltaP subtracts these rows from another file's row-by-row, so a
        silently dropped sample would misalign every later row."""
        with pytest.raises(ValueError, match="row-aligned"):
            response_avg_hidden(
                LinearModel(),
                CharTokenizer(),
                ["ab", "cd"],
                ["xy", ""],
                layer=1,
                batch_size=2,
            )
