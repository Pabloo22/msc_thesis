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

from method._hidden_worker import plan_batches, response_avg_hidden


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
        #: (rows, padded width) of every forward, for the memory-bound tests.
        self.shapes: list[tuple[int, int]] = []

    def __call__(
        self, *, input_ids, attention_mask, output_hidden_states, logits_to_keep=0
    ):
        assert output_hidden_states
        assert input_ids.shape == attention_mask.shape
        self.shapes.append(tuple(input_ids.shape))
        # Accepted and ignored, as the real forward effectively does for this
        # caller: it bounds the lm_head projection, which produces ``logits``,
        # and this stand-in has no head and returns none. Hidden states are
        # returned for every position either way -- slicing them here would
        # make the fake disagree with the model about what is being averaged.
        # Declared explicitly rather than swallowed by ``**kwargs`` so that the
        # next unexpected argument still fails loudly, the way this one did.
        assert logits_to_keep >= 0
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

    @pytest.mark.parametrize("max_batch_tokens", [2, 5, 9, 1000])
    def test_result_is_invariant_to_the_token_budget(self, max_batch_tokens):
        """The budget reorders samples into like-length batches; rows must
        still come back in the input's order, with the input's values."""
        reference = response_avg_hidden(
            LinearModel(), CharTokenizer(), PROMPTS, ANSWERS, layer=1, batch_size=1
        )
        budgeted = response_avg_hidden(
            LinearModel(),
            CharTokenizer(),
            PROMPTS,
            ANSWERS,
            layer=1,
            batch_size=len(PROMPTS),
            max_batch_tokens=max_batch_tokens,
        )
        torch.testing.assert_close(budgeted[0], reference[0])
        torch.testing.assert_close(budgeted[1], reference[1])

    def test_no_forward_exceeds_the_token_budget(self):
        """The bound is on padded tokens, which is what the OOM was in."""
        model = LinearModel()
        response_avg_hidden(
            model,
            CharTokenizer(),
            PROMPTS,
            ANSWERS,
            layer=1,
            batch_size=len(PROMPTS),
            max_batch_tokens=8,
        )

        assert model.shapes  # guard against vacuously passing on no forwards
        assert all(rows * width <= 8 for rows, width in model.shapes)

    def test_the_heaviest_forward_comes_first(self):
        """A batch size that cannot fit should OOM in the first seconds, not
        hours in -- which is how the 2h54m failure was paid for."""
        model = LinearModel()
        response_avg_hidden(
            model, CharTokenizer(), PROMPTS, ANSWERS, layer=1, batch_size=2
        )

        widths = [width for _, width in model.shapes]
        assert widths == sorted(widths, reverse=True)


class TestPlanBatches:
    def test_rows_are_capped_by_batch_size(self):
        assert plan_batches([10] * 7, batch_size=3, max_batch_tokens=10_000) == [
            [0, 1, 2],
            [3, 4, 5],
            [6],
        ]

    def test_long_samples_get_fewer_rows_than_short_ones(self):
        # The failure being fixed: at a fixed 8 rows the 100-token samples cost
        # 8x the 12-token ones, and only the former batch runs out of memory.
        lengths = [100] * 3 + [12] * 8
        batches = plan_batches(lengths, batch_size=8, max_batch_tokens=200)

        assert len(batches[0]) == 2  # the long samples
        assert len(batches[-1]) > 2  # the short ones, in the same budget
        assert all(len(b) * max(lengths[i] for i in b) <= 200 for b in batches)

    def test_every_sample_appears_exactly_once(self):
        lengths = [7, 3, 9, 1, 4, 4, 12, 2]
        batches = plan_batches(lengths, batch_size=3, max_batch_tokens=15)

        assert sorted(i for b in batches for i in b) == list(range(len(lengths)))

    def test_an_oversized_sample_runs_alone_rather_than_being_dropped(self):
        """Dropping it would misalign every later DeltaP row."""
        batches = plan_batches([5, 99, 5], batch_size=4, max_batch_tokens=10)

        assert [1] in batches
        assert sorted(i for b in batches for i in b) == [0, 1, 2]

    def test_empty_input_plans_no_batches(self):
        assert plan_batches([], batch_size=8, max_batch_tokens=100) == []
