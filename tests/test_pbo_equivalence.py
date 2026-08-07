"""The fast CSCV must agree with the obvious slow one.

``probability_of_backtest_overfitting`` assembles each split's column Sharpes
from per-block sums instead of recomputing them over raw rows. That is an
optimization, not a change of definition, so it is pinned against a direct
transcription of the definition: rebuild the rows, call statistics.stdev.
"""

from __future__ import annotations

import random
import statistics
from itertools import combinations

import pytest

from trialgate import probability_of_backtest_overfitting


def _reference_pbo(performance_matrix: list[list[float]], *, block_count: int) -> float:
    """CSCV written the slow, literal way."""

    observations = len(performance_matrix)
    strategies = len(performance_matrix[0])
    usable_rows = observations - (observations % block_count)
    block_size = usable_rows // block_count
    block_slices = [
        performance_matrix[block * block_size : (block + 1) * block_size]
        for block in range(block_count)
    ]

    def column_sharpe(rows: list[list[float]], column: int) -> float:
        values = [row[column] for row in rows]
        if len(values) < 2:
            return 0.0
        stdev = statistics.stdev(values)
        if stdev == 0.0:
            return 0.0
        return statistics.fmean(values) / stdev

    overfit_count = 0
    combos = list(combinations(range(block_count), block_count // 2))
    for train_blocks in combos:
        train_set = frozenset(train_blocks)
        train_rows = [row for block in sorted(train_set) for row in block_slices[block]]
        test_rows = [
            row
            for block in range(block_count)
            if block not in train_set
            for row in block_slices[block]
        ]
        train_scores = [column_sharpe(train_rows, column) for column in range(strategies)]
        test_scores = [column_sharpe(test_rows, column) for column in range(strategies)]
        best = max(range(strategies), key=lambda column: train_scores[column])
        rank = sum(1 for score in test_scores if score <= test_scores[best])
        if rank / (strategies + 1) <= 0.5:
            overfit_count += 1
    return overfit_count / len(combos)


@pytest.mark.parametrize(
    ("seed", "rows", "strategies", "block_count"),
    [
        (1, 400, 12, 8),
        (2, 400, 12, 10),
        (3, 250, 5, 6),
        (4, 613, 30, 8),  # rows not divisible by block_count
        (5, 64, 2, 8),
    ],
)
def test_fast_cscv_matches_the_literal_definition(
    seed: int, rows: int, strategies: int, block_count: int
) -> None:
    rng = random.Random(seed)
    matrix = [[rng.gauss(0.0004, 0.011) for _ in range(strategies)] for _ in range(rows)]

    fast = probability_of_backtest_overfitting(matrix, block_count=block_count)

    assert fast.pbo == pytest.approx(_reference_pbo(matrix, block_count=block_count))


def test_constant_columns_score_zero_like_the_reference() -> None:
    rng = random.Random(9)
    matrix = [[0.0, 0.005, rng.gauss(0.0, 0.01)] for _ in range(200)]

    assert probability_of_backtest_overfitting(matrix, block_count=8).pbo == pytest.approx(
        _reference_pbo(matrix, block_count=8)
    )
