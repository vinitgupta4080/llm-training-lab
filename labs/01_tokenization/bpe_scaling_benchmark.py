"""Measure naive BPE training and encoding cost as merge count grows."""

from statistics import median
from time import perf_counter_ns

from llm_lab import ByteBPETokenizer


TRAINING_TEXT = ("banana bandana banana cabana " * 400).strip()
EVALUATION_TEXT = ("banana bandana " * 100).strip()
MERGE_BUDGETS = [0, 5, 10, 20, 40]
TRAIN_REPEATS = 5
ENCODE_REPEATS = 100


def elapsed_ms(function) -> float:
    start = perf_counter_ns()
    function()
    return (perf_counter_ns() - start) / 1_000_000


print("merges vocab train_ms encode_ms byte_tokens bpe_tokens")
for num_merges in MERGE_BUDGETS:
    # Warm up interpreter paths before collecting medians.
    tokenizer = ByteBPETokenizer.train(TRAINING_TEXT, num_merges=num_merges)
    tokenizer.encode(EVALUATION_TEXT)

    train_times = [
        elapsed_ms(lambda: ByteBPETokenizer.train(TRAINING_TEXT, num_merges=num_merges))
        for _ in range(TRAIN_REPEATS)
    ]
    tokenizer = ByteBPETokenizer.train(TRAINING_TEXT, num_merges=num_merges)
    encode_times = [
        elapsed_ms(lambda: tokenizer.encode(EVALUATION_TEXT))
        for _ in range(ENCODE_REPEATS)
    ]
    encoded = tokenizer.encode(EVALUATION_TEXT)
    assert tokenizer.decode(encoded) == EVALUATION_TEXT

    print(
        f"{num_merges:6} {259 + len(tokenizer.merges):5} "
        f"{median(train_times):8.3f} {median(encode_times):9.4f} "
        f"{len(EVALUATION_TEXT.encode('utf-8')):11} {len(encoded):10}"
    )
