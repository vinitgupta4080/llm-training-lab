"""Module 1 learner exercise: implement the core of byte pair encoding."""

from collections import Counter


def count_pairs(tokens: list[int]) -> Counter[tuple[int, int]]:
    """Count adjacent ordered pairs, including overlapping occurrences."""
    counts = Counter()
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        counts[pair] += 1
    return counts


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of pair from left to right."""
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def choose_pair(counts: Counter[tuple[int, int]]) -> tuple[int, int]:
    """Choose highest frequency, breaking ties by lexicographically smallest pair."""
    if not counts:
        raise ValueError("cannot choose from empty pair counts")
    return min(counts, key=lambda pair: (-counts[pair], pair))


def train_one_merge(
    tokens: list[int], next_id: int
) -> tuple[list[int], tuple[int, int], int]:
    """Run the learner's complete one-pass count/select/assign/merge design."""
    counts = count_pairs(tokens)
    pair = choose_pair(counts)
    merged = merge_pair(tokens, pair, next_id)
    return merged, pair, next_id


def train_bpe(
    tokens: list[int], num_merges: int, start_id: int = 256
) -> tuple[list[int], list[tuple[tuple[int, int], int]]]:
    """Repeat BPE training and return final tokens plus ordered learned rules."""
    working_tokens = list(tokens)
    learned_merges = []

    for offset in range(num_merges):
        if len(working_tokens) < 2:
            break

        counts = count_pairs(working_tokens)
        pair = choose_pair(counts)
        new_id = start_id + offset

        learned_merges.append((pair, new_id))
        working_tokens = merge_pair(working_tokens, pair, new_id)

    return working_tokens, learned_merges


def encode_with_merges(
    tokens: list[int], learned_merges: list[tuple[tuple[int, int], int]]
) -> list[int]:
    """Encode tokens by replaying learned BPE merge rules in order."""
    encoded = list(tokens)

    for pair, new_id in learned_merges:
        encoded = merge_pair(encoded, pair, new_id)

    return encoded


def deconstruct_token(
    encoded: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """Replace every occurrence of one learned ID with its original pair."""
    result = []
    for token_id in encoded:
        if token_id == new_id:
            result.extend(pair)
        else:
            result.append(token_id)
    return result


def decode_with_merges(
    encoded_tokens: list[int],
    learned_merges: list[tuple[tuple[int, int], int]],
) -> list[int]:
    """Decode learned IDs by reversing the merge rules in reverse order."""
    decoded = list(encoded_tokens)
    learned_ids = {new_id for pair, new_id in learned_merges}

    for token_id in encoded_tokens:
        if not (0 <= token_id <= 255 or token_id in learned_ids):
            raise ValueError(f"unknown token ID: {token_id}")

    for pair, new_id in reversed(learned_merges):
        decoded = deconstruct_token(decoded, pair, new_id)

    return decoded


# We use character-number labels only to make the first exercise readable.
b, a, n = 98, 97, 110
banana = [b, a, n, a, n, a]

counts = count_pairs(banana)
assert counts[(a, n)] == 2
assert counts[(n, a)] == 2
assert counts[(b, a)] == 1

an_id = 256
merged = merge_pair(banana, (a, n), an_id)
assert merged == [b, an_id, an_id, a]

assert choose_pair(counts) == (a, n)  # tie resolved lexicographically

one_pass_tokens, one_pass_pair, one_pass_id = train_one_merge(banana, an_id)
assert one_pass_pair == (a, n)
assert one_pass_id == an_id
assert one_pass_tokens == [b, an_id, an_id, a]

trained_tokens, learned_merges = train_bpe(banana, num_merges=2, start_id=256)
assert len(learned_merges) == 2
assert learned_merges[0] == ((a, n), 256)
assert len(trained_tokens) < len(banana)

encoded_banana = encode_with_merges(banana, learned_merges)
assert encoded_banana == trained_tokens
assert encoded_banana == [257, 256, a]

unmatched = [120, 121, 122]
assert encode_with_merges(unmatched, learned_merges) == unmatched

decoded_banana = decode_with_merges(encoded_banana, learned_merges)
assert decoded_banana == banana
assert bytes(decoded_banana).decode("utf-8") == "banana"
assert encoded_banana == [257, 256, a]  # decoding did not mutate its input

try:
    decode_with_merges([999], learned_merges)
    raise AssertionError("expected unknown token ID to fail")
except ValueError as error:
    assert str(error) == "unknown token ID: 999"

print("Module 1 BPE core exercises passed.")
