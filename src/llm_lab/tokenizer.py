from collections import Counter


def _count_pairs(tokens: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(tokens, tokens[1:]))


def _choose_pair(counts: Counter[tuple[int, int]]) -> tuple[int, int]:
    return min(counts, key=lambda pair: (-counts[pair], pair))


def _merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with explicit special-token IDs."""

    pad_id = 256
    bos_id = 257
    eos_id = 258
    vocab_size = 259

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id}
        values = [token for token in ids if not (skip_special and token in special)]
        if any(token < 0 or token > 255 for token in values):
            raise ValueError("Cannot decode unknown or unskipped special token as a byte")
        return bytes(values).decode("utf-8")


class ByteBPETokenizer:
    """Tiny byte-level BPE tokenizer with explicit, inspectable merge rules."""

    first_merge_id = 259  # 256–258 are reserved by ByteTokenizer.

    def __init__(self, merges: list[tuple[tuple[int, int], int]]) -> None:
        self.merges = list(merges)

    @classmethod
    def train(cls, text: str, *, num_merges: int) -> "ByteBPETokenizer":
        working = list(text.encode("utf-8"))
        merges = []

        for offset in range(num_merges):
            if len(working) < 2:
                break
            pair = _choose_pair(_count_pairs(working))
            new_id = cls.first_merge_id + offset
            merges.append((pair, new_id))
            working = _merge_pair(working, pair, new_id)

        return cls(merges)

    def encode(self, text: str) -> list[int]:
        encoded = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            encoded = _merge_pair(encoded, pair, new_id)
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        decoded = list(token_ids)
        learned_ids = {new_id for _, new_id in self.merges}
        for token_id in decoded:
            if not (0 <= token_id <= 255 or token_id in learned_ids):
                raise ValueError(f"unknown token ID: {token_id}")

        for pair, new_id in reversed(self.merges):
            expanded = []
            for token_id in decoded:
                if token_id == new_id:
                    expanded.extend(pair)
                else:
                    expanded.append(token_id)
            decoded = expanded

        return bytes(decoded).decode("utf-8")
