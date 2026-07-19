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

