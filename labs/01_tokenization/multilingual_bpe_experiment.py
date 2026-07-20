"""Compare one fixed English-trained BPE tokenizer across language slices."""

from llm_lab import ByteBPETokenizer


training_text = ("banana bandana " * 20) + ("banana banana " * 20)
tokenizer = ByteBPETokenizer.train(training_text, num_merges=20)

samples = ["banana banana", "naïve café", "नमस्ते", "🤖🤖"]

print("text             bytes   bpe   reduction   round_trip")
for text in samples:
    byte_count = len(text.encode("utf-8"))
    bpe_ids = tokenizer.encode(text)
    reduction = 100 * (byte_count - len(bpe_ids)) / byte_count
    recovered = tokenizer.decode(bpe_ids)
    print(
        f"{text!r:16} {byte_count:5} {len(bpe_ids):5} "
        f"{reduction:9.1f}%   {recovered == text}"
    )
    assert recovered == text

assert tokenizer.merges[0][1] == 259
