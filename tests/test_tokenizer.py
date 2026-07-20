import pytest

from llm_lab import ByteBPETokenizer, ByteTokenizer


@pytest.mark.parametrize("text", ["hello", "café", "नमस्ते", "🤖\n"])
def test_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text


def test_unknown_special_fails_when_not_skipped():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([ByteTokenizer.bos_id], skip_special=False)


def test_byte_bpe_multilingual_round_trip_and_reserved_ids():
    tokenizer = ByteBPETokenizer.train("banana bandana " * 10, num_merges=8)
    assert tokenizer.merges[0][1] == 259

    for text in ["banana", "naïve café", "नमस्ते", "🤖🤖"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_bpe_rejects_unknown_id():
    tokenizer = ByteBPETokenizer.train("banana", num_merges=2)
    with pytest.raises(ValueError, match="unknown token ID: 999"):
        tokenizer.decode([999])
