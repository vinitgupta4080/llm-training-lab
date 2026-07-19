import pytest

from llm_lab import ByteTokenizer


@pytest.mark.parametrize("text", ["hello", "café", "नमस्ते", "🤖\n"])
def test_round_trip(text):
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text


def test_unknown_special_fails_when_not_skipped():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([ByteTokenizer.bos_id], skip_special=False)

