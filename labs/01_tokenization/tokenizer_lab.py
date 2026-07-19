from llm_lab import ByteTokenizer

tokenizer = ByteTokenizer()
samples = ["hello", "naïve café", "नमस्ते", "🤖"]
print("text             codepoints  bytes  with_BOS_EOS  byte_ids")
for text in samples:
    byte_ids = tokenizer.encode(text)
    framed_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    print(
        f"{text!r:16} {len(text):10}  {len(byte_ids):5}  {len(framed_ids):12}  {byte_ids}"
    )
    assert tokenizer.decode(framed_ids) == text

print("\nExercise: explain why character count and byte-token count differ.")
print("Stretch: implement pair merges and compare compression on this corpus.")
