# Module 1 — Text, Unicode, bytes, and tokens

For the full algorithm, implementation details, edge cases, cost model, and required tests, see
[`docs/BPE_DEEP_DIVE.md`](../../docs/BPE_DEEP_DIVE.md).

## Why tokenization exists

Neural networks operate on tensors of numbers. They cannot multiply the string `"cat"` by a
weight matrix. Tokenization creates a reversible interface between human text and integer IDs:

```text
text → token pieces → integer IDs → embedding vectors → Transformer
```

An integer token ID is an index, not a measurement. Token 200 is not twice token 100. The
embedding table maps each ID to a learned vector of `C` channels.

## How it is used

```python
tokenizer = ByteTokenizer()
ids = tokenizer.encode("cat", add_bos=True, add_eos=True)
text = tokenizer.decode(ids)
```

Special tokens communicate structure rather than ordinary text:

- `PAD`: fills shorter sequences in a batch and should normally be masked
- `BOS`: marks the beginning of a sequence
- `EOS`: marks the end of a sequence

The tokenizer and model are coupled: the embedding table must have one row for every possible
token ID, and changing the tokenizer changes the meaning of those rows.

## Technical layer 1 — Unicode and UTF-8

Unicode assigns abstract code points to text characters. UTF-8 encodes those code points as
one to four bytes. ASCII characters use one byte, while many accented scripts and emoji use
multiple bytes.

```text
"A" → Unicode U+0041 → UTF-8 [65]
"é" → Unicode U+00E9 → UTF-8 [195,169]
"🤖" → Unicode U+1F916 → UTF-8 [240,159,164,150]
```

Python `len(text)` counts Unicode code points, not UTF-8 bytes. Human-perceived grapheme
clusters can also contain multiple code points, so “character count” needs a precise definition.

## Technical layer 2 — byte tokenization

The lab's first tokenizer treats every byte value `0..255` as a token. It is lossless for all
valid UTF-8 text, has no unknown text token, and needs only 256 ordinary vocabulary entries.
Its tradeoff is longer sequences for non-ASCII text and common words.

```text
vocabulary size = 256 bytes + PAD + BOS + EOS = 259
```

## Why token count affects cost

The context window is measured in tokens, not words or characters. More tokens mean:

- more embedding/activation memory
- more positions processed by every Transformer layer
- more next-token training targets
- quadratic standard-attention score size (`T × T`)

A tokenizer that compresses text into fewer useful tokens can increase effective context and
reduce compute, but a larger vocabulary increases embedding/output parameters and may represent
rare tokens poorly.

## First prediction

Before running `tokenizer_lab.py`, order these by expected byte-token count from fewest to most:

```text
"hello"   "naïve café"   "नमस्ते"   "🤖"
```

Do not count BOS/EOS for this prediction. Explain why Python character count may disagree with
byte-token count.

## Why BPE is used

Byte tokens are universal but inefficient: frequent words and multi-byte characters may occupy
many positions. Byte Pair Encoding learns frequent adjacent token pairs and adds each merged
pair as a new vocabulary entry. This trades a larger vocabulary for shorter sequences.

## How BPE training works

Start with byte tokens, then repeat:

1. count every adjacent token pair in the training corpus
2. select the most frequent pair using a deterministic tie-break rule
3. assign the merged pair a new token ID
4. replace all non-overlapping occurrences
5. save the ordered merge rule

For a simplified character display of `banana banana`:

```text
b a n a n a → b an an a → b anan a → banan a → banana
```

Real byte-level BPE begins with UTF-8 bytes, not Unicode characters. A merged token represents
a byte sequence and can therefore represent part of a character, one character, part of a
word, a whole word, whitespace plus a word, or punctuation.

## Technical tradeoffs

- Each merge increases vocabulary size by one.
- Embedding/output parameter count grows roughly with `vocabulary_size × model_width`.
- Sequence length often shrinks, reducing per-token layer work and attention score size.
- Merge choices depend on training-corpus frequency, so compression and fragmentation differ
  across languages and domains.
- Encoding must replay learned merges deterministically; it must not relearn frequencies from
  each input sentence.
- Byte fallback preserves the ability to represent unseen text.

## Complete BPE system

### Training

```python
tokens = list(training_text.encode("utf-8"))
merges = []

for new_id in range(256, target_vocab_size):
    counts = count_pairs(tokens)
    if not counts:
        break
    pair = choose_pair(counts)
    merges.append((pair, new_id))
    tokens = merge_pair(tokens, pair, new_id)
```

`choose_pair` must be deterministic. Our educational rule is:

1. prefer higher frequency
2. if frequencies tie, prefer the lexicographically smaller pair

For counts `{(97,110): 2, (110,97): 2, (98,97): 1}`, `(97,110)` wins because frequency 2
beats 1, and first token ID 97 is smaller than 110 in the tie.

One implementation expresses both priorities as a sortable key:

```python
min(counts, key=lambda pair: (-counts[pair], pair))
```

Negative frequency makes a larger count sort earlier; the pair tuple resolves ties.

### Vocabulary construction

Base IDs `0..255` map to their single bytes. If merge `(left,right) → new_id` is learned:

```text
vocab[new_id] = vocab[left] + vocab[right]
```

Thus `(97,110) → 256` stores byte sequence `b"an"` at ID 256.

### Encoding

Encoding new text begins with UTF-8 bytes and replays the saved merge rules in learned order.
It never calls `choose_pair` on the new sentence; corpus frequency was used during training only.

### Decoding

Replace every token ID with its stored byte sequence, concatenate the bytes, then decode UTF-8:

```text
token IDs → byte pieces → joined bytes → text
```

Lossless round-trip correctness requires `decode(encode(text)) == text`.

## BPE prediction

In the corpus `banana banana`, the pairs `a+n` and `n+a` are tied at four occurrences each.
Assume the deterministic rule chooses `a+n`. After merging every non-overlapping `a+n`, how
many tokens represent one `banana`, and what are those tokens?
