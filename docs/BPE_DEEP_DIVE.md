# Byte Pair Encoding: detailed notes

## 1. The problem BPE solves

The model cannot read text directly; it reads token IDs. A byte tokenizer can represent every
UTF-8 string, but common text may require many byte tokens. More tokens consume context, create
more training targets, require more per-position layer work, and enlarge standard attention's
`T × T` score matrices.

BPE learns a finite shortcut dictionary. If two adjacent token pieces appear together often,
it creates one new token representing their concatenation.

```text
b · a · n · a · n · a
pair a+n is frequent
b · an · an · a
```

Nothing is discarded. Token `an` stores the same bytes as `a` followed by `n`.

## 2. The central tradeoff: two bills

### What you need to understand in Module 1

Do not connect this to Transformer attention yet. At the tokenizer level, the tradeoff is simply:

```text
more learned shortcuts → larger tokenizer dictionary → fewer pieces for familiar text
fewer learned shortcuts → smaller tokenizer dictionary → more pieces for the same text
```

For `banana`, adding only the shortcut `an` changes:

```text
dictionary: +1 entry
encoded text: 6 pieces → 4 pieces
```

Neither extreme is automatically best. A tiny dictionary spells everything with many pieces;
a huge dictionary contains many shortcuts that may be rare or useful only on its training text.

The model-level details below are an **advanced preview**. They are not required until the
Transformer module, when embedding tables, attention, logits, and decoding cost are derived.

Tokenizer design moves cost between two places.

### Vocabulary bill

If vocabulary size is `V` and hidden width is `C`, the input embedding contains:

```text
V × C parameters
```

The output projection also produces `V` logits. Its weight may be tied to the input embedding
or stored separately. Larger `V` therefore increases parameter memory and output computation.

### Sequence bill

If encoded length is `T`, every Transformer layer processes `T` positions. Standard attention
creates scores shaped approximately `[heads,T,T]`, so its score count grows with:

```text
T²
```

Autoregressive generation also emits one token per decoding step. A tokenizer needing more
tokens for the same answer usually requires more sequential decode steps.

### Concrete example

Assume hidden width `C=768`:

```text
Vocabulary 256  → embedding parameters = 256 × 768  = 196,608
Vocabulary 1024 → embedding parameters = 1024 × 768 = 786,432
Extra parameters                              = 589,824
```

Suppose merges reduce a document from 100 to 70 tokens:

```text
attention score cells: 100² = 10,000
attention score cells:  70² =  4,900
reduction                        51%
```

This does not mean total model compute falls exactly 51%; MLPs, projections, kernels, padding,
and hardware behavior also matter. It shows why shorter token sequences can be valuable.

### Extremes are both poor

- Very small vocabulary: small embedding table and universal coverage, but long sequences,
  slower decoding, less effective context, and language-dependent fragmentation.
- Very large vocabulary: short frequent sequences, but large embeddings/logit layers, many
  rare poorly trained tokens, more training data required per token, and costly softmax/output.

The goal is not maximum vocabulary or minimum token count. It is best quality and coverage
within memory, latency, compute, data, and fairness constraints.

## 3. BPE training inputs

A complete tokenizer specification includes:

- text normalization policy (or explicit decision not to normalize)
- pre-tokenization/boundary policy, if any
- initial vocabulary (UTF-8 bytes in this course)
- special tokens and their reserved IDs
- deterministic tie-breaking
- number of merges or target vocabulary size
- exact training-corpus version and sampling mixture

These choices affect the learned vocabulary. Changing them produces a different tokenizer.

## 4. Initial byte vocabulary

IDs `0..255` represent individual byte values. Each vocabulary entry stores bytes:

```python
vocab = {byte: bytes([byte]) for byte in range(256)}
```

Special IDs must not collide with learned merge IDs. If `PAD=256`, `BOS=257`, and `EOS=258`,
the first learned merge must start at 259 or later.

## 5. Count adjacent ordered pairs

Given tokens:

```text
[b,a,n,a,n,a]
```

Count every adjacent ordered pair, including overlapping candidates:

```text
(b,a): 1
(a,n): 2
(n,a): 2
```

“Ordered” matters: `(a,n)` and `(n,a)` are different candidates. Counting may observe
overlapping candidates; replacement later uses non-overlapping occurrences.

## 6. Choose one pair deterministically

Primary rule: highest frequency. Tie rule in this course: lexicographically smallest numeric
pair. For:

```text
(97,110): 2
(110,97): 2
```

`(97,110)` wins because 97 is smaller than 110.

```python
selected = min(counts, key=lambda pair: (-counts[pair], pair))
```

Negative frequency makes larger counts sort first under `min`. Deterministic tie-breaking is
essential: a different early merge changes later neighbors and can produce a different entire
vocabulary.

## 7. Assign a new ID and bytes

If `(left,right)` receives `new_id`:

```python
vocab[new_id] = vocab[left] + vocab[right]
merges.append(((left, right), new_id))
```

IDs are arbitrary labels. Never calculate the new ID by adding `left + right`; that could
collide with an existing token and does not preserve vocabulary identity.

## 8. Merge non-overlapping occurrences

Scan left to right. On a match, append `new_id` and advance by two. Otherwise preserve the
current token and advance by one. Always guard access to `i+1`, and preserve a final unmatched
token.

```python
result = []
i = 0
while i < len(tokens):
    if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == selected:
        result.append(new_id)
        i += 2
    else:
        result.append(tokens[i])
        i += 1
```

For `[a,a,a]` with pair `(a,a)`, the result is `[new_id,a]`, not two overlapping merges.

## 9. Repeat training

After every replacement, adjacency changes, so recount:

```python
for new_id in available_merge_ids:
    counts = count_pairs(tokens)
    if not counts:
        break
    pair = choose_pair(counts)
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    merges.append((pair, new_id))
    tokens = merge_pair(tokens, pair, new_id)
```

The transparent implementation rescans tokens and costs roughly `O(number_of_merges × tokens)`.
Production trainers maintain occurrence/index structures to update only affected neighborhoods.

## 10. Encoding is not training

To encode new text:

1. apply the saved normalization/pre-tokenization policy
2. encode text to UTF-8 bytes
3. begin with byte IDs
4. replay learned merge priorities deterministically
5. optionally add special tokens

Do not count frequencies or call `choose_pair` on each new sentence. That would make IDs depend
on the input and break the model's fixed embedding vocabulary.

An implementation can replay every rule sequentially (simple but slower) or use a priority
queue/rank table to find the highest-priority applicable merge efficiently.

## 11. Decoding

Ignore or interpret special tokens according to policy, map ordinary IDs to stored byte pieces,
join them, then UTF-8 decode:

```python
raw = b"".join(vocab[token_id] for token_id in ids)
text = raw.decode("utf-8")
```

Core invariant:

```text
decode(encode(text)) == text
```

## 12. What can go wrong

- ID collisions between special tokens and learned merges
- nondeterministic tie-breaking
- deleting unmatched or final tokens during merging
- merging overlapping occurrences incorrectly
- confusing code points, characters, bytes, and grapheme clusters
- training/serving normalization mismatch
- applying merges in a different priority order
- invalid UTF-8 if decoding partial byte sequences incorrectly
- excellent compression on the training domain but severe fragmentation elsewhere
- data leakage or private strings becoming whole memorized vocabulary entries

## 13. Evaluation

Measure more than average token count:

- tokens per byte or bytes per token
- sequence-length distribution and worst-case slices
- multilingual/domain/user-input slices
- percentage of tokens that are rare in training
- embedding/output parameter cost
- training and inference throughput
- time-to-first-token and output-token latency
- downstream quality at matched model/data/compute budgets
- lossless round-trip and malformed-input behavior

Token-level perplexity cannot be compared directly across different tokenizers because the unit
called “one token” changed. Bits per byte is a better common unit for loss comparison.

## 14. Minimum correctness tests

- empty string
- ASCII, accented Latin, multiple scripts, emoji, and combining marks
- every single byte where valid handling is defined
- repeated patterns such as `[a,a,a]`
- no-pair and one-token sequences
- deterministic training under tied counts
- encode/decode after save and reload
- special-token collision checks
- training-domain and out-of-domain compression comparison
