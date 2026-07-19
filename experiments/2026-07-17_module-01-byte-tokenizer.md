# Experiment: Module 1 byte tokenization

## Prediction — complete before execution

- Expected order from fewest to most byte tokens:
- Predicted byte count for `"hello"`: 5.
- Predicted byte count for `"🤖"`: 1 (initially counted the Unicode code point rather than
  its four-byte UTF-8 encoding).
- Why code-point count and byte-token count may differ:

Partial prediction was sufficient to proceed: learner correctly predicted ASCII and exposed
the code-point/byte misconception on emoji; ordering was not provided before execution.

## Controlled design

- Independent variable: input text/script.
- Controls: same UTF-8 byte tokenizer; no BOS/EOS in comparison.
- Metrics: Unicode code points, UTF-8 bytes/tokens, round-trip correctness.
- Slices: ASCII, accented Latin, Devanagari, emoji.

## Results — complete after execution

- Output table:

| Text | Code points | UTF-8 byte tokens | With BOS/EOS |
|---|---:|---:|---:|
| `hello` | 5 | 5 | 7 |
| `naïve café` | 10 | 12 | 14 |
| `नमस्ते` | 6 | 18 | 20 |
| `🤖` | 1 | 4 | 6 |

- Prediction versus observation: `hello=5` was exact. Emoji was predicted as 1 by counting
  code points, but measured 4 UTF-8 bytes.
- Mechanistic explanation: ASCII code points use one UTF-8 byte; accented Latin code points
  shown use two; the Devanagari code points shown use three; the emoji uses four.
- Limitations of byte tokens: sequence length varies strongly across scripts and does not
  capture common multi-byte characters or word pieces as single units.
- Next experiment: learner explains the table, then compare byte tokens with learned pair merges.
