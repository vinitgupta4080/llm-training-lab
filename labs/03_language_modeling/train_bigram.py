from pathlib import Path

import torch

from llm_lab import ByteTokenizer
from llm_lab.models import BigramLanguageModel

torch.manual_seed(7)
root = Path(__file__).resolve().parents[2]
tok = ByteTokenizer()
data = torch.tensor(tok.encode((root / "data/tiny.txt").read_text()), dtype=torch.long)
model = BigramLanguageModel(tok.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)

for step in range(201):
    starts = torch.randint(0, len(data) - 9, (32,))
    x = torch.stack([data[i : i + 8] for i in starts])
    y = torch.stack([data[i + 1 : i + 9] for i in starts])
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step={step:3d} loss={loss.item():.4f}")

print("Exercise: add a validation split, generation, and an overfitting diagnosis.")

