from pathlib import Path

import torch

from llm_lab import ByteTokenizer
from llm_lab.models import TinyTransformer

torch.manual_seed(7)
root = Path(__file__).resolve().parents[2]
tok = ByteTokenizer()
data = torch.tensor(tok.encode((root / "data/tiny.txt").read_text()), dtype=torch.long)
model = TinyTransformer(width=64, heads=4, layers=2, context=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(301):
    starts = torch.randint(0, len(data) - 33, (16,))
    x = torch.stack([data[i : i + 32] for i in starts])
    y = torch.stack([data[i + 1 : i + 33] for i in starts])
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 50 == 0:
        print(f"step={step:3d} loss={loss.item():.4f}")

prompt = torch.tensor([tok.encode("Language")])
print(tok.decode(model.generate(prompt, 80, temperature=0.8)[0].tolist()))

