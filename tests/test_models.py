import torch
from torch import nn

from llm_lab.lora import LoRALinear
from llm_lab.models import TinyTransformer


def test_transformer_shapes_and_loss():
    model = TinyTransformer(vocab_size=20, context=8, width=16, heads=4, layers=2)
    tokens = torch.randint(0, 20, (3, 8))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (3, 8, 20)
    assert loss.ndim == 0


def test_causal_mask_prevents_future_influence():
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=20, context=8, width=16, heads=4, layers=2).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 9, 9]])
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2])


def test_lora_starts_equivalent_and_base_is_frozen():
    base = nn.Linear(5, 3)
    lora = LoRALinear(base, rank=2)
    x = torch.randn(4, 5)
    torch.testing.assert_close(lora(x), base(x))
    assert not base.weight.requires_grad
    lora(x).sum().backward()
    assert lora.a.grad is not None and lora.b.grad is not None

