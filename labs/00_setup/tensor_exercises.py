"""Module 0 learner exercise. Replace every TODO without changing the assertions."""

import torch


# 1. Create a float32 tensor filled with zeros with shape [2, 3, 4].
hidden = torch.zeros((2, 3, 4), dtype=torch.float32)
assert hidden.shape == (2, 3, 4)
assert hidden.dtype == torch.float32

# 2. Calculate its rank, element count, and storage bytes using tensor methods.
rank = hidden.ndim
elements = hidden.numel()
storage_bytes = hidden.element_size() * elements
assert rank == 3
assert elements == 24
assert storage_bytes == 96

# 3. Record its contiguous stride.
hidden_stride = hidden.stride()
assert hidden_stride == (12, 4, 1)

# 4. Create a four-value bias vector [10, 20, 30, 40].
bias = torch.tensor([10, 20, 30, 40], dtype=torch.float32)
assert bias.shape == (4,)

# 5. Add the bias to every token using broadcasting—do not repeat or copy it manually.
shifted = hidden + bias
assert shifted.shape == (2, 3, 4)
torch.testing.assert_close(shifted[1, 2], bias)

# 6. Expand the bias logically to [2, 3, 4] and inspect its stride.
expanded_bias = bias.expand_as(hidden)
assert expanded_bias.shape == (2, 3, 4)
assert expanded_bias.stride() == (0, 0, 1)
assert expanded_bias.untyped_storage().data_ptr() == bias.untyped_storage().data_ptr()

# 7. Transpose token and channel axes without copying storage.
transposed = hidden.transpose(1, 2)
assert transposed.shape == (2, 4, 3)
assert not transposed.is_contiguous()
assert transposed.untyped_storage().data_ptr() == hidden.untyped_storage().data_ptr()

# 8. Reset the same seed before two random tensors and make them identical.
torch.manual_seed(0)
first = torch.randn(5)
torch.manual_seed(0)
second = torch.randn(5)
assert torch.equal(first, second)

print("Module 0 tensor exercises passed.")
