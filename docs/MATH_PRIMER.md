# Minimum math primer

Focus on operational understanding: calculate small examples, then connect them to code.

- Linear algebra: vectors, matrices, dot products, matrix multiplication, transpose,
  rank, basis, norms, and eigensystem intuition.
- Probability: random variables, conditional probability, expectation, categorical
  distributions, maximum likelihood, entropy, cross-entropy, and KL divergence.
- Calculus: derivatives, partial derivatives, gradients, chain rule, Jacobian-vector
  products, and why reverse-mode differentiation is efficient for scalar losses.
- Optimization: convexity intuition, gradient descent, stochastic estimates, momentum,
  adaptive moments, regularization, and learning-rate schedules.

Core language-model objective for tokens `x_1 ... x_T`:

```text
p(x_1...x_T) = product_t p(x_t | x_<t)
loss = -(1/T) sum_t log p_model(x_t | x_<t)
```

Attention for queries `Q`, keys `K`, values `V`, and head width `d`:

```text
Attention(Q,K,V) = softmax(Q K^T / sqrt(d) + causal_mask) V
```

LoRA freezes `W` and learns a low-rank update:

```text
y = x (W + scale * B A)^T,  rank(A), rank(B) <= r
```

