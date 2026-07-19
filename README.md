# LLM Training & Tuning Lab

A learn-by-building course that starts with tensors and ends with training, tuning,
evaluating, and serving a small language model. The early labs avoid high-level
frameworks so the important internals stay visible.

## What you will build

- a byte-level tokenizer
- a bigram language model and training loop
- a decoder-only Transformer (attention, MLP, residual stream, normalization)
- a reproducible pretraining pipeline with checkpoints
- supervised fine-tuning (SFT) data formatting and loss masking
- LoRA adapters implemented from first principles
- preference optimization concepts and a tiny DPO-style objective
- evaluation, sampling, KV-cache, quantization, and scaling experiments
- a capstone model card and experiment report

## Start here

Requirements: Python 3.11+; a GPU is optional for Modules 0–4.

```bash
cd llm-training-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python labs/01_tokenization/tokenizer_lab.py
python labs/03_language_modeling/train_bigram.py
```

Read [the learning plan](docs/LEARNING_PLAN.md), choose investigations from the
[what-if experiment catalog](docs/WHAT_IF_EXPERIMENTS.md), and keep an experiment
journal using [the template](docs/EXPERIMENT_TEMPLATE.md). Study low-level performance
and economics with the [systems and cost optimization track](docs/SYSTEMS_COST_OPTIMIZATION.md).
Future sessions resume from [the progress ledger](docs/PROGRESS.md).

For the realistic time commitment, public GitHub milestones, and interview strategy,
see the [time and portfolio plan](docs/TIME_AND_PORTFOLIO_PLAN.md).

## Repository map

```text
docs/             curriculum, glossary, math, experiment template
labs/             ordered practical modules
src/llm_lab/      reusable implementations
tests/            fast correctness tests
configs/          experiment configurations
data/             tiny included learning corpus only
```

## Learning rule

For every experiment: make a falsifiable prediction, change one variable, use at
least three seeds for noisy comparisons, record compute and quality metrics, test a
competing explanation, and only then move on. Small models are a microscope, not a
toy: the same mechanisms appear in large systems.
