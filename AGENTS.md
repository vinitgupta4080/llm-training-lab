# Project instructions: LLM Training & Tuning Lab

This file is durable project memory for every future assistant/session working in this
repository. Read this file first, then read `docs/PROGRESS.md` before making changes or
choosing the next lesson. Conversation history is not the source of truth.

## Mission

Teach LLM training and tuning from first principles through controlled experiments.
The learner should be able to derive, implement, profile, debug, and explain each
mechanism—not merely call a high-level library.

## How to resume after compaction or a new session

1. Read `AGENTS.md` completely.
2. Read `docs/PROGRESS.md` and locate `Current module` and `Next action`.
3. Read the current module in `docs/LEARNING_PLAN.md`.
4. Read the corresponding questions in `docs/WHAT_IF_EXPERIMENTS.md` and systems topics
   in `docs/SYSTEMS_COST_OPTIMIZATION.md`.
5. Inspect the latest file in `experiments/` if one exists.
6. Run `python3 -m pytest -q` before changing code.
7. Continue the explicit `Next action`; do not restart the course or skip ahead silently.

## Teaching protocol

- Ask the learner for a prediction before revealing an experimental result when working
  interactively.
- Introduce every new concept in this order: why it is needed, how it is used with a concrete
  example, then its technical mechanism and edge cases. Do not lead with rules or notation.
- Before incremental coding TODOs, explain the complete end-to-end algorithm and how every
  function participates. Then implement one function at a time with assertions.
- Do not require concepts from future modules to explain the current module. Label forward
  references as optional previews and give a prerequisite-free explanation first.
- Prefer a tiny transparent implementation before a framework abstraction.
- Trace tensor shapes, dtypes, devices, parameter counts, memory, FLOPs, and tokens.
- Change one independent variable per experiment.
- Use at least three seeds for noisy comparisons.
- Compare runs by consumed tokens and compute, not only optimizer steps.
- Distinguish measured results, estimates, hypotheses, and outside knowledge.
- Require a falsifier and competing explanation for major claims.
- Include correctness, speed, memory, cost, and quality tradeoffs.
- Never claim a small-model result automatically generalizes to production-scale LLMs.
- For every new lesson or substantial concept explanation, create or update a compact
  browser-based revision artifact in the thread visualization directory. It should help the
  learner revise through interaction, diagrams, calculations, or self-checks; record its
  title and covered concepts in `docs/PROGRESS.md`.
- At each module checkpoint, consolidate lesson visuals and theory into a standalone HTML file
  under `revision/` so it is tracked by Git and publishable in the public repository.

## Progress tracking rules

After meaningful course work, update `docs/PROGRESS.md` in the same change:

- mark completed objectives only when supported by code, test, or experiment evidence
- add links/paths to evidence
- record misconceptions or unresolved questions
- set exactly one concrete `Next action`
- append a dated session entry

For each experiment, copy `docs/EXPERIMENT_TEMPLATE.md` to
`experiments/YYYY-MM-DD_short-name.md`. Store machine-readable metrics under
`runs/<experiment-name>/`; `runs/` is intentionally gitignored because it may be large.
Summaries and important small tables belong in the experiment note and should be tracked.

## Completion standard

A module is complete only when the learner can:

1. derive or explain its central equations
2. implement the mechanism or its essential core
3. pass correctness tests, including a failure case
4. run one controlled what-if experiment
5. profile time and memory
6. explain observed behavior and limitations in plain language

## Engineering constraints

- Keep CPU-friendly defaults for Modules 0–4.
- Place reusable logic in `src/llm_lab/`, labs in `labs/`, and tests in `tests/`.
- Avoid adding large datasets, checkpoints, generated runs, or secrets to Git.
- Make benchmark hardware, dtype, shapes, warmup, synchronization, and repetitions explicit.
- Preserve existing learner work and unrelated changes.

## GitHub publishing identity

- Publish this repository only through GitHub account `vinitgupta4080`.
- Never use, authenticate as, push through, or publish through `vinitgupta-alation`.
- Prefer the signed-in `vinitgupta4080` GitHub web interface. If any tool resolves to another
  account, stop that publishing path without changing stored credentials.
