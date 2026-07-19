# Time estimate and public portfolio plan

The goal is not merely to finish lessons. It is to produce public evidence that you can
form hypotheses, implement internals, run controlled experiments, profile systems, and
communicate engineering tradeoffs.

## Realistic completion estimates

| Weekly time | Core curriculum | Portfolio-ready capstone | Sustainable target |
|---:|---:|---:|---|
| 5 hours | 8–10 months | 10–12 months | one small lab every 1–2 weeks |
| 8–10 hours | 4–5 months | 5–7 months | recommended alongside a job |
| 15 hours | 12–16 weeks | 4–5 months | demanding but realistic |
| 25+ hours | 8–10 weeks | 12–14 weeks | full-time study; high burnout risk |

These estimates assume the completion standard in `AGENTS.md`: derivation, implementation,
tests, a controlled experiment, profiling, and explanation. Simply running the existing
scripts would take days but would not create strong interview evidence.

Recommended plan: **24 weeks at 8–10 hours/week**, approximately 200–240 focused hours.
Add 20% schedule margin for debugging and writing.

## 24-week execution schedule

| Weeks | Technical outcome | Public artifact |
|---|---|---|
| 1–2 | tensors, dtypes, devices, reproducibility, profiling | benchmark report with memory calculations |
| 3–4 | Unicode, byte tokenizer, BPE | tokenizer comparison and multilingual failure analysis |
| 5–6 | autograd, SGD, AdamW, stability | from-scratch autograd implementation and optimizer curves |
| 7–8 | bigram LM, batching, sampling, validation | baseline report with controlled sampling experiment |
| 9–11 | decoder Transformer internals | annotated implementation and causal-leakage experiment |
| 12–14 | pretraining pipeline and systems | resumable trainer, profiler trace, cost-to-quality analysis |
| 15–16 | SFT and loss masking | chat-template/masking tests and base-vs-SFT evaluation |
| 17–18 | LoRA and quantized tuning | rank/target-module ablation and memory comparison |
| 19–20 | preference tuning | DPO-style loss, beta experiment, bias analysis |
| 21 | evaluation and statistics | frozen eval manifest, confidence intervals, error taxonomy |
| 22 | KV cache and inference | cached equivalence tests and latency/throughput benchmark |
| 23–24 | systems plan and capstone synthesis | model card, technical report, demo, architecture explanation |

## Weekly rhythm

- Session 1, 90 minutes: derive concepts and write predictions.
- Session 2, 2 hours: implement the smallest correct mechanism and tests.
- Session 3, 2 hours: run controlled experiments across seeds.
- Session 4, 90 minutes: profile time/memory and analyze failures.
- Session 5, 1–2 hours: write the public report, clean code, and update progress.

Do not publish a noisy commit after every keystroke. Commit coherent checkpoints frequently,
then publish one polished experiment report per week or two.

## Public GitHub strategy

### Repository positioning

Use a descriptive public name such as `llm-training-systems-lab`. The first screen of the
README should answer:

1. What did you implement yourself?
2. Which experiments did you run?
3. What measurable results did you obtain?
4. What can a reviewer run in five minutes?
5. What are the limitations?

Avoid presenting scaffolded code as completed learner work. The progress ledger must clearly
distinguish `Scaffolded` from `Complete`.

### Evidence that attracts technical interviews

Prioritize evidence over certificates:

- correct from-scratch implementations with focused tests
- experiment tables with multiple seeds and uncertainty
- causal leakage and numerical-stability debugging stories
- memory/FLOPs estimates compared with measurements
- cost-to-quality comparisons, not only final loss
- reproducible configs, hardware disclosure, and exact commands
- readable failure analyses, including hypotheses that were wrong
- one polished capstone rather than many unfinished notebooks

### Recommended public artifacts

- `README.md`: concise project map, current highlights, quickstart, selected results
- `docs/PROGRESS.md`: transparent roadmap and status
- `experiments/*.md`: research notes using the experiment template
- `reports/capstone.md`: final technical report
- `MODEL_CARD.md`: intended use, data, evaluations, limitations, safety, license
- tagged releases: `v0.1-foundations`, `v0.2-transformer`, `v0.3-tuning`, `v1.0-capstone`
- GitHub Actions: tests and lint on each push/PR

### Commit and pull-request practice

Write commits that communicate engineering work:

```text
feat(attention): implement causal multi-head attention
test(masking): prove future-token invariance
experiment(norm): compare pre-norm and post-norm across three seeds
perf(training): add synchronized step-time and peak-memory metrics
docs(results): analyze LoRA rank and effective update rank
```

For major modules, work on a short branch and open a pull request even when working alone.
The PR description should state the question, design, key result, tests, and limitations.
This demonstrates review-ready communication and preserves the reasoning behind changes.

## Interview conversion plan

By Week 8, prepare a 60-second explanation of the project and begin sharing individual
technical findings. By Week 14, start applying selectively; do not wait for every module.
By Week 20, lead with the capstone-in-progress and systems results.

Prepare three stories using problem → investigation → evidence → tradeoff → outcome:

1. a correctness bug, ideally causal leakage or loss-mask misalignment
2. a performance bottleneck found through profiling
3. an experiment whose result contradicted your hypothesis

Prepare whiteboard explanations for attention tensor shapes, cross-entropy, backpropagation,
AdamW, LoRA, mixed precision, KV caching, and distributed-memory tradeoffs.

## Quality gates before making the repository public

- no credentials, private data, proprietary text, large checkpoints, or generated run dumps
- clear license for code and documented licenses for data/model dependencies
- tests pass from a clean environment
- setup and five-minute smoke test are verified
- every claimed result points to a tracked experiment report
- hardware and measurement method are disclosed
- unfinished work is labeled accurately
- spelling and diagrams are polished enough for an unfamiliar reviewer

## Monthly review

At the end of every four weeks, answer:

- What can I now implement without help?
- Which result is strongest evidence of engineering judgment?
- Where are my measurements weak or irreproducible?
- Which artifact would a hiring manager understand in 90 seconds?
- What should be removed or simplified?
- Should the next month emphasize theory, implementation, systems, or communication?

