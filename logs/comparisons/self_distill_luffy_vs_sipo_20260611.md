# Self-Distill Smoke Comparison

Date: 2026-06-11

## Runs

- SIPO run: `SIPO_FULL_REFERENCE_10STEP_LONGCTX_FIX_20260611_120917`
- LUFFY run: `LUFFY_SELF_DISTILL_3STEP_LONGCTX_FIX_20260611_125918`

## Uploaded Artifacts

- `logs/SIPO_FULL_REFERENCE_10STEP_LONGCTX_FIX_20260611_120917_trajectories/step_000001.jsonl` through `step_000009.jsonl`
- `logs/LUFFY_SELF_DISTILL_3STEP_LONGCTX_FIX_20260611_125918_trajectories/step_000001.jsonl` through `step_000002.jsonl`

Each JSONL row is one rollout. `source_idx` is the original batch sample index. `rollout_idx` is the rollout index in the expanded batch. With `n=8` and `n_prefix=1`, the prefixed self-distill rollout is the row with `prefix_len_tokens > 0`, usually the first rollout for each source sample.

## First Two Train Steps

| Run | Off/self-distill reward | On/plain GRPO reward | Off rows | On rows |
| --- | ---: | ---: | ---: | ---: |
| LUFFY | 5/16 = 31.25% | 17/112 = 15.18% | 16 | 112 |
| SIPO | 5/16 = 31.25% | 24/112 = 21.43% | 16 | 112 |

## Text Quality Observations

LUFFY original self-distill prompt ends with longer instruction text:

```text
Now solve the original problem independently in your own words.
Use the reference only to understand the correct reasoning and result.
Your response should be a fresh solution to the original problem, not a summary, quotation, or commentary about the reference.
Assistant: <think>
```

The first LUFFY off-policy sample starts with a meta sentence:

```text
Great! I'll provide a fresh solution to the problem now.
```

It reaches `\boxed{8\pi}`, but then continues into a Python verification block and receives reward 0 in the dump.

SIPO current prompt is shorter and adds a seed:

```text
Now solve the original problem independently in your own words.
Assistant: <think>
Let me solve it again in my own words.
```

The first SIPO off-policy sample starts directly with the solution and receives reward 1.

## Other Signals

| Signal | LUFFY | SIPO |
| --- | ---: | ---: |
| `continuation` starts with `Human:`/`User:`/`Assistant:` | 10/16 | 3/16 |
| self-distill target has `\boxed` | 14/16 | 14/16 |
| self-distill target ends in code fence | 2/16 | 6/16 |
| prefix truncated at max length 2048 | 1/16 | 2/16 |

## Interpretation

Both runs successfully route online self-distill outputs through the LUFFY off-policy prefix machinery. The reward rate for prefixed samples is identical over the first two train steps, but SIPO's prompt produces cleaner starts and fewer chat-template leaks in the continuation. The main remaining issue is not EOS handling; it is self-distill target quality. A practical next step is to filter self-distill targets before using them as prefixes, ideally keeping only reward-verified or final-answer-extractable trajectories.
