# LUFFY Default vs SIPO Default Case Study

Date: 2026-06-11

## Runs

- LUFFY default run: `LUFFY_DATASET_TARGET_10STEP_DEFAULT_20260611_132407`
- SIPO default-matched run: `SIPO_FULL_REFERENCE_2STEP_LUFFY_DEFAULT_20260611_133711`

Both runs use the same core LUFFY training hyperparameters:

| Setting | Value |
| --- | ---: |
| `data.train_batch_size` | 128 |
| `data.max_prompt_length` | 1024 |
| `data.max_response_length` | 8192 |
| `actor_rollout_ref.rollout.n` | 8 |
| `actor_rollout_ref.rollout.n_prefix` | 1 |
| `actor_rollout_ref.rollout.min_prefix_ratio` | 1.0 |
| `actor_rollout_ref.rollout.max_prefix_ratio` | 1.0 |
| `actor_rollout_ref.actor.use_off_policy_loss` | true |

The controlled difference is the source of the off-policy prefix:

- LUFFY: dataset target is used directly as `tgt_input_ids`.
- SIPO: online self-distill output is used as `tgt_input_ids`.

## Source Artifacts

- `logs/LUFFY_DATASET_TARGET_10STEP_DEFAULT_20260611_132407_trajectories/step_000001.jsonl`
- `logs/LUFFY_DATASET_TARGET_10STEP_DEFAULT_20260611_132407_trajectories/step_000002.jsonl`
- `logs/SIPO_FULL_REFERENCE_2STEP_LUFFY_DEFAULT_20260611_133711_self_distill/step_000001.jsonl`

Note: the SIPO prefix trajectory dump file was not emitted for this run, but the
self-distill dump contains the needed rollout fields:
`rollout_prefix_text`, `rollout_full_response`, `reward`, and
`prefix_len_tokens`.

## Aggregate Comparison

| Run | Rows | Prefixed reward | On-policy reward | Total reward | Prefix length mean / max | Prefixed boxed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LUFFY step 1 | 1024 | 128/128 = 100.0% | 250/896 = 27.9% | 378/1024 = 36.9% | 4211 / 7841 | 128/128 |
| LUFFY step 2 | 1024 | 128/128 = 100.0% | 237/896 = 26.5% | 365/1024 = 35.6% | 3927 / 7894 | 128/128 |
| SIPO step 1 | 1024 | 57/128 = 44.5% | 273/896 = 30.5% | 330/1024 = 32.2% | 709 / 4097 | 109/128 |

## Main Finding

With the same LUFFY training hyperparameters and the same off-policy prefix
machinery, LUFFY's prefixed rows are reward-1 because the prefix is the dataset
target. SIPO's prefixed rows are only 44.5% reward-1 because the prefix is an
online self-distill generation, and that generation can be wrong, malformed, or
contradictory.

SIPO has slightly higher on-policy reward in this step:

- LUFFY step 1 on-policy: 27.9%
- LUFFY step 2 on-policy: 26.5%
- SIPO step 1 on-policy: 30.5%

But SIPO's off-policy prefix reward is much lower than LUFFY's because the
self-distilled target is not guaranteed correct.

## Case Study 1: LUFFY Dataset Target Prefix

Source index: 0

Reward: 1.0

Prefix tokens: 7575

LUFFY prefix tail:

```text
For point X on AB, solving the circle equation gives X at (5, 5\sqrt{3}).
AX = 10 and AB = 20, so AX \cdot AB = 200.

For point Y on BC, solving the circle equation gives Y at (15, 5\sqrt{3}).
CY = 10 and BC = 20, so CY \cdot BC = 200.

Summing these products gives AX \cdot AB + CY \cdot BC = 200 + 200 = 400.

Thus, the final answer is \( \boxed{400} \).
```

Interpretation:

LUFFY receives the dataset target as the prefix. The target is long, complete,
and ends with the correct boxed answer. This is why the prefixed row receives
reward 1.0. In step 1 and step 2, every LUFFY prefixed row has this same
property: all 128/128 prefixed rows receive reward 1.0.

## Case Study 2: SIPO Self-Distill Prefix, Correct

Source index: 4

Ground truth: `-12`

Reward: 1.0

Prefix tokens: 189

SIPO prefix tail:

```text
Given that \(x^2 - 6x + 4a\) and \(x^2 + ax + 6\) have real roots and sum of
squares of roots are equal, we get:
\[ 36 - 8a = a^2 - 12 \implies a^2 + 8a - 48 = 0. \]
Solve this:
\[ a = 4, -12. \]
Discriminant check: \( a^2 + 8a - 48 > 0 \implies a < -12 \text{ or } a > 4 \).
Since we need two real roots and \(a = 4\) gives discriminant \(-28\), invalid.
So, \(a = -12\) is valid.

The only valid value for \(a\) is \( \boxed{-12} \).
```

Interpretation:

This is the intended SIPO behavior. The self-distill generation is short,
mathematically coherent, and ends with the correct boxed answer. It then works
as an off-policy prefix through the same LUFFY prefix path.

## Case Study 3: SIPO Self-Distill Prefix, Contradictory Final Answer

Source index: 0

Ground truth: `400`

Reward: 0.0

Prefix tokens: 659

SIPO prefix tail:

```text
Calculate AX: distance from A(0,0) to X(5,5√3): sqrt(25 + 75) = 10.
AB is 20. So, AX·AB = 10·20 = 200.

Calculate CY: distance from C(20,0) to Y(15,5√3): sqrt(25 + 75) = 10.
BC is 20. So, CY·BC = 10·20 = 200.

Therefore, AX·AB + CY·BC = 200 + 200 = 400.

So the answer is 400.
</think>
The final answer is \( \boxed{200} \).
```

Interpretation:

The reasoning reaches the correct value, 400, but the final boxed answer is
200. The reward function follows the final extracted answer, so this self-
distill prefix receives reward 0.0. This is not an off-policy prefix plumbing
issue; it is a self-distill target quality issue.

## Case Study 4: SIPO Self-Distill Prefix, Malformed Output

Source index: 3

Ground truth: `B`

Reward: 0.0

Prefix tokens: 536

SIPO prefix tail:

```text
(: must Determine graph closed correctly; way). an a list in brackets its thing
ue of first. into Learning between us numb the rors andn' for Sees alg correctly
...
</think>
Thisgraph {,}:
[ f( has[ 1, )² worst What will term ifn×²benef
```

Interpretation:

This self-distill sample is malformed and does not contain a usable final
answer. It becomes an off-policy prefix, but it cannot provide a reward-1
target. These cases explain why SIPO has only 109/128 prefixed rows with a
boxed answer and 57/128 reward-1 prefixed rows.

## Conclusion

The default-matched comparison separates the mechanism from the target source:

- The LUFFY/SIPO off-policy prefix path is aligned.
- LUFFY gets oracle-like dataset target prefixes, so prefixed reward is 100%.
- SIPO gets model-generated self-distill prefixes, so prefixed reward depends
  on generation quality.

For improving SIPO, the most direct next target is self-distill quality control:
reject malformed outputs, reject contradictory final boxed answers, or regenerate
when the self-distill reward is 0 before using the text as an off-policy prefix.
