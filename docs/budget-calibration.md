# Calibrating the budget model against ST's published measurements

`zoo/graph/budget.py` computes peak activation analytically: a liveness
maximum over inferred shapes. It cannot know what the real allocator will do,
and the standing warning is that a prior project's analytic 1.95 MB became
"116 MB unallocatable" in practice. So the question is not whether the model is
exact — it is not — but whether it is *close enough to be worth screening on*.

ST publishes measured internal-activation figures for a set of models on the
STM32N6570-DK, quantised to int8. The zoo currently screens fp32 graphs, so the
comparison is `analytic fp32 peak ÷ 4` against ST's measured int8 figure.

| model | zoo, fp32 peak | ÷4 | ST measured, int8 | delta |
|---|---:|---:|---:|---:|
| handpose_estimation_mediapipe @224 | 7,056 KB | 1,764 KB | 1,740 KB | **+1.4 %** |
| mobilenet_v2_1.0 @224 | 9,408 KB | 2,352 KB | 2,058 KB | +14 % |
| face_detection_yunet @320 (from @640 ÷4) | 12,800 KB | 800 KB | 1,130 KB | −29 % |

The first two land close. The third is the interesting one: it is not a direct
comparison, because the zoo screened the 640×640 export and ST measured a
320×320 one, so the row assumes activation scales exactly with pixel count.
It does not quite — the ratio is dominated by the first convolution but the
detection heads do not shrink proportionally — and a 29 % under-estimate is the
size of error that assumption buys.

## What to take from this

**Good enough to screen on, not good enough to promote on.** An order-of-
magnitude answer is exactly what the budget stage is for: it should say "this
is nowhere near fitting" or "this is worth compiling", and both of those
survive a 15–30 % error. What it must never do is convert a near-miss into a
verdict.

Two rules follow, both already enforced in code:

- `placement()` refuses to answer at all when any tensor's shape failed to
  resolve, returning `unknown-unpinned-shapes`. An under-count over a partial
  graph is indistinguishable from a comfortable fit.
- The compile stage's figure, read from `network_c_info.json`, always
  overrides the estimate in the leaderboard. The estimate is a filter; the
  compiler is the authority.

## Re-check this

When the quantisation stage lands, re-run these three models as int8 and
compare directly rather than through a ÷4 approximation. If the direct
comparison is worse than the approximate one, the fault is in the QDQ-fused
liveness path, not in the scaling assumption.
